import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fftpack import dct

class MCDLoss(nn.Module):
    """
    Mel Cepstral Distortion (MCD) Loss for batched inputs.
    Computes spectral envelope difference between predicted and ground truth spectrograms.
    
    Inputs:
    - mel_true: Ground truth Mel spectrograms, shape (B, D, T)
    - mel_pred: Predicted Mel spectrograms, shape (B, D, T)
    
    Output:
    - Scalar MCD loss
    """
    def __init__(self):
        super(MCDLoss, self).__init__()
        self.log_base_10 = 10 / torch.log(torch.tensor(10.0))
        
    def log_mel_to_cepstral(self, log_mel_spec, num_ceps=13):
        """
        Convert log-Mel spectrogram to cepstral coefficients using DCT, then apply mean-variance normalization.
        
        Args:
            log_mel_spec: Tensor (B, 1, F, T) - Raw Mel spectrogram.
            num_ceps: Number of cepstral coefficients to keep (default=13).
        
        Returns:
            cepstral_coeffs: Tensor (B, num_ceps, T) - Cepstral coefficients (CMVN applied).
        """
        B, _, F, T = log_mel_spec.shape  # Extract batch size, frequency, time dims
        
        # 1) Apply log compression to handle negatives
        log_mel_spec = torch.log1p(torch.abs(log_mel_spec))  # log(1 + |S|)
    
        # 2) Flatten batch for DCT transform
        log_mel_spec_cpu = log_mel_spec.squeeze(1).detach().cpu().numpy()  # (B, F, T)
    
        # 3) DCT along frequency axis => shape still (B, F, T)
        cepstral_coeffs = dct(log_mel_spec_cpu, type=2, axis=1, norm='ortho')  # (B, F, T) in numpy
    
        # 4) Keep only the first `num_ceps` coefficients => shape (B, num_ceps, T)
        cepstral_coeffs = cepstral_coeffs[:, :num_ceps, :]
    
        # Convert back to torch
        cepstral_coeffs = torch.tensor(cepstral_coeffs, dtype=torch.float32)  # (B, num_ceps, T)
    
        # 5) Mean-Variance Normalization (across batch & time for each dimension)
        #    Flatten (B, D, T) => (B*T, D)
        B_, D_, T_ = cepstral_coeffs.shape
        flattened = cepstral_coeffs.permute(0, 2, 1).reshape(B_ * T_, D_)  # (B*T, D)
    
        # 5a) Compute mean, std for each dimension (D)
        mean = flattened.mean(dim=0, keepdim=True)  # (1, D)
        std = flattened.std(dim=0, keepdim=True) + 1e-6  # (1, D), add epsilon to avoid /0
    
        # 5b) Normalize
        flattened = (flattened - mean) / std
    
        # 5c) Reshape back to (B, D, T)
        cepstral_coeffs = flattened.reshape(B_, T_, D_).permute(0, 2, 1)  # (B, D, T)
    
        return cepstral_coeffs
    
    def forward(self, mel_true, mel_pred):
        """
        Compute MCD loss between batched spectrograms, converting each to cepstral,
        applying CMVN, then computing MCD.
        
        Args:
            mel_true: Tensor (B, 1, F, T) - Ground truth log-Mel spectrogram
            mel_pred: Tensor (B, 1, F, T) - Predicted log-Mel spectrogram
        
        Returns:
            Scalar loss value (averaged over batch and time).
        """
        # 1) Convert log-mel to CMVN cepstral
        cepstral_true = self.log_mel_to_cepstral(mel_true)  # (B, D, T)
        cepstral_pred = self.log_mel_to_cepstral(mel_pred)  # (B, D, T)
        assert cepstral_true.shape == cepstral_pred.shape, "Mismatch in cepstral shapes"

        # 2) Compute difference, then L2 norm
        diff = cepstral_true - cepstral_pred  # (B, D, T)
        mse_dist = torch.sum(diff**2, dim=1)  # Sum over cepstral dims => (B, T)

        # 3) Apply MCD formula => shape (B, T)
        mcd = self.log_base_10 * torch.sqrt(2 * mse_dist)  # (B, T)

        # 4) Average over batch and time => scalar
        return torch.mean(mcd)



def spatial_smoothness_loss(phi, weight=1.0):
    """
    Computes isotropic gradient penalty across F and T dims.
    """
    dphi_dt = phi[:, :, :, 1:] - phi[:, :, :, :-1]  # along T
    dphi_df = phi[:, :, 1:, :] - phi[:, :, :-1, :]  # along F

    loss_t = dphi_dt.pow(2).mean()
    loss_f = dphi_df.pow(2).mean()

    return weight * (loss_t + loss_f)


def general_sparsity_loss(phi, weight=1.0):
    """
    L1 regularization encouraging sparse field values.
    phi: Tensor [B, C, F, T]
    """
    return weight * phi.abs().mean()


def sparsity_loss(pred_fields, field_names, selected_fields, weight=1.0):
    """
    Apply L1 sparsity penalty only to selected fields.
    
    Args:
        phi_pred: Tensor of shape [B, C, F, T]
        field_names: List of C field names (e.g., ['t_stretch', 'f_stretch', ...])
        selected_fields: List of fields to apply sparsity to
        weight: scaling factor for loss

    Returns:
        scalar tensor loss
    """
    loss = 0.0
    for i, name in enumerate(field_names):
        if name in selected_fields:
            loss += pred_fields[:, i, :, :].abs().mean()
    return weight * loss



def cosine_field_loss(pred, target, eps=1e-8):
    # Flatten (F, T) into one dimension: [B, C, F*T]
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)

    # Cosine similarity: [B, C]
    cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=2, eps=eps)
    
    # weights = torch.tensor([1.0, 1.0, 2.0, 2.0, 0.5])  # emphasize warp, de-emphasize amp
    # weighted_cosine = F.cosine_similarity(pred_flat * weights, target_flat * weights, dim=2, eps=eps)

    # Turn into loss (1 - sim), then average over batch & channels
    cos_loss = 1 - cos_sim
    return cos_loss.mean()


def phi4_loss_fn(pred, target):
    return torch.mean((pred - target)**4)


def hat_loss_fn(phi_pred, phi_true, v=0.3, field_threshold=1e-4, 
                lambda_pot=1, lambda_align=1.0, lambda_bg=1.0):
    """
    Hybrid field-theoretic loss:
    - Potential term encourages ||phi_pred|| ≈ v in active regions
    - Alignment term encourages phi_pred ≈ phi_true where signal is present
    - Background term suppresses phi_pred where signal is absent
    """

    # Compute mask: [B, 1, F, T]
    mask = (phi_true.abs().sum(dim=1, keepdim=True) > field_threshold).float()  # Sum over field channels

    # [B, F, T] --> [B, 1, F, T] for broadcasting
    phi_norm = phi_pred.norm(dim=1, keepdim=True)

    # Loss terms: shape [B, 1, F, T]
    potential = ((phi_norm - v)**2) * mask
    alignment = ((phi_pred - phi_true)**2) * mask
    background = (phi_pred**2) * (1 - mask)

    # Mean over all dimensions
    loss = (
        lambda_pot * potential.mean() +
        lambda_align * alignment.mean() +
        lambda_bg * background.mean()
    )

    return loss





class GaussianConv2D(nn.Module):
    def __init__(self, channels, kernel_size=5, sigma=1.0):
        super().__init__()
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=self.padding, groups=channels, bias=False)
        self.init_weights(channels, kernel_size, sigma)

    def init_weights(self, channels, kernel_size, sigma):
        coords = torch.arange(kernel_size)
        grid = coords - kernel_size // 2
        yy, xx = torch.meshgrid(grid, grid, indexing='ij')
        gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        gaussian /= gaussian.sum()
        kernel = gaussian.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        self.conv.weight.data.copy_(kernel)
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)

def convolutional_kernel_loss(pred, target, conv_layer):
    pred_blur = conv_layer(pred)
    target_blur = conv_layer(target)
    return F.mse_loss(pred_blur, target_blur)
