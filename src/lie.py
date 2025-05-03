import torch
import math
import random

from pprint import pprint as pp

from torchvision.utils import make_grid
from copy import deepcopy
import torch.nn.functional as F

def localized_smooth_field_1d(length,
                              num_blobs=2,
                              mask_radius_frac=0.1,
                              epsilon=0.2,
                              device='cpu'):
    """
    Create a 1D field composed of a few localized sinusoidal blobs.
    Each blob is sinusoidal but confined to a local region via a soft mask.
    Output: Tensor of shape [length]
    """
    t = torch.linspace(0, 1, length, device=device)
    field = torch.zeros_like(t)

    for _ in range(num_blobs):
        # Sinusoid parameters
        freq = torch.rand(1).item() * 1.5 + 0.5  # 0.5 to 2.0 cycles
        phase = torch.rand(1).item() * 2 * math.pi
        amplitude = torch.randn(1).item() * epsilon

        # Sinusoid
        wave = amplitude * torch.sin(2 * math.pi * freq * t + phase)

        # Soft Gaussian mask
        center = torch.randint(int(length * 0.2), int(length * 0.8), (1,)).item()
        sigma = int(length * mask_radius_frac)
        coords = torch.arange(length, device=device)
        mask = torch.exp(-((coords - center) ** 2) / (2 * sigma ** 2))

        # Apply masked blob
        field += wave * mask

    return field



def localized_smooth_field_2d(num_freq_bins, num_time_steps,
                              num_blobs=2,
                              mask_radius_frac=0.1,
                              epsilon=0.3,
                              device='cpu'):
    """
    Create a sparse 2D field with a few local sinusoidal blobs.
    Each blob has its own soft mask and sinusoidal deformation.
    Output: Tensor of shape [F, T]
    """
    F, T = num_freq_bins, num_time_steps
    field = torch.zeros(F, T, device=device)

    # Create meshgrid directly in [F, T] format
    f = torch.linspace(0, 1, F, device=device)
    t = torch.linspace(0, 1, T, device=device)
    ff, tt = torch.meshgrid(f, t, indexing='ij')  # [F, T]

    for _ in range(num_blobs):
        # Random sinusoid parameters
        freq_t = torch.rand(1).item() * 3 + 1
        freq_f = torch.rand(1).item() * 3 + 1
        phase_t = torch.rand(1).item() * 2 * math.pi
        phase_f = torch.rand(1).item() * 2 * math.pi
        amplitude = torch.randn(1).item() * epsilon

        # Sinusoidal wave in [F, T]
        wave = amplitude * torch.sin(2 * math.pi * freq_t * tt + phase_t) * \
                             torch.sin(2 * math.pi * freq_f * ff + phase_f)

        # Localized soft mask
        center_f = torch.randint(int(F * 0.2), int(F * 0.8), (1,)).item()
        center_t = torch.randint(int(T * 0.2), int(T * 0.8), (1,)).item()
        sigma_f = int(F * mask_radius_frac)
        sigma_t = int(T * mask_radius_frac)

        f_coords = torch.arange(F, device=device).unsqueeze(1)  # [F, 1]
        t_coords = torch.arange(T, device=device).unsqueeze(0)  # [1, T]

        mask = torch.exp(-((f_coords - center_f) ** 2) / (2 * sigma_f ** 2)) * \
               torch.exp(-((t_coords - center_t) ** 2) / (2 * sigma_t ** 2))  # [F, T]

        # Apply masked blob
        field += wave * mask

    return field





def generate_lie_generator_fields(num_freq_bins, num_time_steps, epsilon_dict=None, device='cpu'):
    """
    Returns a dict of all 5 Lie generator fields, each of shape [F, T]
    """
    if epsilon_dict is None:
        epsilon_dict = {
            't_stretch': 0.05,
            'f_stretch': 0.05,
            'warp_2d': 0.05,
            'amplitude': 0.1,
            'phase': 0.1,
        }

    # 1. Time stretch: v(t) → broadcast to (F, T)
    v_t = localized_smooth_field_1d(num_time_steps, epsilon=epsilon_dict['t_stretch'], device=device)
    v_t_broadcasted = v_t.unsqueeze(0).expand(num_freq_bins, -1)

    # 2. Frequency stretch: w(f) → broadcast to (F, T)
    w_f = localized_smooth_field_1d(num_freq_bins, epsilon=epsilon_dict['f_stretch'], device=device)
    w_f_broadcasted = w_f.unsqueeze(1).expand(-1, num_time_steps)

    # 3. 2D warp: v_2d and w_2d (already [F, T])
    v_2d = localized_smooth_field_2d(num_freq_bins, num_time_steps, epsilon=epsilon_dict['warp_2d'], device=device)
    w_2d = localized_smooth_field_2d(num_freq_bins, num_time_steps, epsilon=epsilon_dict['warp_2d'], device=device)

    # 4. Amplitude modulation α(f,t)
    alpha = localized_smooth_field_2d(num_freq_bins, num_time_steps, epsilon=epsilon_dict['amplitude'], device=device)

    # 5. Phase modulation β(f,t)
    beta = localized_smooth_field_2d(num_freq_bins, num_time_steps, epsilon=epsilon_dict['phase'], device=device)

    return {
        't_stretch': v_t_broadcasted,  # shape: [F, T]
        'f_stretch': w_f_broadcasted,  # shape: [F, T]
        'warp_2d': (v_2d,w_2d),        # shape: ([F, T],[F, T])
        'amplitude': alpha,            # shape: [F, T]
        'phase': beta                  # shape: [F, T]
    }




def apply_transformation(spectrogram, field, mode='t_stretch'):
    """
    Apply a transformation to a log-Mel spectrogram of shape [F, T].
    Returns: [F, T]
    """
    F_bins, T_steps = spectrogram.shape
    device = spectrogram.device

    # Coordinate grids
    t_coords = torch.linspace(-1, 1, T_steps, device=device)
    f_coords = torch.linspace(-1, 1, F_bins, device=device)
    f_grid, t_grid = torch.meshgrid(f_coords, t_coords, indexing='ij')  # [F, T]

    if mode == 't_stretch':
        if field.ndim == 1:
            field = field.unsqueeze(0).expand(F_bins, -1)  # [F, T]
        delta_t = field / T_steps * 2
        delta_f = torch.zeros_like(delta_t)

    elif mode == 'f_stretch':
        if field.ndim == 1:
            field = field.unsqueeze(1).expand(-1, T_steps)
        delta_t = torch.zeros_like(field)
        delta_f = field / F_bins * 2

    elif mode == 'warp_2d':
        v_field, w_field = field  # both [F, T]
        delta_t = v_field / T_steps * 2
        delta_f = w_field / F_bins * 2

    elif mode == 'amplitude':
        return spectrogram * (1.0 + field)

    elif mode == 'phase':
        return spectrogram  # placeholder

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    warped_t = t_grid + delta_t  # x-axis (time)
    warped_f = f_grid + delta_f  # y-axis (freq)
    grid = torch.stack([warped_t, warped_f], dim=-1)  # [F, T, 2]
    grid = grid.unsqueeze(0)  # [1, F, T, 2]

    # Input spectrogram: [1, 1, F, T]
    spec = spectrogram.unsqueeze(0).unsqueeze(0)

    warped = F.grid_sample(spec, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped.squeeze(0).squeeze(0)  # [F, T]



def apply_random_transformations(spectrogram, num_transforms=1, transform_pool=None, device='cpu'):
    """
    Apply N randomly chosen transformations to the input spectrogram.
    """
    if transform_pool is None:
        transform_pool = ['t_stretch', 'f_stretch', 'warp_2d', 'amplitude', 'phase']

    selected = random.sample(transform_pool, num_transforms)
    S = spectrogram.clone().to(device)

    epsilon_dict = {
        't_stretch': 2.0,
        'f_stretch': 2.0,
        'warp_2d': 2.0,
        'amplitude': 1.0,
        'phase': 1.0,
    }

    fields = generate_lie_generator_fields(S.shape[-2], S.shape[-1], epsilon_dict=epsilon_dict, device=device)
    # pp(fields)

    for transform in selected:
        print(f"Applying {transform}")
        field = fields[transform].to(device)
        S = apply_transformation(S, field, mode=transform) 

    return S, fields, selected




def apply_random_curriculum_transform(S_n, 
                                      epoch, 
                                      max_epochs, 
                                      device='cpu', 
                                      epsilon_dict = {},
                                      transform_ids = [0]):
    """
    Apply 1–N progressively harder distortions based on current epoch.
    """

    # Increase number of transforms step-wise 
    d_epoch = min(3,max_epochs)
    n_steps = max_epochs//d_epoch    
    num_transforms_min = 1
    num_transforms_max = 1
    d_num_transforms = (num_transforms_max - num_transforms_min)/n_steps
    num_transforms = num_transforms_min + int((epoch//d_epoch)*d_num_transforms)

    # Randomly choose transformations
    available_modes = ['t_stretch', 'f_stretch', 'warp_2d', 'amplitude']
    selected = [available_modes[i] for i in transform_ids]
    selected = selected[:num_transforms]

    S = S_n.clone().to(device)
    fields = generate_lie_generator_fields(S.shape[-2], S.shape[-1], epsilon_dict=epsilon_dict, device=device)

    for transform in selected:
        # print(f"Applying {transform} with epsilon={epsilon}")
        field = fields[transform]
        S = apply_transformation(S, field, mode=transform) 

    return S, fields, epsilon_dict, selected



def apply_inverse_transform(S_distorted, pred_fields, epsilon_dict=None):
    """
    Applies the inverse transformation based on predicted Lie generator fields.

    Args:
        S_distorted: Tensor of shape [B, 1, F, T] — normalized distorted spectrogram
        pred_fields: Tensor of shape [B, 5, F, T] — predicted normalized Lie fields
        epsilon_dict: dict mapping each field to max ε (needed to rescale from [-1, 1])

    Returns:
        S_recon: Tensor of shape [B, 1, F, T] — spectrogram after inverse warp
    """

    B, C, F, T = pred_fields.shape
    device = pred_fields.device
    S_recon = S_distorted.clone()

    # print("apply_inverse_transform: S_distorted",S_distorted.shape)

    # Unnormalize fields back to real ε ranges
    if epsilon_dict is None:
        epsilon_dict = {
            't_stretch': 0.1,
            'f_stretch': 0.2,
            'warp_2d': 0.2,
            'amplitude': 0.05,
            'phase': 0.1,
        }

    # Denormalize predicted fields from [-1, 1] to [-ε, +ε]
    t_stretch = pred_fields[:, 0] * epsilon_dict['t_stretch']
    f_stretch = pred_fields[:, 1] * epsilon_dict['f_stretch']
    warp_v    = pred_fields[:, 2] * epsilon_dict['warp_2d']
    warp_w    = pred_fields[:, 3] * epsilon_dict['warp_2d']
    amplitude = pred_fields[:, 4] * epsilon_dict['amplitude']

    # print("t_stretch,f_stretch,warp_v,warp_w,amplitude shapes:")
    # print(t_stretch.shape,f_stretch.shape,warp_v.shape,warp_w.shape,amplitude.shape)

    # Reverse time and frequency stretch
    S_recon = batch_grid_warp(S_recon, -t_stretch, -f_stretch)
    # print("apply_inverse_transform: -t_stretch, -f_stretch",S_recon.shape)

    # Reverse 2D warp
    S_recon = batch_grid_warp(S_recon, -warp_v, -warp_w)
    # print("apply_inverse_transform: -warp_v, -warp_w",S_recon.shape)

    # Reverse amplitude modulation (1 / (1 + α) ≈ 1 - α for small α)
    amplitude = amplitude.unsqueeze(1) # Add channel dim
    # print("apply_inverse_transform: amplitude",amplitude.shape)
    
    S_recon = S_recon / (1.0 + amplitude).clamp(min=0.5)  # Safe division

    # print("apply_inverse_transform: S_recon",S_recon.shape)

    return S_recon



def batch_grid_warp(spectrogram, delta_t, delta_f):
    """
    Applies a batched 2D warp using delta_t and delta_f.

    Args:
        spectrogram: [B, 1, F, T]
        delta_t: [B, F, T] — time axis deformation
        delta_f: [B, F, T] — freq axis deformation

    Returns:
        Warped spectrogram [B, 1, F, T]
    """
    B, _, F_bins, T_steps = spectrogram.shape
    # print("batch_grid_warp: spectrogram",spectrogram.shape)
    device = spectrogram.device

    # Coordinate grid
    t_coords = torch.linspace(-1, 1, T_steps, device=device)
    f_coords = torch.linspace(-1, 1, F_bins, device=device)
    f_grid, t_grid = torch.meshgrid(f_coords, t_coords, indexing='ij')  # [F, T]
    f_grid = f_grid.expand(B, -1, -1)  # [B, F, T]
    t_grid = t_grid.expand(B, -1, -1)

    warped_f = f_grid + delta_f / F_bins * 2
    warped_t = t_grid + delta_t / T_steps * 2
    grid = torch.stack([warped_t, warped_f], dim=-1)  # [B, F, T, 2]

    grid_sample =  F.grid_sample(
        spectrogram,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    # print("batch_grid_warp: grid_sample",grid_sample.shape)

    return grid_sample
