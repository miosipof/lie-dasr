####################################################################################################
# Helper functions ####################################################################################################
####################################################################################################


def pad_spectrogram(spec, target_height=96):
    """
    Pads a spectrogram [B, 1, F, T] to height=96 (bottom-padding).
    """
    pad_height = target_height - spec.shape[2]
    return F.pad(spec, (0, 0, 0, pad_height))  # pad = (left, right, top, bottom)

def imagenet_normalize_1ch(spec):
    """
    Normalize a 1-channel [B, 1, F, T] spectrogram using ImageNet stats
    (broadcasting mean/std of grayscale version)
    """
    mean = 0.485
    std = 0.229
    return (spec - mean) / std


def normalize_spectrogram(S, min_val=-7.0, max_val=1.0):
    """
    Normalize log-Mel spectrogram from [min_val, max_val] to [0, 1]
    Input: Tensor [F, T]
    Output: Tensor [F, T]
    """
    return (S - min_val) / (max_val - min_val)

def denormalize_spectrogram(S_norm, min_val=-7.0, max_val=1.0):
    return S_norm * (max_val - min_val) + min_val


def normalize_field(field, max_abs):
    """
    Normalize a field from [-max_abs, +max_abs] → [-1, 1]
    """
    return torch.clamp(field / max_abs, min=-1.0, max=1.0)

def denormalize_field(norm_field, max_abs):
    return norm_field * max_abs



def normalize_fields(fields, norm_ranges):
    normed_fields = {}
    for key, value in fields.items():
        if key == 'warp_2d':
            v, w = value
            v_norm = normalize_field(v, norm_ranges['warp_2d'])
            w_norm = normalize_field(w, norm_ranges['warp_2d'])
            normed_fields[key] = (v_norm, w_norm)
        else:
            normed_fields[key] = normalize_field(value, norm_ranges[key])
    return normed_fields

def denormalize_fields(normed_fields, norm_ranges):
    denormed = {}
    for key, value in normed_fields.items():
        if key == 'warp_2d':
            v_norm, w_norm = value
            v = denormalize_field(v_norm, norm_ranges['warp_2d'])
            w = denormalize_field(w_norm, norm_ranges['warp_2d'])
            denormed[key] = (v, w)
        else:
            denormed[key] = denormalize_field(value, norm_ranges[key])
    return denormed

    


