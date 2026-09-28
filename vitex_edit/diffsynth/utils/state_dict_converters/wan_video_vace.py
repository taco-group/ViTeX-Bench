import torch


def VaceWanModelDictConverter(state_dict):
    state_dict_ = {name: state_dict[name] for name in state_dict if name.startswith("vace")}
    return state_dict_


def expand_patch_embedding_channels(model, state_dict, glyph_channels):
    """Expand vace_patch_embedding Conv3D input channels to accommodate glyph channels.

    Pretrained weights cover the original 96 input channels (inactive + reactive + mask).
    New glyph channels (16) are zero-initialized so the model starts from pretrained
    behavior and gradually learns to use glyph information during fine-tuning.
    """
    if glyph_channels <= 0:
        return

    key_w = "vace_patch_embedding.weight"
    key_b = "vace_patch_embedding.bias"

    if key_w not in state_dict:
        return

    pretrained_w = state_dict[key_w]  # (out_ch, 96, 1, 2, 2)
    out_ch = pretrained_w.shape[0]
    orig_in = pretrained_w.shape[1]
    kernel = pretrained_w.shape[2:]

    expected_in = orig_in + glyph_channels
    if model.vace_patch_embedding.weight.shape[1] != expected_in:
        return

    new_w = torch.zeros(out_ch, expected_in, *kernel, dtype=pretrained_w.dtype)
    new_w[:, :orig_in] = pretrained_w
    state_dict[key_w] = new_w

    # Bias is per output channel, no change needed
    if key_b in state_dict:
        pass  # bias shape is (out_ch,), unchanged
