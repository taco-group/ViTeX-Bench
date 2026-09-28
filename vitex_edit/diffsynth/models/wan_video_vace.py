import torch
import torch.nn as nn
from .wan_video_dit import DiTBlock, CrossAttention, flash_attention
from ..core.gradient import gradient_checkpoint_forward


class _OffloadToCPU(torch.autograd.Function):
    """Move tensor to CPU in forward, move gradient to GPU in backward."""
    @staticmethod
    def forward(ctx, tensor):
        ctx.gpu_device = tensor.device
        return tensor.detach().cpu().requires_grad_(True)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.to(ctx.gpu_device)


class _RestoreToGPU(torch.autograd.Function):
    """Move CPU tensor back to GPU in forward, move gradient to CPU in backward."""
    @staticmethod
    def forward(ctx, tensor, device):
        ctx.save_for_backward(torch.empty(0))
        ctx._device_str = str(device)
        return tensor.to(device)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.cpu(), None


def tokenize_target_text(text, max_len=64, vocab_size=8192):
    """Convert target text string to character-level token IDs.

    Uses modular hashing of Unicode code points to map any character
    (ASCII, CJK, Cyrillic, math symbols, etc.) into a fixed vocabulary.
    Token 0 is reserved for padding.
    """
    ids = []
    for ch in text[:max_len]:
        ids.append(ord(ch) % (vocab_size - 1) + 1)
    ids += [0] * (max_len - len(ids))
    return ids


class ConditionCrossAttention(nn.Module):
    """Cross-attention for injecting condition features into VACE blocks.

    Used for both glyph visual features (v2) and character-level text
    tokens (v3). Each spatial position in the VACE hidden states queries
    the condition tokens to determine what should be rendered at that location.

    Zero-initialized output projection ensures the model starts from
    pretrained VACE behavior and gradually learns to use the condition.
    """

    def __init__(self, dim, num_heads, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(dim, eps=eps)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        # Zero-init output projection so condition attention starts as no-op
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def forward(self, x, condition_tokens):
        """
        Args:
            x: VACE hidden states (B, seq_len, dim)
            condition_tokens: condition features (B, num_tokens, dim)
        Returns:
            x + condition_attn_output (B, seq_len, dim)
        """
        residual = x
        x = self.norm(x)
        q = self.q(x)
        k = self.k(condition_tokens)
        v = self.v(condition_tokens)
        out = flash_attention(q, k, v, self.num_heads)
        return residual + self.o(out)


# Backward compatibility alias
GlyphCrossAttention = ConditionCrossAttention


class VaceWanAttentionBlock(DiTBlock):
    def __init__(self, has_image_input, dim, num_heads, ffn_dim, eps=1e-6, block_id=0):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = torch.nn.Linear(self.dim, self.dim)
        self.after_proj = torch.nn.Linear(self.dim, self.dim)

    def forward(self, c, x, context, t_mod, freqs, condition_tokens=None):
        if self.block_id == 0:
            c = self.before_proj(c) + x
        c = super().forward(c, context, t_mod, freqs)

        # Condition cross-attention (injected after text cross-attn + FFN)
        # Works for both glyph tokens (v2) and character tokens (v3)
        if condition_tokens is not None and hasattr(self, 'condition_cross_attn'):
            c = self.condition_cross_attn(c, condition_tokens)

        c_skip = self.after_proj(c)
        return c_skip, c


class GlyphEncoder(nn.Module):
    """Lightweight encoder that compresses glyph latents into tokens
    for cross-attention in VACE blocks.

    glyph_latent (16ch, from VAE) → Conv3D patch embed → spatial pooling
    → compact token sequence that each VACE block attends to.
    """

    def __init__(self, in_channels=16, dim=1536, num_tokens=64, patch_size=(1, 2, 2)):
        super().__init__()
        self.num_tokens = num_tokens

        # Patch embedding (same architecture as vace_patch_embedding)
        self.patch_embed = nn.Conv3d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

        # Compress spatial sequence to fixed number of tokens via cross-attention pooling
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.pool_norm = nn.LayerNorm(dim)

        # Output projection (zero-init for safe initialization)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, glyph_latent):
        """
        Args:
            glyph_latent: (B, 16, T, H, W) from VAE-encoded glyph video
        Returns:
            glyph_tokens: (B, num_tokens, dim) compressed glyph features
        """
        # Patch embed: (B, dim, T, H/2, W/2)
        x = self.patch_embed(glyph_latent)
        B = x.shape[0]
        # Flatten spatial: (B, dim, N) → (B, N, dim)
        x = x.flatten(2).transpose(1, 2)

        # Cross-attention pooling: compress N spatial tokens → num_tokens
        queries = self.query_tokens.expand(B, -1, -1).to(dtype=x.dtype, device=x.device)
        x_normed = self.pool_norm(x)
        pooled, _ = self.pool_attn(queries, x_normed, x_normed)

        return self.out_proj(pooled)


class TargetTextEncoder(nn.Module):
    """Character-level encoder for target text strings.

    Encodes the target text (e.g., "NAD", "CARLIFE", "善") at the character
    level using learned embeddings and a small Transformer. This provides
    character-precise identity information that T5's subword tokenization
    cannot guarantee.

    The output tokens are used as K/V in ConditionCrossAttention within
    each VACE block, allowing each spatial position to query which character
    it should render.
    """

    def __init__(self, vocab_size=8192, max_len=64, dim=1536,
                 num_layers=2, num_heads=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.char_embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 4,
            batch_first=True, norm_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        # Zero-init output projection for safe initialization
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, token_ids):
        """
        Args:
            token_ids: (B, max_len) long tensor of character IDs (0 = PAD)
        Returns:
            text_tokens: (B, max_len, dim)
        """
        B, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)

        x = self.char_embed(token_ids) + self.pos_embed(positions)

        # Padding mask: True means ignore this position
        pad_mask = (token_ids == 0)

        x = self.transformer(x, src_key_padding_mask=pad_mask)
        return self.out_proj(x)


class VaceWanModel(torch.nn.Module):
    def __init__(
        self,
        vace_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
        vace_in_dim=96,
        glyph_channels=0,
        glyph_num_tokens=64,
        use_target_text_encoder=False,
        text_vocab_size=8192,
        text_max_len=64,
        text_num_layers=2,
        patch_size=(1, 2, 2),
        has_image_input=False,
        dim=1536,
        num_heads=12,
        ffn_dim=8960,
        eps=1e-6,
    ):
        super().__init__()
        self.vace_layers = vace_layers
        self.vace_in_dim = vace_in_dim
        self.glyph_channels = glyph_channels
        self.use_target_text_encoder = use_target_text_encoder
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # VACE blocks
        self.vace_blocks = torch.nn.ModuleList([
            VaceWanAttentionBlock(has_image_input, dim, num_heads, ffn_dim, eps, block_id=i)
            for i in self.vace_layers
        ])

        # VACE patch embedding (original 96 channels, no glyph concat)
        self.vace_patch_embedding = torch.nn.Conv3d(
            vace_in_dim, dim, kernel_size=patch_size, stride=patch_size
        )

        # Glyph pathway (v2: glyph video → GlyphEncoder → cross-attention)
        if glyph_channels > 0:
            self.glyph_encoder = GlyphEncoder(
                in_channels=glyph_channels,
                dim=dim,
                num_tokens=glyph_num_tokens,
                patch_size=patch_size,
            )
            for block in self.vace_blocks:
                block.condition_cross_attn = ConditionCrossAttention(dim, num_heads, eps)

        # Target text pathway (v3: text string → TargetTextEncoder → cross-attention)
        if use_target_text_encoder:
            self.target_text_encoder = TargetTextEncoder(
                vocab_size=text_vocab_size,
                max_len=text_max_len,
                dim=dim,
                num_layers=text_num_layers,
                num_heads=num_heads,
            )
            for block in self.vace_blocks:
                block.condition_cross_attn = ConditionCrossAttention(dim, num_heads, eps)

    def _has_new_modules(self):
        """Check if this model has modules not present in pretrained checkpoints."""
        return self.glyph_channels > 0 or self.use_target_text_encoder

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if self._has_new_modules():
            # New modules won't be in pretrained checkpoints.
            # First, materialize any meta-tensor parameters so they can be assigned.
            for name, param in self.named_parameters():
                if param.is_meta:
                    materialized = torch.zeros(param.shape, dtype=param.dtype, device='cpu')
                    parts = name.split('.')
                    module = self
                    for p in parts[:-1]:
                        module = getattr(module, p)
                    setattr(module, parts[-1], torch.nn.Parameter(materialized))

            result = super().load_state_dict(state_dict, strict=False, assign=assign)

            # Initialize a new module only if its weights are absent from `state_dict`,
            # i.e. when building from the pretrained Wan2.1-VACE checkpoint. Re-initializing
            # unconditionally would overwrite trained ViTeX-Edit-14B weights on load.
            missing = set(result.missing_keys)

            def is_missing(prefix):
                return any(k.startswith(prefix) for k in missing)

            # Re-initialize glyph modules (v2 mode)
            if hasattr(self, 'glyph_encoder') and is_missing('glyph_encoder.'):
                for name, module in self.glyph_encoder.named_modules():
                    if isinstance(module, (torch.nn.Linear, torch.nn.Conv3d)):
                        if 'out_proj' not in name:
                            torch.nn.init.xavier_uniform_(module.weight)
                            if module.bias is not None:
                                torch.nn.init.zeros_(module.bias)
                if hasattr(self.glyph_encoder, 'query_tokens'):
                    torch.nn.init.normal_(self.glyph_encoder.query_tokens, std=0.02)

            # Re-initialize target text encoder modules (v3 mode)
            if hasattr(self, 'target_text_encoder') and is_missing('target_text_encoder.'):
                for name, module in self.target_text_encoder.named_modules():
                    if isinstance(module, (torch.nn.Linear,)):
                        if 'out_proj' not in name:
                            torch.nn.init.xavier_uniform_(module.weight)
                            if module.bias is not None:
                                torch.nn.init.zeros_(module.bias)
                    elif isinstance(module, torch.nn.Embedding):
                        torch.nn.init.normal_(module.weight, std=0.02)
                        if module.padding_idx is not None:
                            torch.nn.init.zeros_(module.weight[module.padding_idx])

            # Re-initialize condition cross-attention modules
            for block_idx, block in enumerate(self.vace_blocks):
                if hasattr(block, 'condition_cross_attn') and is_missing(f'vace_blocks.{block_idx}.condition_cross_attn.'):
                    ca = block.condition_cross_attn
                    for name, module in ca.named_modules():
                        if isinstance(module, torch.nn.Linear):
                            if name == 'o':
                                torch.nn.init.zeros_(module.weight)
                                torch.nn.init.zeros_(module.bias)
                            else:
                                torch.nn.init.xavier_uniform_(module.weight)
                                if module.bias is not None:
                                    torch.nn.init.zeros_(module.bias)

            # Cast re-initialized modules to model dtype (xavier_uniform_ etc. produce float32)
            target_dtype = next((p.dtype for p in self.parameters() if p.dtype != torch.float32), torch.bfloat16)
            if hasattr(self, 'glyph_encoder'):
                self.glyph_encoder.to(dtype=target_dtype)
            if hasattr(self, 'target_text_encoder'):
                self.target_text_encoder.to(dtype=target_dtype)
            for block in self.vace_blocks:
                if hasattr(block, 'condition_cross_attn'):
                    block.condition_cross_attn.to(dtype=target_dtype)

            return result
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(
        self, x, vace_context, context, t_mod, freqs,
        glyph_latent=None,
        target_text_ids=None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, x.shape[1] - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])

        # Encode condition tokens for cross-attention
        condition_tokens = None
        if glyph_latent is not None and hasattr(self, 'glyph_encoder'):
            condition_tokens = self.glyph_encoder(glyph_latent)
        elif target_text_ids is not None and hasattr(self, 'target_text_encoder'):
            condition_tokens = self.target_text_encoder(target_text_ids)

        hints = []
        for block in self.vace_blocks:
            result = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                c, x, context, t_mod, freqs, condition_tokens
            )
            c_skip, c = result
            hints.append(c_skip)
        return hints
