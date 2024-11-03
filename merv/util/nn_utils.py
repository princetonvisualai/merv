"""
nn_utils.py

Utility functions and PyTorch submodule definitions.
"""

import math
from abc import ABC, abstractmethod
from functools import partial

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import LayerNorm2d, trunc_normal_
from timm.models.regnet import RegStage
from torch.nn.init import xavier_uniform_
from torch.nn.parameter import Parameter


# === Definitions for Various Projection Modules, with Signature :: [..., in_dim] --> [..., out_dim] ===
class LinearProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, pre_proj_layernorm: bool = False) -> None:
        super().__init__()
        self.projector = nn.Linear(vision_dim, llm_dim, bias=True)
        if pre_proj_layernorm:
            self.layernorm = nn.LayerNorm(vision_dim)
        else:
            self.layernorm = nn.Identity()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(self.layernorm(img_patches))


class MLPProjector(nn.Module):
    def __init__(
        self, vision_dim: int, llm_dim: int, mlp_type: str = "gelu-mlp", pre_proj_layernorm: bool = False
    ) -> None:
        super().__init__()
        if pre_proj_layernorm:
            self.layernorm = nn.LayerNorm(vision_dim)
        else:
            self.layernorm = nn.Identity()

        if mlp_type == "gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Projector with `{mlp_type = }` is not supported!")

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(self.layernorm(img_patches))

    @property
    def output_token_length(self) -> int:
        return 1


class MLPDeepProjector(nn.Module):
    def __init__(
        self, vision_dim: int, llm_dim: int, mlp_type: str = "gelu-mlp", pre_proj_layernorm: bool = False
    ) -> None:
        super().__init__()
        if pre_proj_layernorm:
            self.layernorm = nn.LayerNorm(vision_dim)
        else:
            self.layernorm = nn.Identity()
        if mlp_type == "gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Projector with `{mlp_type = }` is not supported!")

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(self.layernorm(img_patches))


class FusedMLPProjector(nn.Module):
    def __init__(
        self, fused_vision_dim: int, llm_dim: int, mlp_type: str = "fused-gelu-mlp", pre_proj_layernorm: bool = False
    ) -> None:
        super().__init__()
        if pre_proj_layernorm:
            self.layernorm = nn.LayerNorm(fused_vision_dim)
        else:
            self.layernorm = nn.Identity()
        self.initial_projection_dim = fused_vision_dim * 4
        if mlp_type == "fused-gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(fused_vision_dim, self.initial_projection_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.initial_projection_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Fused Projector with `{mlp_type = }` is not supported!")

    def forward(self, fused_img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(self.layernorm(fused_img_patches))


def get_mlp_projector(fused_vision_dim: int, llm_dim: int, mlp_type: str = "gelu-mlp") -> nn.Module:
    if mlp_type == "linear":
        return LinearProjector(fused_vision_dim, llm_dim)
    elif mlp_type == "gelu-mlp":
        return MLPProjector(fused_vision_dim, llm_dim, mlp_type)
    elif mlp_type == "fused-gelu-mlp":
        return FusedMLPProjector(fused_vision_dim, llm_dim, mlp_type)
    elif mlp_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Projector with `{mlp_type = }` is not supported!")


class TokenResampler(nn.Module, ABC):
    """Resamples token length as well. Abstract class for interfacing resulting token length."""

    @property
    @abstractmethod
    def output_token_length(self) -> int: ...

    @property
    @abstractmethod
    def output_frame_length(self) -> int: ...


class AveragePoolingProjector(TokenResampler):
    """Emu-2 style projector using nxn average pooling on the output followed by a linear project.
    https://arxiv.org/abs/2312.13286
    """

    def __init__(
        self, fused_vision_dim: int, llm_dim: int, output_size: int, output_frames: int = 8, mlp_type: str = "gelu-mlp"
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.output_frames = output_frames

        self.avg_pool = nn.AdaptiveAvgPool2d((output_size, output_size))
        self.projector = get_mlp_projector(fused_vision_dim, llm_dim, mlp_type)

    def forward(self, fused_img_patches: torch.Tensor) -> torch.Tensor:
        assert fused_img_patches.dim() == 4
        # Reshape the fused_img_patches to [B, F, N, C] -> [B*F, C, H, W]
        num_frames = fused_img_patches.shape[1]
        assert num_frames == self.output_frames
        H = int(math.sqrt(fused_img_patches.shape[2]))
        fused_img_patches = einops.rearrange(
            fused_img_patches,
            "B F (H W) C -> (B F) C H W",
            H=H,
        )
        pooled_img_patches = self.avg_pool(fused_img_patches)
        pooled_img_patches = einops.rearrange(pooled_img_patches, "(B F) C H W -> B F (H W) C", F=num_frames)
        pooled_img_patches = self.projector(pooled_img_patches)
        pooled_img_patches = einops.rearrange(pooled_img_patches, "B F (H W) C -> B (F H W) C", H=self.output_size)
        return pooled_img_patches

    @property
    def output_token_length(self) -> int:
        return self.output_size**2

    @property
    def output_frame_length(self) -> int:
        return self.output_frames


class AttentivePooler(TokenResampler):
    """Attentive Pooler from JEPA. Different Implementation than Flamingo.
    https://github.com/facebookresearch/jepa/blob/main/src/models/attentive_pooler.py
    """

    def __init__(
        self,
        fused_vision_dim: int,
        llm_dim: int,
        num_query_tokens: int,
        num_heads: int = 8,
        output_frames: int = 8,
        mlp_type: str = "gelu-mlp",
    ) -> None:
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.output_frames = output_frames
        assert fused_vision_dim % num_heads == 0, "fused_vision_dim must be divisible by num_heads"

        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, fused_vision_dim))
        self.cross_attn = CrossAttentionBlock(
            dim=fused_vision_dim,
            num_heads=num_heads,
            qkv_bias=True,
        )
        self.projector = get_mlp_projector(fused_vision_dim, llm_dim, mlp_type)

        trunc_normal_(self.query_tokens, std=0.02)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        rescale(self.cross_attn.xattn.proj.weight.data, 1)
        rescale(self.cross_attn.mlp.fc2.weight.data, 1)

    def _init_weights(self, m):
        init_std = 0.02
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [B, F, N, C]
        num_frames = x.shape[1]
        assert num_frames == self.output_frames
        x = einops.rearrange(x, "B F N C -> (B F) N C")
        q = self.query_tokens.expand(x.shape[0], -1, -1)
        q = self.cross_attn(q, x)
        q = self.projector(q)
        q = einops.rearrange(q, "(B F) N C -> B (F N) C", F=num_frames)
        return q

    @property
    def output_token_length(self) -> int:
        return self.num_query_tokens

    @property
    def output_frame_length(self) -> int:
        return self.output_frames


class ConvolutionalProjector(TokenResampler):
    """C-Abstractor module as proposed in Honeybee, using a Resnet to preserve local info.
    https://arxiv.org/abs/2312.06742
    https://github.com/kakaobrain/honeybee/blob/main/honeybee/projectors.py
    """

    def __init__(
        self,
        fused_vision_dim: int,
        llm_dim: int,
        output_size: int,
        block_depth: int,
        output_frames: int = 8,
        mlp_type: str = "gelu-mlp",
    ) -> None:
        super().__init__()
        RegBlock = partial(RegStage, stride=1, dilation=1, act_layer=nn.SiLU, norm_layer=LayerNorm2d)
        self.output_size = output_size
        self.output_frames = output_frames

        # Final sequence is of size B x output_size**2 x llm_dim
        self.convolution_pooling = nn.Sequential(
            RegBlock(block_depth, fused_vision_dim, llm_dim),
            nn.AdaptiveAvgPool2d((output_size, output_size)),
            RegBlock(block_depth, llm_dim, llm_dim),
        )
        # Output length is output_size**2

        self.projector = get_mlp_projector(llm_dim, llm_dim, mlp_type)

    def forward(self, fused_img_patches: torch.Tensor) -> torch.Tensor:
        # Reshape the fused_img_patches to [B, F, N, C] -> [B, C, H, W]
        H = int(math.sqrt(fused_img_patches.shape[2]))
        assert fused_img_patches.dim() == 4
        num_frames = fused_img_patches.shape[1]
        assert num_frames == self.output_frames

        fused_img_patches = einops.rearrange(
            fused_img_patches,
            "B F (H W) C -> (B F) C H W",
            H=H,
        )
        pooled_img_patches = self.convolution_pooling(fused_img_patches)
        pooled_img_patches = einops.rearrange(pooled_img_patches, "(B F) C H W -> B F (H W) C", F=num_frames)
        pooled_img_patches = self.projector(pooled_img_patches)
        pooled_img_patches = einops.rearrange(pooled_img_patches, "B F (H W) C -> B (F H W) C", H=self.output_size)
        return pooled_img_patches

    @property
    def output_token_length(self) -> int:
        return self.output_size**2

    @property
    def output_frame_length(self) -> int:
        return self.output_frames


class AveragePooling3DProjector(TokenResampler):
    """3D-average pooling projector."""

    def __init__(
        self, fused_vision_dim: int, llm_dim: int, output_frames: int, output_size: int, mlp_type: str = "gelu-mlp"
    ) -> None:
        super().__init__()

        self.output_frames = output_frames
        self.output_size = output_size

        self.avg_pooling = nn.AdaptiveAvgPool3d((output_frames, output_size, output_size))
        self.projector = get_mlp_projector(fused_vision_dim, llm_dim, mlp_type)

    def forward(self, fused_img_patches: torch.Tensor) -> torch.Tensor:
        # Reshape the fused_img_patches to [B, F, N, C] -> [B, C, F, H, W]
        H = int(math.sqrt(fused_img_patches.shape[2]))
        fused_img_patches = einops.rearrange(
            fused_img_patches,
            "B F (H W) C -> B C F H W",
            H=H,
        )
        pooled_img_patches = self.avg_pooling(fused_img_patches)
        pooled_img_patches = einops.rearrange(pooled_img_patches, "B C F H W -> B (F H W) C")
        return self.projector(pooled_img_patches)

    @property
    def output_token_length(self) -> int:
        return self.output_size * self.output_size

    @property
    def output_frame_length(self) -> int:
        return self.output_frames


class Convolutional3DProjector(TokenResampler):
    """3D-convolutional projector."""

    def __init__(
        self, fused_vision_dim: int, llm_dim: int, output_frames: int, output_size: int, mlp_type: str = "gelu-mlp"
    ) -> None:
        super().__init__()

        self.output_frames = output_frames
        self.output_size = output_size

        self.convolution_pooling = nn.Sequential(
            nn.Conv3d(fused_vision_dim, llm_dim, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool3d((output_frames, output_size, output_size)),
        )
        self.projector = get_mlp_projector(llm_dim, llm_dim, mlp_type)

    def forward(self, fused_img_patches: torch.Tensor) -> torch.Tensor:
        # Reshape the fused_img_patches to [B, F, N, C] -> [B, C, F, H, W]
        assert fused_img_patches.dim() == 4
        H = int(math.sqrt(fused_img_patches.shape[2]))
        fused_img_patches = einops.rearrange(
            fused_img_patches,
            "B F (H W) C -> B C F H W",
            H=H,
        )
        pooled_img_patches = self.convolution_pooling(fused_img_patches)
        pooled_img_patches = einops.rearrange(pooled_img_patches, "B C F H W -> B (F H W) C")
        return self.projector(pooled_img_patches)

    @property
    def output_token_length(self) -> int:
        return self.output_size * self.output_size

    @property
    def output_frame_length(self) -> int:
        return self.output_frames


class CrossAttention(nn.Module):
    """From JEPA."""

    def __init__(self, dim, num_heads=12, qkv_bias=False, use_sdpa=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim * 2), bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_sdpa = use_sdpa

    def forward(self, q, x):
        B, n, C = q.shape
        q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                q = F.scaled_dot_product_attention(q, k, v)
        else:
            xattn = (q @ k.transpose(-2, -1)) * self.scale
            xattn = xattn.softmax(dim=-1)  # (batch_size, num_heads, query_len, seq_len)
            q = xattn @ v

        q = q.transpose(1, 2).reshape(B, n, C)
        q = self.proj(q)

        return q


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    """From JEPA."""

    def __init__(
        self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, out_dim=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_dim, act_layer=act_layer)

    def forward(self, q, x):
        y = self.xattn(q, self.norm1(x))
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q


# Adapters here
class CrossAttentionAdapterLearnableQuery(nn.Module):
    def __init__(
        self, embed_dim=3072, llm_dim=4098, token_length=8, averagetoken=False, num_encoder=4, positional_embedding=False
    ) -> None:
        super().__init__()

        self.llm_dim = llm_dim
        self.token_length = token_length
        self.averagetoken = averagetoken
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=1,
            dropout=0.0,
            batch_first=True,
            kdim=llm_dim if averagetoken else token_length * llm_dim,
            vdim=llm_dim if averagetoken else token_length * llm_dim,
        )

        self.Q = Parameter(torch.empty((1, embed_dim)))
        self.num_encoder = num_encoder
        self.positional_embedding = positional_embedding

        if positional_embedding:
            self.pe = Parameter(torch.empty((self.num_encoder, llm_dim)))

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.Q)
        if self.positional_embedding:
            xavier_uniform_(self.pe)

    def forward(
        self,
        V,
    ):
        # Q should be the emb of the query. So, size is (B, 3072)
        # V is a list of tensors with size (B, T, K). T should all be same, or 1.

        for emb in V:
            assert emb.shape[1] == self.token_length or emb.shape[1] == 1, (self.token_length, [e.shape for e in V])

        B = V[0].shape[0]
        E = len(V)
        Q = self.Q.repeat(B, 1)

        Q = Q.unsqueeze(1)  # (B, 3072) -> (B, 1, 3072)
        V = [(emb.repeat(1, self.token_length, 1) if emb.shape[1] == 1 else emb) for emb in V]
        V = torch.stack(V, 1)  # (B, encoders, T, C)

        # V = V.reshape(B, E, self.llm_dim * self.token_length)
        # _, weights = self.attention(query=Q, key=V, value=V)

        if self.averagetoken:
            V_ = V.mean(2)
            if self.positional_embedding:
                V_ = V_ + self.pe.unsqueeze(0).repeat(B, 1, 1)
            p, weights = self.attention(query=Q, key=V_, value=V_)
            V = V.reshape(B, E, self.llm_dim * self.token_length)

        else:
            V = V.reshape(
                B, E, self.llm_dim * self.token_length
            )  # this can lead to humonguous MLP layer in the attention.
            p, weights = self.attention(query=Q, key=V, value=V)

        return torch.bmm(weights, V).reshape(B, self.token_length, self.llm_dim), weights[:, 0]


class ScalarAdapter(nn.Module):
    def __init__(self, num_encoder=4) -> None:
        super().__init__()
        self.scalar = torch.nn.Parameter(torch.randn(4))

    def forward(self, projected_patch_embeddings):
        # projected_patch_embeddings = [[B, 8, 4096], [B, 8, 4096],]

        projected_patch_embeddings = torch.stack(projected_patch_embeddings, 0)  # [E, B, 8, 4096]

        mixer_value = self.scalar.softmax(0).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        projected_patch_embeddings = (projected_patch_embeddings * mixer_value).sum(0)

        return projected_patch_embeddings, self.scalar.softmax(0).unsqueeze(0)


if __name__ == "__main__":
    # Test the AverageProjector
    fused_vision_dim = 128
    llm_dim = 32
    input_size = 14
    output_size = 8
    mlp_type = "linear"

    projector = AveragePoolingProjector(fused_vision_dim, llm_dim, input_size, output_size, mlp_type)
    print(projector)
    dummy_input = torch.randn(4, input_size * input_size, fused_vision_dim)
    output = projector(dummy_input)
    print(output.shape)
    print("Output token length:", projector.output_token_length)

    input_size = 32
    output_size = 16
    projector = AveragePoolingProjector(fused_vision_dim, llm_dim, input_size, output_size, mlp_type)
    print(projector)
    dummy_input = torch.randn(4, input_size * input_size, fused_vision_dim)
    output = projector(dummy_input)
    print(output.shape)
    print("Output token length:", projector.output_token_length)

    # Test the AttentionProjector
    num_query_tokens = 64
    num_heads = 1
    depth = 1
    attn_proj = AttentivePooler(fused_vision_dim, llm_dim, num_query_tokens, num_heads, mlp_type=mlp_type)
    print(attn_proj)
    output = attn_proj(dummy_input)
    print(output.shape)
    print("Output token length:", attn_proj.output_token_length)

    # Test the ConvolutionalProjector
    block_depth = 3  # same as paper
    output_size = 8
    print("conv proj")
    conv_proj = ConvolutionalProjector(fused_vision_dim, llm_dim, output_size, block_depth, mlp_type=mlp_type)
    output = conv_proj(dummy_input)
    print(output.shape)
    print("Output token length:", conv_proj.output_token_length)
