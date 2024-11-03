"""
dinov2_vit.py
"""

from functools import partial
from typing import Callable, Tuple

import timm
import torch
from timm.models.vision_transformer import Block, VisionTransformer
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision.transforms import Compose, Resize
from torchvision.transforms import functional as F

from merv.models.backbones.video import VideoBackbone
from merv.models.backbones.video.base_video import LetterboxPad, unpack_tuple

# Registry =>> Supported DINOv2 Vision Backbones (from TIMM) =>> Note:: Using DINOv2 w/ Registers!
#   => Reference: https://arxiv.org/abs/2309.16588
DINOv2_VISION_BACKBONES = {
    "dinov2-video": "vit_large_patch14_reg4_dinov2.lvd142m",
    "dinov2-video-all-tokens": "vit_large_patch14_reg4_dinov2.lvd142m",
    "dinov2-video-classemb-at-first": "vit_large_patch14_reg4_dinov2.lvd142m",
    "dinov2-video-all-token-with-cls": "vit_large_patch14_reg4_dinov2.lvd142m",
}


class DinoV2VideoBackbone(VideoBackbone):
    def __init__(
        self, video_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224, num_frames: int = 8
    ) -> None:
        super().__init__(
            video_backbone_id,
            image_resize_strategy,
            default_image_size=default_image_size,
            num_frames=num_frames,
        )
        self.timm_path_or_url = DINOv2_VISION_BACKBONES[video_backbone_id]

        # Initialize Featurizer (ViT) by downloading from HF / TIMM Hub if necessary
        self.featurizer: VisionTransformer = timm.create_model(
            self.timm_path_or_url,
            pretrained=True,
            num_classes=0,
            img_size=self.default_image_size,
        )

        if "all-token-with-cls" in self.identifier:
            self.featurizer.forward = unpack_tuple(
                partial(
                    self.featurizer.get_intermediate_layers,
                    n={len(self.featurizer.blocks) - 2},
                    return_prefix_tokens=True,
                )
            )
        elif "classemb-at-first" in self.identifier:
            self.featurizer.forward = unpack_tuple(
                partial(
                    self.featurizer.get_intermediate_layers,
                    n={len(self.featurizer.blocks) - 2},
                    return_prefix_tokens=True,
                )
            )
        elif "all-token" in self.identifier:
            self.featurizer.forward = unpack_tuple(
                partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
            )

        self.featurizer.eval()

        # Usually we Monkey-Patch the `forward()` function of the featurizers to ensure FSDP-compatibility
        #   => Note: To adapt DINO to video, we return a class token for every single frame (automatically)
        #            forward() already does this, so we don't need to do anything here!

        # Get Configs for featurizers =>> Note :: Override default image size for larger resolution models
        self.data_cfg = timm.data.resolve_model_data_config(self.featurizer)
        self.data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        # Initialize Default Image Transform --> Modified by `self.image_resize_strategy`
        default_image_transform = timm.data.create_transform(**self.data_cfg, is_training=False)

        # Fix =>> SigLIP & IN1K default transforms resize to *larger* than `self.default_image_size` (crops image)!
        if "siglip" in self.timm_path_or_url or "in1k" in self.timm_path_or_url:
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert isinstance(resize_transform := default_image_transform.transforms[0], Resize)
            default_image_transform = Compose(
                [
                    Resize(self.default_image_size, interpolation=resize_transform.interpolation),
                    *default_image_transform.transforms[1:],
                ]
            )

        # Switch on `image_resize_strategy`
        if self.image_resize_strategy == "resize-naive":
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert isinstance(resize_transform := default_image_transform.transforms[0], Resize)

            target_size = (self.default_image_size, self.default_image_size)
            image_transform = Compose(
                [
                    Resize(target_size, interpolation=resize_transform.interpolation),
                    *default_image_transform.transforms[1:],
                ]
            )

        elif self.image_resize_strategy == "resize-crop":
            image_transform = default_image_transform

        elif self.image_resize_strategy == "letterbox":
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert "mean" in self.data_cfg, "TIMM `data_cfg` missing image normalization mean!"

            # Compute Padding Fill Value (rescaled normalization mean if applicable)
            fill = tuple([int(x * 255) for x in self.data_cfg["mean"]])

            # Build New Transform
            image_transform = Compose([LetterboxPad(fill), *default_image_transform.transforms])

        else:
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!")

        # Since our input is [B C F H W], we need to rearrange to [B F C H W] for processing first
        self.video_transform_ = Compose([F.to_pil_image, *image_transform.transforms])
        self.video_transform = lambda video: torch.stack([self.video_transform_(frame) for frame in video])

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps each ViT block and then both of the _entire_ featurizers."""
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(self, video_values: torch.Tensor, is_image: torch.Tensor) -> torch.Tensor:
        # Forward Pass through the Featurizer (ViT) =>> Note :: Already returns class token for every frame!

        B, F, C, H, W = video_values.shape
        video_values = video_values.reshape(-1, C, H, W)
        # __init__ makes sure to select featurizer vs. featurizer.forward_features
        video_features = self.featurizer(video_values)

        if "classemb-at-first" in self.identifier:
            video_features0 = video_features[0].reshape(B, -1, self.embed_dim)
            video_features1 = (
                video_features[1][:, :1].reshape(B, -1, self.embed_dim).mean(1, keepdim=True)
            )  # 5 prefix tokens, 1st is class, res is register token.
            video_features = torch.concat([video_features1, video_features0], 1)

        elif "all-token-with-cls" in self.identifier:
            video_features0 = video_features[0].reshape(B, -1, self.embed_dim)
            video_features1 = video_features[1][:, :1].reshape(B, -1, self.embed_dim)
            video_features = torch.concat([video_features1, video_features0], 1)
        else:
            video_features = video_features.reshape(B, -1, self.embed_dim)

        return video_features

    @property
    def default_video_resolution(self) -> Tuple[int, int, int, int]:
        return (self.num_frames, *self.data_cfg["input_size"])

    @property
    def embed_dim(self) -> int:
        return self.featurizer.embed_dim

    @property
    def num_patches(self) -> int:
        if "classemb-at-first" in self.identifier:
            return self.num_frames * self.featurizer.patch_embed.num_patches
        elif "all-tokens" not in self.identifier:
            return self.num_frames
        else:
            return self.num_frames * self.featurizer.patch_embed.num_patches

    @property
    def spatial_resolution(self):
        return self.num_patches // self.num_frames

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16
