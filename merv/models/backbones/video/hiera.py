"""
hiera.py

Credit: https://github.com/facebookresearch/hiera
"""

from functools import partial
from typing import Callable, Tuple

import timm
import torch
from hiera import Hiera, HieraBlock
from torch.distributed.fsdp.wrap import (_module_wrap_policy, _or_policy,
                                         transformer_auto_wrap_policy)
from torchvision.transforms import Compose, Resize, functional as F

from merv.models.backbones.video import VideoBackbone
from merv.models.backbones.video.base_video import LetterboxPad, unpack_tuple

# Grab the corresponding models from torch hub
HIERA_VIDEO_BACKBONES = {
    "hiera-base-video": "hiera_base_16x224",
    "hiera-base-plus-video": "hiera_base_plus_16x224",
    "hiera-large-video": "hiera_large_16x224",
}


class HieraVideoBackbone(VideoBackbone):
    def __init__(
        self,
        video_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        num_frames: int = 8,
    ) -> None:
        super().__init__(
            video_backbone_id,
            image_resize_strategy,
            default_image_size=default_image_size,
            num_frames=num_frames,
        )
        self.torchhub_path_or_url = HIERA_VIDEO_BACKBONES[video_backbone_id]

        # Initialize Featurizer (ViT) by downloading from HF / TIMM Hub if necessary
        self.featurizer: Hiera = torch.hub.load(
            "facebookresearch/hiera", model=self.torchhub_path_or_url, pretrained=True, checkpoint="mae_k400_ft_k400"
        )

        self.featurizer.eval()

        # Get Configs for featurizers =>> Note :: Override default image size for larger resolution models
        # Slightly different setup for Hiera models
        self.data_cfg = self.featurizer.config
        self.data_cfg["input_size"] = (num_frames, self.default_image_size, self.default_image_size)
        self.data_cfg["in_chans"] = 3

        # Initialize Default Image Transform --> Modified by `self.image_resize_strategy`
        default_image_transform = timm.data.create_transform(**self.data_cfg, is_training=False)

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

        self.video_transform_ = Compose([F.to_pil_image, *image_transform.transforms])
        self.video_transform = lambda video: torch.stack([self.video_transform_(frame) for frame in video])

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps each ViT block and then both of the _entire_ featurizers."""
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={Hiera})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={HieraBlock})
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(self, video_values: torch.Tensor, is_image: torch.Tensor) -> torch.Tensor:
        # Forward Pass through the Featurizer (ViT) =>> Note :: Already returns class token for every frame!

        B, F, C, H, W = video_values.shape
        video_values = video_values.permute(0, 2, 1, 3, 4).contiguous()
        # __init__ makes sure to select featurizer vs. featurizer.forward_features
        # Grab last feature, which is spatially aligned
        video_features = self.featurizer(video_values, return_intermediates=True)[1][-1]

        return video_features

    @property
    def default_video_resolution(self) -> Tuple[int, int, int, int]:
        F, H, W = self.data_cfg["input_size"]
        return (F, 3, H, W)

    @property
    def embed_dim(self) -> int:
        return self.featurizer.embed_dim

    @property
    def num_patches(self) -> int:
        # Hiera downsamples x2 three times, w/ initial patch embedding of 4x4
        # For the normal case, overall downsample is 2x temporal, 32x spatial
        f, h, w = [i // s for i, s in zip(self.data_cfg["input_size"], self.data_cfg["patch_stride"])]
        q_pool = self.data_cfg["q_pool"]
        assert h % 2**q_pool == 0 and w % 2**q_pool == 0, "Hiera patch size must be divisible by 2^q_pool"
        return f * h * w // 2**(2*q_pool)

    @property
    def spatial_resolution(self):
        _, h, w = [i // s for i, s in zip(self.data_cfg["input_size"], self.data_cfg["patch_stride"])]
        q_pool = self.data_cfg["q_pool"]
        assert h % 2**q_pool == 0 and w % 2**q_pool == 0, "Hiera patch size must be divisible by 2^q_pool"

        return h * w // 2**(2*q_pool)

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16
