from functools import partial
from typing import Callable, Tuple

import torch
import torchvision.transforms as transforms
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms import functional as F
from transformers import VivitImageProcessor, VivitModel
from transformers.models.vivit.modeling_vivit import VivitLayer

from merv.models.backbones.video import VideoBackbone
from merv.models.backbones.video.base_video import LetterboxPad

ViVIT_VISION_BACKBONES = {
    "vivit-google-b-cls-token": "google/vivit-b-16x2-kinetics400",
    "vivit-google-b-all-tokens": "google/vivit-b-16x2-kinetics400",
    "vivit-google-b-all-no-cls": "google/vivit-b-16x2-kinetics400",
    "vivit-google-b-all-no-cls-16frames": "google/vivit-b-16x2-kinetics400",
    "vivit-google-b-classemb-at-first-16frames": "google/vivit-b-16x2-kinetics400",
}


class ViVITVideoBackbone(VideoBackbone):
    def __init__(
        self, video_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224, num_frames: int = 32
    ) -> None:
        super().__init__(
            video_backbone_id,
            image_resize_strategy,
            default_image_size=default_image_size,
            num_frames=num_frames,
        )

        self.video_backbone_id = video_backbone_id

        # if "vivit-google-b-" in self.video_backbone_id:
        #     assert num_frames == 32, "Selected ViVIT checkpoint requires 32 frames per second for model"

        self.huggingface_path_or_url = ViVIT_VISION_BACKBONES[video_backbone_id]

        self.featurizer: VivitModel = VivitModel.from_pretrained(
            self.huggingface_path_or_url,
            add_pooling_layer=False,
            image_size=self.default_image_size,
            num_frames=num_frames,
        )
        self.featurizer.eval()

        self.data_input_size = (3, self.default_image_size, self.default_image_size)
        image_processor = VivitImageProcessor.from_pretrained(self.huggingface_path_or_url)

        default_image_transform = []

        default_image_transform.append(transforms.Resize(image_processor.size["shortest_edge"]))
        default_image_transform.append(transforms.CenterCrop(self.default_image_size))
        default_image_transform.append(ToTensor())
        default_image_transform.append(
            transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        )
        default_image_transform = Compose(default_image_transform)

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

            # Compute Padding Fill Value (rescaled normalization mean if applicable)
            fill = tuple([int(x * 255) for x in image_processor.image_mean])

            # Build New Transform
            image_transform = Compose([LetterboxPad(fill), *default_image_transform.transforms])

        else:
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!")

        self.video_transform_ = Compose([F.to_pil_image, *image_transform.transforms])
        self.video_transform = lambda video: torch.stack([self.video_transform_(frame) for frame in video])

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps each ViT block and then both of the _entire_ featurizers."""
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={VivitModel})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={VivitLayer})
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(self, video_values: torch.Tensor, is_image: torch.Tensor) -> torch.Tensor:
        # Forward Pass through the Featurizer (ViT)

        B, F, C, H, W = video_values.shape
        output_features = self.featurizer(video_values)
        video_features = output_features.last_hidden_state

        if "cls-token" in self.video_backbone_id:
            video_features = video_features[:, 0].unsqueeze(1)
        elif "all-no-cls" in self.video_backbone_id:
            video_features = video_features[:, 1:]  # [B, (16 x 14 x 14), D]
            video_features = video_features.reshape(B, 16, 14, 14, -1)

            if "16frames" in self.video_backbone_id:
                return video_features.reshape(B, 16 * 14 * 14, -1)

            video_features = video_features[:, ::2]
            video_features = video_features.reshape(B, 8 * 14 * 14, -1)
        return video_features

    @property
    def default_video_resolution(self) -> Tuple[int, int, int, int]:
        return (self.num_frames, *self.data_input_size)

    @property
    def embed_dim(self) -> int:
        return self.featurizer.config.hidden_size

    @property
    def num_patches(self) -> int:
        if "cls-token" in self.video_backbone_id:
            return 1
        elif "all-tokens" in self.video_backbone_id:
            if self.huggingface_path_or_url == "google/vivit-b-16x2-kinetics400":
                return 3137
        elif "all-no-cls-16frames" in self.video_backbone_id:
            return 3136
        elif "all-no-cls" in self.video_backbone_id:
            return 3136 // 2
        elif "classemb-at-first" in self.video_backbone_id:
            return 3136
        else:
            raise NotImplementedError

    @property
    def spatial_resolution(self) -> int:
        if "all-no-cls" in self.video_backbone_id:
            return 196
        elif "classemb-at-first" in self.video_backbone_id:
            return 196

        return self.num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16
