from functools import partial
from typing import Callable, Optional, Tuple
import os

import torch
from torch import nn
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy

from merv.models.backbones.video import VideoBackbone

from .image.configuration_image import LanguageBindImageConfig
from .image.modeling_image import LanguageBindImage
from .image.processing_image import LanguageBindImageProcessor
from .video.configuration_video import LanguageBindVideoConfig
from .video.modeling_video import CLIPEncoderLayer, CLIPVisionTransformer, LanguageBindVideo
from .video.processing_video import LanguageBindVideoProcessor, get_video_transform

config_dict = {
    "image": LanguageBindImageConfig,
    "video": LanguageBindVideoConfig,
}
model_dict = {
    "image": LanguageBindImage,
    "video": LanguageBindVideo,
}
transform_dict = {
    "video": LanguageBindVideoProcessor,
    "image": LanguageBindImageProcessor,
}
HF_HUB_CACHE = os.getenv("HF_HUB_CACHE", "~/.cache/huggingface/hub")


class LangBindVideoBackbone(VideoBackbone):
    def __init__(
        self,
        video_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        num_frames: int = 8,
        token: Optional[str] = None,
    ) -> None:
        """
        video_backbone_id: should be always languagebind-video
        image_resize_strategy: not used here. passed to video pre-processor.
        default_image_size: fixed value of 224?
        num_frames:
        token: None = use all 257 tokens, average = average per frame, classemb = class token
        """
        super().__init__(
            video_backbone_id,
            image_resize_strategy,
            default_image_size=default_image_size,
            num_frames=num_frames,
        )

        # this is called in merv/models/materialize.py, L104
        assert "languagebind-video" in video_backbone_id, video_backbone_id
        self.video_backbone_id = video_backbone_id
        self.token = token
        self.featurizer = LanguageBindVideo.from_pretrained(
            "LanguageBind/LanguageBind_Video_merge", 
            cache_dir=HF_HUB_CACHE,
        ).vision_model
        assert image_resize_strategy == "resize-naive"

        self.featurizer.eval()
        self.featurizer.requires_grad_(False)
        self.video_processor = get_video_transform()

        # required item. Expected input is [F, C, H, W]
        self.video_transform = lambda video: self.video_processor(video.permute(1, 0, 2, 3))

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps each ViT block and then both of the _entire_ featurizers."""
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={CLIPVisionTransformer})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={CLIPEncoderLayer})
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(self, video_values: torch.Tensor, is_image: torch.Tensor) -> torch.Tensor:
        # input is expected to be [B, C, F, H, W]
        B = video_values.shape[0]

        video_forward_outs = self.featurizer(video_values, output_hidden_states=True)
        # video_features = self.feature_select(video_features)
        video_features = video_forward_outs.hidden_states[-2]
        # self.featurizer output is expected to be batch_size, num_frames, 257, 1024
        assert video_features.shape[-2] == 257, video_features.shape

        if self.token == "average":
            video_features = video_features.mean(-2)
        elif self.token == "classemb":
            video_features = video_features[:, :, 0, :]
        elif self.token == "noclass":
            video_features = video_features[:, :, 1:, :]
        elif self.token == "classemb-at-first":
            classtoken = video_features[:, :, 0, :].mean(1, keepdim=True)
            video_features = video_features[:, :, 1:, :]
            video_features = torch.concat([classtoken, video_features.reshape(B, -1, self.embed_dim)], 1)

        # since final output wants the shape to be [batchsize, num_tokens, emb channel], we do:
        video_features = video_features.reshape(B, -1, self.embed_dim)

        return video_features

    @property
    def embed_dim(self) -> int:
        return 1024

    @property
    def default_video_resolution(self) -> Tuple[int, int, int, int]:
        # model shape after transform.
        return (3, self.num_frames, 224, 224)

    @property
    def num_patches(self) -> int:
        if self.token is None:
            return self.num_frames * 257
        elif self.token == "average":
            return self.num_frames
        elif self.token == "classemb":
            return self.num_frames
        elif self.token == "noclass":
            return self.num_frames * 256
        elif self.token == "classemb-at-first":
            return self.num_frames * 256
        else:
            return self.num_frames * 257

    @property
    def spatial_resolution(self) -> int:
        return self.num_patches // self.num_frames

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16
