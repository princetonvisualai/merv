from functools import partial
from typing import Callable, Optional, Tuple

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


# class LanguageBindImageTower(nn.Module):
#     def __init__(self, image_tower, args, delay_load=False, cache_dir="./cache_dir"):
#         super().__init__()

#         self.is_loaded = False

#         self.image_tower_name = image_tower
#         self.select_layer = args.mm_vision_select_layer
#         self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

#         self.cache_dir = cache_dir

#         if not delay_load:
#             self.load_model()
#         else:
#             self.cfg_only = LanguageBindImageConfig.from_pretrained(
#                 self.image_tower_name, cache_dir=self.cache_dir
#             )

#     ############################################################
#     def load_model(self):
#         model = LanguageBindImage.from_pretrained(
#             self.image_tower_name, cache_dir=self.cache_dir
#         )
#         self.image_tower = model.vision_model
#         self.image_tower.requires_grad_(False)

#         self.image_processor = LanguageBindImageProcessor(model.config)

#         self.is_loaded = True

#     def feature_select(self, image_forward_outs):
#         image_features = image_forward_outs.hidden_states[self.select_layer]
#         if self.select_feature == "patch":
#             image_features = image_features[:, 1:]
#         elif self.select_feature == "cls_patch":
#             image_features = image_features
#         else:
#             raise ValueError(f"Unexpected select feature: {self.select_feature}")
#         return image_features

#     @torch.no_grad()
#     def forward(self, images):
#         if type(images) is list:
#             image_features = []
#             for image in images:
#                 image_forward_out = self.image_tower(
#                     image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
#                     output_hidden_states=True,
#                 )
#                 image_feature = self.feature_select(image_forward_out).to(image.dtype)
#                 image_features.append(image_feature)
#         else:
#             # print('images', images.shape)
#             image_forward_outs = self.image_tower(
#                 images.to(device=self.device, dtype=self.dtype),
#                 output_hidden_states=True,
#             )
#             # print('image_forward_outs', len(image_forward_outs), image_forward_outs[0].shape)
#             image_features = self.feature_select(image_forward_outs).to(images.dtype)
#             # print('image_features', image_features.shape)

#         return image_features

#     @property
#     def dummy_feature(self):
#         return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

#     @property
#     def dtype(self):
#         return self.image_tower.embeddings.class_embedding.dtype  #############

#     @property
#     def device(self):
#         return self.image_tower.embeddings.class_embedding.device  ##############

#     @property
#     def config(self):
#         if self.is_loaded:
#             return self.image_tower.config
#         else:
#             return self.cfg_only

#     @property
#     def hidden_size(self):
#         return self.config.hidden_size

#     @property
#     def num_patches(self):
#         return (self.config.image_size // self.config.patch_size) ** 2


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
            "LanguageBind/LanguageBind_Video_merge", cache_dir="./cache_dir"
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
