"""
materialize.py

Factory class for initializing Vision Backbones, LLM Backbones, and VidLMs from a set registry; provides and exports
individual functions for clear control flow.
"""

from typing import List, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from merv.models.backbones.llm import LLaMa2LLMBackbone, LLaMa3LLMBackbone, LLaMA31LLMBackbone, LLMBackbone
from merv.models.backbones.video import (
    DinoV2VideoBackbone,
    LangBindVideoBackbone,
    HieraVideoBackbone,
    SiglipVideoBackbone,
    VideoBackbone,
    VideoTransform,
    ViVITVideoBackbone,
)
from merv.models.vidlms import MERV

# === Registries =>> Maps ID --> {cls(), kwargs} :: Different Registries for Vision Backbones, LLM Backbones, VidLMs ===
# fmt: off



# === Video Backbone Registry ===
# ruff: noqa: E501
VIDEO_BACKBONES = {
    # === DINOv2 Video Backbone ===
    "dinov2-video": {"cls": DinoV2VideoBackbone, "kwargs": {"default_image_size": 224}},
    'dinov2-video-all-tokens': {"cls": DinoV2VideoBackbone, "kwargs": {"default_image_size": 224}},
    'dinov2-video-all-token-with-cls': {"cls": DinoV2VideoBackbone, "kwargs": {"default_image_size": 224}},
    'dinov2-video-classemb-at-first': {"cls": DinoV2VideoBackbone, "kwargs": {"default_image_size": 224}},

    # === LanguageBind Video Backbone ===
    'languagebind-video': {"cls": LangBindVideoBackbone, "kwargs": {"default_image_size": 224}},
    'languagebind-video-averagetoken': {"cls": LangBindVideoBackbone, "kwargs": {"default_image_size": 224, 'token': 'average'}},
    'languagebind-video-classemb': {"cls": LangBindVideoBackbone, "kwargs": {"default_image_size": 224, 'token': 'classemb'}},
    'languagebind-video-noclass': {"cls": LangBindVideoBackbone, "kwargs": {"default_image_size": 224, 'token': 'noclass'}},
    'languagebind-video-classemb-at-first': {"cls": LangBindVideoBackbone, "kwargs": {"default_image_size": 224, 'token': 'classemb-at-first'}},

    # === ViViT Video Backbone ===
    'vivit-google-b-cls-token': {"cls": ViVITVideoBackbone, "kwargs": {"default_image_size": 224}},
    'vivit-google-b-all-tokens': {"cls": ViVITVideoBackbone, "kwargs": {"default_image_size": 224}},
    'vivit-google-b-all-no-cls': {"cls": ViVITVideoBackbone, "kwargs": {"default_image_size": 224}},
    'vivit-google-b-all-no-cls-16frames': {"cls": ViVITVideoBackbone, "kwargs": {"default_image_size": 224}},
    'vivit-google-b-classemb-at-first-16frames': {"cls": ViVITVideoBackbone, "kwargs": {"default_image_size": 224}},

    # === SigLIP Video Backbone ===
    "siglip-vit-b16-224px": {"cls": SiglipVideoBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-b16-224px-all-tokens": {"cls": SiglipVideoBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-b16-224px-all-no-cls": {"cls": SiglipVideoBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-b16-224px-classemb-at-first": {"cls": SiglipVideoBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-b16-256px": {"cls": SiglipVideoBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-b16-256px-all-tokens": {"cls": SiglipVideoBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-b16-384px": {"cls": SiglipVideoBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-b16-384px-all-tokens": {"cls": SiglipVideoBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-so400m": {"cls": SiglipVideoBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-so400m-all-tokens": {"cls": SiglipVideoBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-so400m-384px": {"cls": SiglipVideoBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-so400m-384px-all-tokens": {"cls": SiglipVideoBackbone, "kwargs": {"default_image_size": 224}},

    # === Hiera Video Backbone ===
    "hiera-base-video": {"cls": HieraVideoBackbone, "kwargs": {"default_image_size": 224}},
    "hiera-base-plus-video": {"cls": HieraVideoBackbone, "kwargs": {"default_image_size": 224}},
    "hiera-large-video": {"cls": HieraVideoBackbone, "kwargs": {"default_image_size": 224}},
}


# === Language Model Registry ===
LLM_BACKBONES = {
    # === LLaMa-2 Pure (Non-Chat) Backbones ===
    "llama2-7b-pure": {"cls": LLaMa2LLMBackbone, "kwargs": {}},
    "llama2-13b-pure": {"cls": LLaMa2LLMBackbone, "kwargs": {}},

    # === LLaMa-2 Chat Backbones ===
    "llama2-7b-chat": {"cls": LLaMa2LLMBackbone, "kwargs": {}},
    "llama2-13b-chat": {"cls": LLaMa2LLMBackbone, "kwargs": {}},

    # === Vicuna-v1.5 Backbones ===
    "vicuna-v15-7b": {"cls": LLaMa2LLMBackbone, "kwargs": {}},
    "vicuna-v15-13b": {"cls": LLaMa2LLMBackbone, "kwargs": {}},

    # === LLaMa-3 Pure + Chat Backbones ===
    "llama3-8b-pure": {"cls": LLaMa3LLMBackbone, "kwargs": {}},
    "llama3-8b-chat": {"cls": LLaMa3LLMBackbone, "kwargs": {}},

    # === LLaMa-3.1 Chat Backbones ===
    "llama3.1-8b-chat": {"cls": LLaMA31LLMBackbone, "kwargs": {}},
}


# fmt: on


def get_video_backbone_and_transform(
    video_backbone_ids: List[str], image_resize_strategy: str, num_frames: List[int]
) -> Tuple[List[VideoBackbone], List[VideoTransform]]:
    """Instantiate a Vision Backbone, returning both the nn.Module wrapper class and default Image Transform."""

    video_backbones = []
    video_transforms = []

    for video_backbone_id, num_frame in zip(video_backbone_ids, num_frames):
        if video_backbone_id in VIDEO_BACKBONES:
            video_cfg = VIDEO_BACKBONES[video_backbone_id]
            video_backbone: VideoBackbone = video_cfg["cls"](
                video_backbone_id, image_resize_strategy, num_frames=num_frame, **video_cfg["kwargs"]
            )
            video_transform = video_backbone.get_video_transform()

            video_backbones.append(video_backbone)
            video_transforms.append(video_transform)

        else:
            raise ValueError(f"Video Backbone `{video_backbone_id}` is not supported!")

    return video_backbones, video_transforms


def get_llm_backbone_and_tokenizer(
    llm_backbone_id: str,
    llm_max_length: int = 2048,
    hf_token: Optional[str] = None,
    inference_mode: bool = False,
) -> Tuple[LLMBackbone, PreTrainedTokenizerBase]:
    if llm_backbone_id in LLM_BACKBONES:
        llm_cfg = LLM_BACKBONES[llm_backbone_id]
        llm_backbone: LLMBackbone = llm_cfg["cls"](
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            **llm_cfg["kwargs"],
        )
        tokenizer = llm_backbone.get_tokenizer()
        return llm_backbone, tokenizer

    else:
        raise ValueError(f"LLM Backbone `{llm_backbone_id}` is not supported!")


def get_vidlm(
    model_id: str,
    arch_specifier: str,
    video_backbones: List[VideoBackbone],
    llm_backbone: LLMBackbone,
    adapter: Optional[str],
    projector_token_length: int,
    visual_feature_length: int,
    enable_mixed_precision_training: bool = True,
) -> MERV:
    """Lightweight wrapper around initializing a VidLM, mostly for future-proofing (if one wants to add a new VidLM)."""
    return MERV(
        model_id,
        video_backbones,
        llm_backbone,
        enable_mixed_precision_training=enable_mixed_precision_training,
        arch_specifier=arch_specifier,
        adapter=adapter,
        projector_token_length=projector_token_length,
        visual_feature_length=visual_feature_length,
    )
