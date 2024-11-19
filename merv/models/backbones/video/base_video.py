"""
base_video.py

Abstract class definition of a Video Backbone (Visual Featurizer), with full annotations of class methods, utility
functions, and initialization logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

# from functools import partial
from typing import Any, Callable, Dict, Protocol, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms.functional as TVF
from PIL.Image import Image


class ImageTransform(Protocol):
    def __call__(self, img: Image, **kwargs: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]: ...


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# === Custom Torchvision Image Transforms ===
@dataclass
class LetterboxPad:
    padding_fill_value: Tuple[int, int, int]

    def __call__(self, image: Image) -> Image:
        """Given a PIL.Image, pad to square by adding a symmetric border around the height/width."""
        (w, h), max_wh = image.size, max(image.size)
        horizontal_pad, vertical_pad = int((max_wh - w) / 2), int((max_wh - h) / 2)
        padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)
        return TVF.pad(image, padding, fill=self.padding_fill_value, padding_mode="constant")


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


# === Interface for a Video Transform ===
class VideoTransform(Protocol):
    def __call__(self, video: torch.Tensor, **kwargs: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]: ...


# === Abstract Base Class for arbitrary Video Backbones ===
class VideoBackbone(nn.Module, ABC):
    def __init__(
        self, video_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224, num_frames: int = 8
    ) -> None:
        super().__init__()
        self.identifier: str = video_backbone_id
        self.image_resize_strategy: str = image_resize_strategy
        self.default_image_size: int = default_image_size
        self.num_frames: int = num_frames

        # Instance attributes for a Video Backbone
        self.featurizer: nn.Module = None
        self.video_transform: VideoTransform = None

    def get_video_transform(self) -> VideoTransform:
        return self.video_transform

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable: ...

    @abstractmethod
    def forward(self, video_values: torch.Tensor, is_image: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the featurizer given a set of processed images, returning patch/grid features."""
        raise NotImplementedError

    @property
    @abstractmethod
    def default_video_resolution(self) -> Tuple[int, int, int, int]: ...

    @property
    @abstractmethod
    def embed_dim(self) -> int: ...

    @property
    @abstractmethod
    def num_patches(self) -> int: ...

    # Not always just divided by number of frames, depending on tubelet size, etc.
    @property
    @abstractmethod
    def spatial_resolution(self) -> int: ...

    @property
    def temporal_resolution(self) -> int:
        assert self.num_patches % self.spatial_resolution == 0
        return self.num_patches // self.spatial_resolution

    @property
    @abstractmethod
    def half_precision_dtype(self) -> torch.dtype: ...
