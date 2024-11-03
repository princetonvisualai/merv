"""
datasets.py

Draccus Dataclass Definition for a DatasetConfig object, with various registered subclasses for each dataset variant
and processing scheme. A given dataset variant (e.g., `llava-lightning`) configures the following attributes:
    - Dataset Variant (Identifier) --> e.g., "llava-v15"
    - Align Stage Dataset Components (annotations, images)
    - Finetune Stage Dataset Components (annotations, images)
    - Dataset Root Directory (Path)
"""

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Tuple

from draccus import ChoiceRegistry


@dataclass
class DatasetConfig(ChoiceRegistry):
    # fmt: off
    dataset_id: str                                 # Unique ID that fully specifies a dataset variant

    # Dataset Components for each Stage in < align | finetune >
    align_stage_components: Tuple[Path, Path]       # Path to annotation file and images directory for `align` stage
    finetune_stage_components: Tuple[Path, Path]    # Path to annotation file and images directory for `finetune` stage

    dataset_root_dir: Path                          # Path to dataset root directory; others paths are relative to root
    # fmt: on


# LLaVA-v15 + Valley for alignment, LLaVa-v15 + VideoChatGPT for finetuning (combined datasets)
@dataclass
class VideoLLaVA_Config(DatasetConfig):
    dataset_id: str = "videollava"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/videollava/valley_llavaimage.json"),
        Path("download/videollava/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/videollava/videochatgpt_llavaimage_tune.json"),
        Path("download/videollava/"),
    )
    dataset_root_dir: Path = Path("data/")


# === Define a Dataset Registry Enum for Reference & Validation =>> all *new* datasets must be added here! ===
@unique
class DatasetRegistry(Enum):
    # === LLaVa v1.5 ===
    VIDEOLLAVA = VideoLLaVA_Config

    @property
    def dataset_id(self) -> str:
        return self.value.dataset_id


# Register Datasets in Choice Registry
for dataset_variant in DatasetRegistry:
    DatasetConfig.register_subclass(dataset_variant.dataset_id, dataset_variant.value)
