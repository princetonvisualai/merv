"""
materialize.py

Factory class for initializing pretraining datasets on a per-VidLM basis; provides and exports individual functions for
clear control flow.
"""

from typing import List, Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from merv.conf import DatasetConfig
from merv.models.backbones.llm.prompting import PromptBuilder
from merv.models.backbones.video import VideoTransform
from merv.preprocessing.datasets import AlignVideoDataset, FinetuneVideoDataset
from merv.util.data_utils import PaddedCollatorForLanguageModeling

# Dataset Initializers =>> Maps Stage --> cls()
DATASET_INITIALIZER = {
    "align": AlignVideoDataset,
    "full-align": AlignVideoDataset,
    "finetune": FinetuneVideoDataset,
    "full-finetune": FinetuneVideoDataset,
}


def get_dataset_and_collator(
    stage: str,
    dataset_cfg: DatasetConfig,
    video_transforms: List[VideoTransform],
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    num_frames: List[int],
    default_video_resolutions: List[Tuple[int, int, int, int]],
    padding_side: str = "right",
) -> Tuple[Dataset, PaddedCollatorForLanguageModeling]:
    dataset_cls = DATASET_INITIALIZER[stage]
    dataset_root_dir = dataset_cfg.dataset_root_dir
    collator = PaddedCollatorForLanguageModeling(
        tokenizer.model_max_length,
        tokenizer.pad_token_id,
        default_video_resolutions,
        padding_side=padding_side,
    )

    # Switch on `stage`
    if stage.endswith("align"):
        annotation_json, image_dir = dataset_cfg.align_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            dataset_root_dir / image_dir,
            video_transforms,
            tokenizer,
            num_frames=num_frames,
        )
        return dataset, collator

    elif stage == "finetune" or stage == "second_finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            dataset_root_dir / image_dir,
            video_transforms,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
            num_frames=num_frames,
        )
        return dataset, collator

    else:
        raise ValueError(f"Stage `{stage}` is not supported!")
