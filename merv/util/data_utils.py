"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_video_resolutions: List[Tuple[int, int, int, int]]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_video_values = [
            torch.zeros(default_video_resolution, dtype=self.pixel_values_dtype)
            for default_video_resolution in self.default_video_resolutions
        ]

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        video_values = [instance["video_values"] for instance in instances]
        is_image = [instance["is_image"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(instances)) if (video_values[idx][0] is not None)],
            dtype=torch.long,
        )

        if len(multimodal_indices) == 0:
            video_values = [
                torch.stack([dummy_video_value for _ in range(len(input_ids))])
                for dummy_video_value in self.dummy_video_values
            ]
        else:
            video_values = [
                torch.stack(
                    [
                        (
                            video_value[encoder_idx]
                            if video_value[encoder_idx] is not None
                            else self.dummy_video_values[encoder_idx]
                        )
                        for video_value in video_values
                    ]
                )
                for encoder_idx in range(len(video_values[0]))
            ]

        return dict(
            video_values=video_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
            is_image=torch.tensor(is_image),
        )
