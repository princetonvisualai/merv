"""
datasets.py

PyTorch Dataset Definitions for MERV models; supports processing for both the `align` and `finetune` stages, with
utilities for formatting conversations during the `finetune` stage subject to the given LLM backbone's expected
formatting (e.g., SYS_PROMPT + USER: ... ASSISTANT: ... for Vicuña v1.5 Chat models).

We currently only support Map-style Datasets; assumes that all files (annotations, images) are on local disk, and that
random access image reading is relatively cheap/fast.
"""

import copy
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Type

import cv2
import decord
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image, ImageSequence
from torch.utils.data import Dataset
from transformers import LlamaTokenizerFast, PreTrainedTokenizerBase
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from merv.models.backbones.llm.prompting import PromptBuilder
from merv.models.backbones.video import VideoTransform

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def load_video(
    video_path: Path,
    decode_backend="decord",
    clip_start_sec=0.0,
    clip_end_sec=None,
    num_frames=8,
    end_frame=None,
) -> torch.Tensor:
    # very annoying: test set in TVQA has one nan, nan pair
    # For Hound, we need to add a special load since the videos are provided as frames

    if clip_start_sec is not None:
        if math.isnan(clip_start_sec):
            clip_start_sec = 0

    if clip_end_sec is not None:
        if math.isnan(clip_end_sec):
            clip_end_sec = None

    """Load video from disk and return as a torch.Tensor."""
    if decode_backend == "decord":
        decord.bridge.set_bridge("torch")
        if Path(
            video_path
        ).is_dir():  # If path points to dir that contains the frames for the video (for VLEP we have a folder with frames instead of .mp4/.avi file. The dataset gives 3 FPS instead of video)
            video_path = Path(video_path)
            if (
                "vlep" in str(video_path).lower()
            ):  # VLEP is 3 fps. see: https://github.com/jayleicn/VideoLanguageFuturePred/tree/main/data
                fps_in_dir = 3
                images = sorted([str(img_path) for img_path in video_path.glob("*.jpg")])
                assert len(images) > 0, "video directory contains no frames to load video - " + video_path
                video_num_frames = len(images)
                total_secs = video_num_frames / fps_in_dir

                # Handle Clip Start / End
                if clip_end_sec is None:
                    clip_end_sec = total_secs
                frame_id_list = np.linspace(
                    clip_start_sec * fps_in_dir,
                    min(video_num_frames - 1, clip_end_sec * fps_in_dir - 1),
                    num_frames,
                    dtype=int,
                )

                video_data = []
                for frame_id in frame_id_list:
                    frame_id = int(frame_id)
                    if frame_id < len(images * fps_in_dir):
                        image = cv2.imread(images[frame_id])
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # covert from BGR to RGB (opencv)
                        video_data.append(image)

                video_data = torch.from_numpy(np.stack(video_data)).permute(
                    0, 3, 1, 2
                )  # [T, H, W, C] -> [T, C, H, W] for pre-processing

            elif "sharegpt" in str(video_path).lower():
                images = sorted([str(img_path) for img_path in video_path.glob("*.jpeg")])
                assert len(images) > 0, "video directory contains no frames to load video - " + video_path
                video_num_frames = len(images)

                frame_id_list = np.linspace(
                    0,
                    video_num_frames - 1,
                    num_frames,
                    dtype=int,
                )

                video_data = []
                for frame_id in frame_id_list:
                    image = cv2.imread(images[frame_id])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # covert from BGR to RGB (opencv)
                    video_data.append(image)

                video_data = torch.from_numpy(np.stack(video_data)).permute(
                    0, 3, 1, 2
                )  # [T, H, W, C] -> [T, C, H, W] for pre-processing
            else:  # default just in case for in future
                raise NotImplementedError

        elif Path(video_path).suffix == ".gif":
            im = Image.open(str(video_path))
            frames = np.stack([np.array(f.convert("RGB")) for f in ImageSequence.Iterator(im)], 0)
            frames = torch.from_numpy(frames)
            video_num_frames = frames.shape[0]
            frame_id_list = np.linspace(0, video_num_frames - 1, num_frames, dtype=int)
            video_data = frames[frame_id_list].permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W] for pre-processing

        else:
            decord_vr = VideoReader(str(video_path), ctx=cpu(0))
            video_num_frames = len(decord_vr)
            avg_fps = decord_vr.get_avg_fps()
            total_secs = video_num_frames / avg_fps

            # Handle Clip Start / End
            if end_frame is None or end_frame < 0:
                if clip_end_sec is None:
                    clip_end_sec = total_secs

                frame_id_list = np.linspace(
                    clip_start_sec * avg_fps, min(video_num_frames - 1, clip_end_sec * avg_fps - 1), num_frames, dtype=int
                )
            else:
                frame_id_list = np.linspace(
                    0, min(video_num_frames - 1, end_frame), num_frames, dtype=int
                )

            import os
            if os.path.basename(video_path) in ['l0w4V7yPdPJQQphx.mp4', 'x4oT5lcBVwKl9s27.mp4']:

                assert num_frames == 32
                tmp = []
                for i in range(4):
                    idx = frame_id_list[8*i:8*i+8]
                    decord_vr = VideoReader(str(video_path), ctx=cpu(0))
                    tmp.append(decord_vr.get_batch(idx))
                video_data = torch.concat(tmp, 0)
            else:
                video_data = decord_vr.get_batch(frame_id_list)
            del decord_vr
            video_data = video_data.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W] for pre-processing

        return video_data
    else:
        raise NameError(f"Unknown decode backend: {decode_backend}")


class AlignVideoDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        chat_json: Path,
        image_dir: Path,
        video_dir: Path,
        video_transforms: List[VideoTransform],
        tokenizer: PreTrainedTokenizerBase,
        num_frames: int,
    ) -> None:
        super().__init__()
        self.chat_json, self.image_dir = chat_json, image_dir
        self.tokenizer = tokenizer
        self.video_dir, self.video_transforms = video_dir, video_transforms
        self.dataset_type = "align"
        self.num_frames = num_frames

        # Create Prompt Template
        self.prompt_template = "{caption}" + self.tokenizer.eos_token

        # Load Chat JSON
        with open(self.chat_json, "r") as f:
            self.examples = json.load(f)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            conversation = self.examples[idx]["conversations"]
            assert (
                (len(conversation) == 2)
                and ("<image>" not in conversation[-1]["value"])
                and ("<video>" not in conversation[-1]["value"])
            ), "Unexpected text!"

            # Format Caption --> {caption}{eos_token}
            caption = self.prompt_template.format(caption=conversation[-1]["value"].strip())
            input_ids = self.tokenizer(caption, truncation=True, return_tensors="pt").input_ids[0]
            labels = copy.deepcopy(input_ids)

            # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
            labels[0] = IGNORE_INDEX

            video_values = [None for _ in self.video_transforms]
            is_image = False

            if "image" in self.examples[idx]:
                image_path = Path(self.examples[idx]["image"])
                image = Image.open(self.image_dir / image_path).convert("RGB")
                video = torch.from_numpy(np.array(image).transpose(2, 0, 1)[None,].repeat(max(self.num_frames), 0))
                video_values = [
                    video_transform(video[:: max(self.num_frames) // num_frame])
                    for video_transform, num_frame in zip(self.video_transforms, self.num_frames)
                ]
                is_image = True

            if "video" in self.examples[idx]:
                video_path = Path(self.examples[idx]["video"])
                video = load_video(self.video_dir / video_path, num_frames=max(self.num_frames))
                video_values = [
                    video_transform(video[:: max(self.num_frames) // num_frame])
                    for video_transform, num_frame in zip(self.video_transforms, self.num_frames)
                ]

        except Exception as e:
            print(f"Error in processing example {idx}!")
            print(e)
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))

        return dict(
            video_values=video_values,
            input_ids=input_ids,
            labels=labels,
            is_image=is_image,
        )

    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example or "video" in example
            n_words = sum([len(turn["value"].split()) for turn in example["conversations"]])
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)


class FinetuneVideoDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        instruct_json: Path,
        image_dir: Path,
        video_dir: Path,
        video_transforms: List[VideoTransform],
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
        num_frames: int,
    ) -> None:
        super().__init__()
        self.instruct_json, self.image_dir = instruct_json, image_dir
        self.tokenizer = tokenizer
        self.video_dir, self.video_transforms = video_dir, video_transforms
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_type = "finetune"
        self.num_frames = num_frames

        self.embedding_dim = 4096

        # Load Instruct JSON
        with open(self.instruct_json, "r") as f:
            self.examples = json.load(f)

    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # ruff: noqa: E501
        """
        Unlike the *align* stage handling, for the *finetune* stage, we actually need to handle multiple "turns" of
        dialog grounded in a single image.

        To do this, we leverage the `prompt_builder_fn` which instantiates a PromptBuilder object. By calling the
        methods for adding turns and getting a prompt, we ensure proper formatting and consistency for each example.

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"video_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        try:
            conversation = self.examples[idx]["conversations"]

            # Create Prompt Builder --> add each message sequentially
            prompt_builder, input_ids, labels = self.prompt_builder_fn(model_family="merv"), [], []

            for turn_idx, turn in enumerate(conversation):
                # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
                # msg = prompt_builder.add_turn(turn["from"], turn["value"].replace("<image>", "").replace("<video>",""))
                msg = prompt_builder.add_turn(turn["from"], turn["value"])

                # Llama Tokenizer (Fast) adds extra character if a string ends in whitespace --> strip if non-empty!
                if isinstance(self.tokenizer, LlamaTokenizerFast) or isinstance(self.tokenizer, PreTrainedTokenizerFast):
                    msg = msg.rstrip()
                else:
                    raise ValueError(f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!")

                # Tokenize Input IDs
                turn_input_ids = self.tokenizer(msg, add_special_tokens=turn_idx == 0).input_ids

                # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
                turn_labels = (
                    [IGNORE_INDEX for _ in range(len(turn_input_ids))] if (turn_idx % 2) == 0 else list(turn_input_ids)
                )

                # TODO: for llama3.1: remove '<|start_header_id|>assistant<|end_header_id|>\n\n' and '<|eot_id|>
                input_ids.extend(turn_input_ids)
                labels.extend(turn_labels)

            # Tensorize =>> Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches after)
            #   - IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
            input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

            # Handle Truncation (if necessary)
            input_ids, labels = input_ids[: self.tokenizer.model_max_length], labels[: self.tokenizer.model_max_length]

            # === Handle "unimodal" (language-only) vs. "multimodal" ===
            # Store "image"-patches (pixel) and "video"-patches (video) separately
            video_values = [None for _ in self.video_transforms]
            is_image = False
            if "image" in self.examples[idx]:
                image_path = Path(self.examples[idx]["image"])

                # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
                labels[0] = IGNORE_INDEX

                # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
                image = Image.open(self.image_dir / image_path).convert("RGB")
                video = torch.from_numpy(np.array(image).transpose(2, 0, 1)[None,].repeat(max(self.num_frames), 0))
                video_values = [
                    video_transform(video[:: max(self.num_frames) // num_frame])
                    for video_transform, num_frame in zip(self.video_transforms, self.num_frames)
                ]
                is_image = True

            if "video" in self.examples[idx]:
                video_path = Path(self.examples[idx]["video"])

                # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the video patches right after)
                labels[0] = IGNORE_INDEX

                video = load_video(self.video_dir / video_path, num_frames=max(self.num_frames))
                video_values = [
                    video_transform(video[:: max(self.num_frames) // num_frame])
                    for video_transform, num_frame in zip(self.video_transforms, self.num_frames)
                ]

        except Exception as e:
            print(f"Error in processing example {idx}!")
            print(e)
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))

        # ! Key assumption: only one image or video per example, but OK w/in batch
        return dict(
            video_values=video_values,
            input_ids=input_ids,
            labels=labels,
            is_image=is_image,
        )

    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example or "video" in example
            n_words = sum([len(turn["value"].split()) for turn in example["conversations"]])
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)
