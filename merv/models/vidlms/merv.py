"""
merv.py

PyTorch Module defining MERV, our general interface for defining the various different VidLMs in our work.

Notes from Prismatic:
    - For now, we don't subclass `transformers.PretrainedModel` (or CausalLM). Instead, we assume a very limited subset
      of the {Model}ForCausalLM API that enables dispatch to the underlying LLM's `generate` utilities (feeding inputs
      through our custom projection shim).
"""

# ruff: noqa

from __future__ import annotations

import os
from PIL import Image
import numpy as np
import re
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type

import torch
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers.modeling_outputs import CausalLMOutputWithPast

from merv.models.backbones.llm import LLMBackbone
from merv.models.backbones.llm.prompting import PromptBuilder
from merv.models.backbones.video import VideoBackbone
from merv.models.vidlms.base_vidlm import VidLM
from merv.overwatch import initialize_overwatch
from merv.preprocessing.datasets.datasets import load_video
from merv.util.nn_utils import (
    AttentivePooler,
    AveragePooling3DProjector,
    AveragePoolingProjector,
    Convolutional3DProjector,
    ConvolutionalProjector,
    CrossAttentionAdapterLearnableQuery,
    FusedMLPProjector,
    LinearProjector,
    MLPDeepProjector,
    MLPProjector,
    ScalarAdapter,
)

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class MERV(VidLM):
    def __init__(
        self,
        model_id: str,
        video_backbones: List[VideoBackbone],
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        adapter: Optional[str] = None,
        projector_token_length: int = 64,
        visual_feature_length: int = 512,
        pre_proj_layernorm: bool = False,
        normalize_post_proj: bool = False,
        text_embedding_dim: int = 3072,
    ) -> None:
        super().__init__(
            "merv",
            model_id,
            video_backbones,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
        )

        # set adapter strategy
        self.feature_fusion_type = adapter
        self.pre_proj_layernorm = pre_proj_layernorm
        self.normalize_post_proj = normalize_post_proj
        self.text_embedding_dim = text_embedding_dim

        # set number of frames per encoder
        # Set Weight Initialization Seed for Projector Consistency
        torch.manual_seed(video_backbones[0].embed_dim)

        # Initialize Projection (Adapter) based on `arch_specifier`
        self.arch_specifier = arch_specifier
        if arch_specifier.endswith("linear"):
            mlp_type = "linear"
            Projector = LinearProjector
        elif arch_specifier.endswith("fused-gelu-mlp"):
            mlp_type = "fused-gelu-mlp"
            Projector = FusedMLPProjector
        elif arch_specifier.endswith("gelu-mlp"):
            mlp_type = "gelu-mlp"
            Projector = MLPProjector
        elif arch_specifier.endswith("none"):
            # will get overriden later
            mlp_type = "none"
            Projector = torch.nn.Identity()
        else:
            raise ValueError(f"MERV with `{arch_specifier = }` is not supported!")

        # Override Projection with one for multiple encoders
        self.tokens_resampled = False
        # If we want to downsample the number of frames, for use in the Avg3D/Conv3D
        factor = 1
        projector_output_size = int(projector_token_length**0.5)
        assert projector_token_length == projector_output_size**2, "projector_token_length should be square number"

        def extract_frame_number(query):
            # Returns the num in "frame{num}"
            return int(re.search(r"frame(\d+)", query).group(1))

        if "avg" in arch_specifier.split("+"):
            # initialize this one first b/c it depends on each video_backbone
            self.tokens_resampled = True
            Projector = partial(
                AveragePoolingProjector,
                output_size=projector_output_size,
            )
        elif "attntv" in arch_specifier.split("+"):
            self.tokens_resampled = True
            Projector = partial(
                AttentivePooler,
                num_query_tokens=projector_token_length,
                num_heads=8,
            )
        elif "conv" in arch_specifier.split("+"):
            self.tokens_resampled = True
            Projector = partial(ConvolutionalProjector, output_size=projector_output_size, block_depth=3)
        elif "3davg" in arch_specifier.split("+"):
            if "frame" in arch_specifier:
                factor = extract_frame_number(arch_specifier)
            self.tokens_resampled = True
            Projector = partial(
                AveragePooling3DProjector,
                output_size=projector_output_size,
            )
        elif "3dconv" in arch_specifier.split("+"):
            if "frame" in arch_specifier:
                factor = extract_frame_number(arch_specifier)
            self.tokens_resampled = True
            Projector = partial(
                Convolutional3DProjector,
                output_size=projector_output_size,
            )

        if self.tokens_resampled:
            self.projectors = torch.nn.ModuleList(
                [
                    Projector(
                        video_backbone.embed_dim,
                        llm_backbone.embed_dim,
                        output_frames=video_backbone.temporal_resolution // factor,
                        mlp_type=mlp_type,
                    )
                    for video_backbone in video_backbones
                ]
            )
        else:
            self.projectors = torch.nn.ModuleList(
                [
                    Projector(
                        video_backbone.embed_dim, llm_backbone.embed_dim, pre_proj_layernorm=self.pre_proj_layernorm
                    )
                    for video_backbone in video_backbones
                ]
            )

        # * Make sure that the output token length is consistent across all projectors
        if len(self.video_backbones) > 1:
            if self.tokens_resampled:
                assert all(
                    projector.output_token_length * projector.output_frame_length in [1, visual_feature_length]
                    for projector in self.projectors
                ), (
                    "Output token length is not consistent across all projectors!"
                    f" visual_feature_length={visual_feature_length}."
                    f" {[(projector.__class__.__name__, projector.output_token_length, 'X',  projector.output_frame_length) for projector in self.projectors]}"  # noqa
                )
            else:
                assert all(
                    projector.output_token_length * backbone.temporal_resolution in [1, visual_feature_length]
                    for (projector, backbone) in zip(self.projectors, self.video_backbones)
                ), (
                    "Output token length is not consistent across all projectors!"
                    f" visual_feature_length={visual_feature_length}."
                    f" {[(backbone.__class__.__name__, projector.output_token_length * backbone.temporal_resolution) for (projector, backbone) in zip(self.projectors, self.video_backbones)]}"  # noqa
                )
        else:
            # ! This is just here to make our lives easier for bwd compatability.
            if self.tokens_resampled:
                correct_length = self.projectors[0].output_token_length * self.projectors[0].output_frame_length
            else:
                correct_length = self.video_backbones[0].num_patches
            if correct_length != visual_feature_length:
                overwatch.info(
                    f"Visual feature length {visual_feature_length} is not consistent "
                    f"with the output token length of the projector! Changing to {correct_length}",
                    ctx_level=1,
                )
                visual_feature_length = correct_length
        self.visual_feature_length = visual_feature_length

        # Initialize Specific Adapter

        if self.feature_fusion_type == "query_mlp":
            self.feature_fusion = MLPProjector(3072, len(video_backbones))
        elif self.feature_fusion_type == "cross_attention_avg_lq":
            self.feature_fusion = CrossAttentionAdapterLearnableQuery(
                embed_dim=3072, llm_dim=llm_backbone.embed_dim, token_length=visual_feature_length, averagetoken=True
            )
        elif self.feature_fusion_type == "concat_channel":
            self.feature_fusion = LinearProjector(len(video_backbones) * llm_backbone.embed_dim, llm_backbone.embed_dim)
        elif self.feature_fusion_type == "concat_channel_ln":
            self.feature_fusion = torch.nn.Sequential(
                torch.nn.LayerNorm(len(video_backbones) * llm_backbone.embed_dim),
                LinearProjector(len(video_backbones) * llm_backbone.embed_dim, llm_backbone.embed_dim),
            )
        elif self.feature_fusion_type == "scalar":
            self.feature_fusion = ScalarAdapter(len(video_backbones))
        else:
            self.feature_fusion = None

        overwatch.info(f"Feature Fusion: {self.feature_fusion}")

        # Trackers
        self.video_backbone_requires_grad = False

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["llm_backbone", "projectors", "video_backbone", "adapter"]
        self.trainable_module_keys = []

        # === Generation Utilities ===
        #   => For computing likelihoods --> get tokens corresponding to "True", "False" and "Yes", "No"
        self.string2idx = {}
        for trigger_string in ["True", "False", "Yes", "No"] + [chr(ord("A") + i) for i in range(26)]:
            token_idx_list = self.llm_backbone.tokenizer.encode(trigger_string, add_special_tokens=False)
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        video_backbones: List[VideoBackbone],
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        adapter: Optional[str] = None,
        visual_feature_length: Optional[int] = -1,
        projector_token_length: Optional[int] = -1,
    ) -> MERV:
        """Initialize a MERV from a pretrained checkpoint, freezing all weights, tailored for inference."""
        vidlm = cls(
            model_id,
            video_backbones,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            adapter=adapter,
            visual_feature_length=visual_feature_length,
            projector_token_length=projector_token_length,
        )

        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        if "projector" in model_state_dict:
            model_state_dict["projectors"] = {"0." + k: v for k, v in model_state_dict["projector"].items()}

        assert "projectors" in model_state_dict and "llm_backbone" in model_state_dict, (
            "MERV `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"
            + f'{("projectors" in model_state_dict, "llm_backbone" in model_state_dict)}'
        )

        vidlm.projectors.load_state_dict(model_state_dict["projectors"])
        vidlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])

        if vidlm.feature_fusion is not None:
            vidlm.feature_fusion.load_state_dict(model_state_dict["adapter"])
        else:
            assert "adapter" not in model_state_dict or len(model_state_dict["adapter"]) == 0, model_state_dict[
                "adapter"
            ]

        # Freeze Weights
        vidlm.requires_grad_(False)
        vidlm.eval()

        return vidlm

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        prompt_initializer: Type[PromptBuilder] = self.llm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

    def freeze_backbones(self, stage: str) -> None:
        """
        This function sets `requires_grad_` on each of the component modules explicitly, depending on stage.

        We support two separate stages --> "align" and "finetune".
            => "align" --> vision_backbone*, llm_backbone* are frozen; only the `projector` is trained.
            => "finetune" --> vision_backbone* is frozen; both `projector` and `llm_backbone` are trained.

        :param stage: Pretraining stage in < "align" | "finetune" | "full-finetune" >
        """
        if stage == "align":
            self.video_backbones.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            self.projectors.requires_grad_(True)
            if self.feature_fusion is not None:
                self.feature_fusion.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projectors", "adapter"]

            # Update Trackers
            self.video_backbone_requires_grad = False

            # Explicitly Log Frozen / Trainable Components
            overwatch.info(
                "[Frozen]    🥶 =>> Video Backbones"
                f" `{[video_backbone.identifier for video_backbone in self.video_backbones]}`",
                ctx_level=1,
            )
            overwatch.info(f"[Frozen]    🥶 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projectors `{self.arch_specifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Adapters `{self.feature_fusion_type}`", ctx_level=1)

        elif stage == "full-align":
            self.video_backbones.requires_grad_(False)
            self.llm_backbone.requires_grad_(True)
            self.projectors.requires_grad_(True)
            if self.feature_fusion is not None:
                self.feature_fusion.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projectors", "llm_backbone", "adapter"]

            # Update Trackers
            self.video_backbone_requires_grad = False

            # Explicitly Log Frozen / Trainable Components
            overwatch.info(
                "[Frozen]    🥶 =>> Video Backbones"
                f" `{[video_backbone.identifier for video_backbone in self.video_backbones]}`",
                ctx_level=1,
            )
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projectors `{self.arch_specifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Adapters `{self.feature_fusion_type}`", ctx_level=1)
        elif stage == "finetune" or stage == "second_finetune":
            self.video_backbones.requires_grad_(False)
            self.llm_backbone.requires_grad_(True)
            self.projectors.requires_grad_(True)
            if self.feature_fusion is not None:
                self.feature_fusion.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projectors", "llm_backbone", "adapter"]

            # Update Trackers
            self.video_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(
                "[Frozen]    🥶 =>> Video Backbones"
                f" `{[video_backbone.identifier for video_backbone in self.video_backbones]}`",
                ctx_level=1,
            )
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Adapters `{self.feature_fusion_type}`", ctx_level=1)

        elif stage == "full-finetune":
            raise NotImplementedError

        else:
            raise ValueError(f"Stage `{stage}` is not supported for MERV! Try < align | finetune >")

    def load_from_checkpoint(self, stage: str, run_dir: Path, pretrained_checkpoint: Optional[Path] = None) -> None:
        """Load weights from checkpoint (if required by the given stage)."""
        assert stage in {
            "align",
            "full-align",
            "finetune",
            "full-finetune",
            "second_finetune",
        }, f"Stage {stage} is not supported!"

        # If we're running a `no-align` architecture, we're good!
        if self.arch_specifier.startswith("no-align"):
            overwatch.info(f"MERV with `{self.arch_specifier = }` does not require pretrained weights!", ctx_level=1)
            return

        # Otherwise, handle stage-specific logic!
        if stage == "align" or stage == "full-align":
            overwatch.info(
                "Stage `align` or `full-align` does not require pretrained weights =>> Starting Training", ctx_level=1
            )
            return

        # Otherwise, load from `pretrained_checkpoint` or match on `run_dir` (s/+stage-finetune/+stage-align/g)
        overwatch.info("Stage `finetune` requires `align` pretrained weights", ctx_level=1)

        if stage == "second_finetune" and pretrained_checkpoint is not None:
            overwatch.info(
                f"Loading from Provided Checkpoint `{pretrained_checkpoint}` for second finetuning!", ctx_level=1
            )
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            if "projector" in model_state_dict:
                model_state_dict["projectors"] = {"0." + k: v for k, v in model_state_dict["projector"].items()}

            assert "projectors" in model_state_dict and "llm_backbone" in model_state_dict, (
                "MERV `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"
                + f'{("projectors" in model_state_dict, "llm_backbone" in model_state_dict)}'
            )

            self.projectors.load_state_dict(model_state_dict["projectors"])
            self.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])

            if self.feature_fusion is not None:
                self.feature_fusion.load_state_dict(model_state_dict["adapter"])
            else:
                assert "adapter" not in model_state_dict or len(model_state_dict["adapter"]) == 0, model_state_dict[
                    "adapter"
                ]

            return

        # Config specifies path to a checkpoint to load
        if pretrained_checkpoint is not None:
            overwatch.info(f"Loading from Provided Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            self.projectors.load_state_dict(model_state_dict["projectors"])

            return

        # [Contract] If no `pretrained_checkpoint`, assume `align` lives in the run directory; string substitution!
        model, scale, _, seed = run_dir.name.split("+")
        align_dirs = [
            d
            for d in run_dir.parent.iterdir()
            if (d.name.startswith(f"{model}+{scale}") and d.name.endswith(f"+stage-align+{seed}"))
        ]
        assert len(align_dirs) == 1, "Multiple or No Valid Pretrained Directories Exist -- Double Check `runs`!"
        if (pretrained_checkpoint := (align_dirs[0] / "checkpoints" / "latest-checkpoint.pt")).exists():
            overwatch.info(f"Loading from Discovered Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            self.projectors.load_state_dict(model_state_dict["projectors"])
        else:
            raise ValueError(f"Could not find valid `align` checkpoint at {pretrained_checkpoint}!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VidLM policy)."""
        video_fsdp_wrapping_policies = [
            video_backbone.get_fsdp_wrapping_policy() for video_backbone in self.video_backbones
        ]
        llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy()

        # Get VideoLM Wrapping Policy =>> just a module wrapping policy around `self.projector`
        videolm_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={
                LinearProjector,
                MLPProjector,
                FusedMLPProjector,
                AveragePoolingProjector,
                ConvolutionalProjector,
                MLPDeepProjector,
                AveragePooling3DProjector,
                Convolutional3DProjector,
            },
        )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VidLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                *video_fsdp_wrapping_policies,
                llm_fsdp_wrapping_policy,
                videolm_fsdp_wrapping_policy,
            ],
        )

    # Note =>> We're not explicitly subclassing `PreTrainedModel` because we don't need the bloat; however, `forward()`
    #          *must* match the signature of a `{Model}ForCausalLM` so that we can inherit from `GenerationMixin`

    # ruff: noqa: C901
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        video_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        is_image: Optional[torch.BoolTensor] = None,  # True if that item is image modality, false if video or text-only
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the VidLM, returning a CausalLMOutputWithPast instance (contains loss)."""

        # Handle Inference (leverage cache, short-circuit on just LLM forward)
        if input_ids.shape[1] == 1 and past_key_values is not None:
            # We're leveraging the cache, so just redirect to `self.llm_backbone` with `input_ids` and `past_key_values`
            output = self.llm_backbone(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            return output

        elif input_ids.shape[1] == 1 or (video_values is None):
            raise RuntimeError("Invalid `forward()` call!")

        # Handle Multimodal Indices is None --> pretend like the batch is fully multimodal (always image + text)!
        if multimodal_indices is None:
            multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)

        # Handle Multimodal Indices is Empty (len == 0) --> simple unimodal forward
        elif len(multimodal_indices) == 0:
            return self.llm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        with torch.set_grad_enabled(self.video_backbone_requires_grad):
            video_features = [
                self.video_backbones[encoder_idx](video_values[encoder_idx], is_image)
                for encoder_idx in range(len(self.video_backbones))
            ]

        # Recombine based on the indices
        patch_features = video_features

        # Do some shape-matching
        patch_features = [patch_feature[multimodal_indices] for patch_feature in patch_features]

        # Simplified now, move reshaping into each projector
        # Input: [B, (F N), C] -> [B, N', C']
        if self.tokens_resampled:
            patch_features = [
                patch_features[i].reshape(
                    -1,
                    self.video_backbones[i].temporal_resolution,
                    self.video_backbones[i].spatial_resolution,
                    patch_features[i].shape[-1],
                )
                for i in range(len(patch_features))
            ]

        projected_patch_embeddings = [
            projector(patch_feature) for projector, patch_feature in zip(self.projectors, patch_features)
        ]
        if self.feature_fusion_type == "cross_attention_classKV":
            classembs = [
                projector.projector(patch_feature) for projector, patch_feature in zip(self.projectors, classembs)
            ]

        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)

        mixer_value = None

        if self.feature_fusion_type == "first":
            projected_patch_embeddings = projected_patch_embeddings[0].unsqueeze(1)
        elif self.feature_fusion_type == "concat":
            projected_patch_embeddings = torch.concat(projected_patch_embeddings, 1).unsqueeze(1)
        elif self.feature_fusion_type == "concat_channel" or self.feature_fusion_type == "concat_channel_ln":
            # Concat along channel, then add adapter afterwards to reshape into LLM dim
            projected_patch_embeddings = torch.concat(projected_patch_embeddings, -1)
            projected_patch_embeddings = self.feature_fusion(projected_patch_embeddings).unsqueeze(1)
        elif "cross_attention" in self.feature_fusion_type:
            projected_patch_embeddings, mixer_value = self.feature_fusion(projected_patch_embeddings)
            projected_patch_embeddings = projected_patch_embeddings.unsqueeze(1)
        else:
            print(f'Adapter "{self.feature_fusion_type}" doesn\'t exist')
            raise NotImplementedError

        #################################################################################

        assert (
            len(projected_patch_embeddings.shape) == 4
        ), "projected_patch_embeddings should have [B, L, token, channel], L being number of turns. {}".format(
            projected_patch_embeddings.shape
        )

        projected_patch_embeddings = projected_patch_embeddings[:, 0]
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),  # batch x token
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # Build Multimodal Embeddings (and build resulting attention mask)
        multimodal_embeddings = torch.cat(
            [
                input_embeddings[multimodal_indices, :1, :],
                projected_patch_embeddings,
                input_embeddings[multimodal_indices, 1:, :],
            ],
            dim=1,
        )
        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [
                    attention_mask[multimodal_indices, :1],
                    projected_patch_attention_mask,
                    attention_mask[multimodal_indices, 1:],
                ],
                dim=1,
            )

        # [Contract] We assume the first token of `labels` (associated with <BOS>) is already marked as "IGNORE"
        #   => We'll ignore the per-token outputs for each of the patch embeddings as well!
        multimodal_labels = None
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            multimodal_labels = torch.cat(
                [labels[multimodal_indices, :1], projected_patch_labels, labels[multimodal_indices, 1:]], dim=1
            )

        # === Add Unimodal Handling ===

        # Create Fused Embeddings, Attention Mask, and Labels by Merging with "unimodal" Inputs (if applicable)
        unimodal_indices = torch.tensor(
            [idx for idx in range(len(input_ids)) if idx not in multimodal_indices],
            dtype=torch.long,
            device=multimodal_indices.device,
        )

        # No "unimodal" data --> Fused == Multimodal
        if len(unimodal_indices) == 0:
            fused_embeddings = multimodal_embeddings
            fused_attention_mask = multimodal_attention_mask
            fused_labels = multimodal_labels

        else:
            # Otherwise --> Merge w/ unimodal data

            # This doesn't matter --> but in the "normal" case this is the embedding of the <PAD> token
            #   => NOTE :: Verified that `zeros/randn/empty/<PAD> embedding` all return the same result!
            unimodal_embeddings_pad = torch.zeros(
                (len(unimodal_indices), projected_patch_embeddings.shape[1], input_embeddings.shape[2]),
                dtype=input_embeddings.dtype,
                device=input_embeddings.device,
            )
            unimodal_attention_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                False,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            unimodal_labels_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )

            padcount = (multimodal_embeddings.shape[1] - input_embeddings.shape[1]) // projected_patch_embeddings.shape[
                1
            ]

            unimodal_embeddings = torch.cat(
                [input_embeddings[unimodal_indices]] + [unimodal_embeddings_pad] * padcount, dim=1
            )
            unimodal_attention_mask = torch.cat(
                [attention_mask[unimodal_indices]] + [unimodal_attention_pad] * padcount, dim=1
            )
            unimodal_labels = torch.cat([labels[unimodal_indices]] + [unimodal_labels_pad] * padcount, dim=1)

            # Create "Fused" Tensors by Stacking Multimodal & Unimodal
            fused_embeddings = torch.vstack([multimodal_embeddings, unimodal_embeddings])
            fused_attention_mask = torch.vstack([multimodal_attention_mask, unimodal_attention_mask])
            fused_labels = torch.vstack([multimodal_labels, unimodal_labels])

        # Run LLM Forward --> returns CausalLMOutputWithPast!

        return self.llm_backbone(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=fused_embeddings,
            labels=fused_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # === GenerationMixin Methods ===
    #   => Note: The following methods override the functionality of `transformers.GenerationMixin`; these expect the
    #            contract in each of the function signatures, and also expect our `forward` function to roughly take
    #            the same arguments as the underlying LLM (see `LlamaModelForCausalLM` as an example)

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        video_values: Optional[torch.FloatTensor] = None,
        text_emb: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        is_image: Optional[torch.BoolTensor] = None,
        use_cache: Optional[bool] = None,
        input_loc: Optional[List] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
        **kwargs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` --> in general, just handles caching logic during generation."""
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "video_values": video_values,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "is_image": is_image,
                "multimodal_indices": multimodal_indices,
            }
        )

        return model_inputs

    @torch.inference_mode()
    def generate(self, video: str, prompt_text: str, num_frames: List[int], **kwargs) -> str:
        assert len(num_frames) == len(self.video_backbones), "Number of frames should match number of video backbones!"
        # For now, only support generation with a batch size of 1 for simplicity
        tokenizer = self.llm_backbone.tokenizer
        video_transforms = [video_backbone.video_transform for video_backbone in self.video_backbones]

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        if video is not None and ".jpg" in video:
            image = Image.open(video).convert("RGB")
            video = torch.from_numpy(np.array(image).transpose(2, 0, 1)[None,].repeat(max(num_frames), 0))
            video_values = [
                video_transform(video[:: max(num_frames) // num_frame]).unsqueeze(0).to(self.device)
                for video_transform, num_frame in zip(video_transforms, num_frames)
            ]

        elif video is not None:
            clip_start_sec = kwargs.pop("clip_start_sec", 0.0)
            clip_end_sec = kwargs.pop("clip_end_sec", None)

            video = load_video(
                video, clip_start_sec=clip_start_sec, clip_end_sec=clip_end_sec, num_frames=max(num_frames)
            )
            video_values = [
                video_transform(video[:: max(num_frames) // num_frame]).unsqueeze(0).to(self.device)
                for video_transform, num_frame in zip(video_transforms, num_frames)
            ]
        else:
            video_values = [
                torch.zeros(video_backbone.default_video_resolution).unsqueeze(0).to(self.device)
                for video_backbone in self.video_backbones
            ]

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype

        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super().generate(
                input_ids=input_ids,            # Shape: [1, seq]
                video_values=video_values,      # Shape: [1, 3, res, res] or Dict[str, Shape[1, 3, res, res]]
                is_image = torch.tensor([False]).unsqueeze(0).to(self.device),
                attention_mask = torch.ones_like(input_ids),
                pad_token_id=self.llm_backbone.tokenizer.pad_token_id,
                **kwargs
            )
            # fmt: on

        generated_text = tokenizer.decode(generated_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()

        return generated_text

    def _supports_default_dynamic_cache(self):
        # some new HF stuff.
        return False
