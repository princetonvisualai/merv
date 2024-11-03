"""
models.py

Draccus Dataclass Definition for a ModelConfig object, with various registered subclasses for each model family and
variant thereof. A given model variant configures the following attributes:
    - Pretrained Visual Representation(s) (e.g., OpenAI CLIP ViT-L/14) + Pretrained LLM Backbone (e.g., LLaMa-2 7B)
    - VidLM Configuration + Parameters (e.g., MLP Projector, Image Preprocessing, etc.)
    - [Optional] Stage 1 (`align`) Optimization Hyperparameters
    - Stage 2 (`finetune`) Optimization Hyperparameters
"""

# ruff: noqa: E501

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import List, Optional, Union

from draccus import ChoiceRegistry

from merv.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


@dataclass
class ModelConfig(ChoiceRegistry):
    # fmt: off
    model_id: str                                           # Unique Model ID that fully specifies a given variant
    arch_specifier: str                                     # Architecture specifier string (e.g., "gelu-mlp")
                                                            # Can be {,no-align} + {avg,attntv,conv,3davg,3dconv} + {linear,mlp}
    feature_fusion: str                                     # Feature Fusion Strategy (e.g., "learnable query, etc.")
                                                            # Can be {query_mlp}, {cross_attention_avg_lq}, {concat_channel}, etc.

    # Pretrained Backbones
    video_backbone_ids: List[str]                           # Pretrained Video Backbone to load
    llm_backbone_id: str                                    # Pretrained LLM (from HF Transformers) to load

    # Backbone Parameters
    image_resize_strategy: str                              # Resizing strategy in < crop | letterbox | corner-pad >
    llm_max_length: int                                     # Maximum context length for LLM (can be < than max!)
    num_frames: Union[int, List[int]]                       # Number of frames to sample in each video backbone
    projector_token_length: int                             # number of tokens to project to; must be square!
    visual_feature_length: int                              # Length of visual features after projection; must be equal!

    # === Multi-Stage Optimization Hyperparameters ===
    # By default, we assume an AdamW optimizer with FSDP (Gradient Sharding or Full Sharding depending on stage)

    # Align Stage Optimization Parameters
    align_epochs: int                                       # Epochs to Run (in case `max_steps` is not specified)
    align_max_steps: Optional[int]                          # [Optional] Max Gradient Steps (overrides epochs)
    align_global_batch_size: int                            # Global Batch Size (divided across processes)
    align_per_device_batch_size: int                        # Per-Device Batch Size (per-process)
                                                            #   => # of accumulation steps is auto-computed

    align_learning_rate: float                              # Peak Learning Rate (lr_scheduler sets warmup/decay)
    align_weight_decay: float                               # Weight Decay for AdamW Optimizer
    align_max_grad_norm: float                              # Max Grad Norm (for global gradient clipping)
    align_lr_scheduler_type: str                            # LR Scheduler (default: "linear-warmup+cosine-decay")
    align_warmup_ratio: float                               # Fraction of total steps to warmup

    align_train_strategy: str                               # Align Train Strategy (default: "fsdp-shard-grad-op")

    # Finetune Stage Optimization Parameters
    finetune_epochs: int                                    # Epochs to Run (in case `max_steps` is not specified)
    finetune_max_steps: Optional[int]                       # [Optional] Max Gradient Steps (overrides epochs)
    finetune_global_batch_size: int                         # Global Batch Size (divided across processes)
    finetune_per_device_batch_size: int                     # Per-Device Batch Size (per-process)
                                                            #   => # of accumulation steps is auto-computed

    finetune_learning_rate: float                           # Peak Learning Rate (lr_scheduler sets warmup/decay)
    finetune_weight_decay: float                            # Weight Decay for AdamW Optimizer
    finetune_max_grad_norm: float                           # Max Grad Norm (for global gradient clipping)
    finetune_lr_scheduler_type: str                         # LR Scheduler (default: "linear-warmup+cosine-decay")
    finetune_warmup_ratio: float                            # Fraction of total steps to warmup

    finetune_train_strategy: str                            # Finetune Train Strategy (default: "fsdp-full-shard")

    # Enable Gradient/Activation Checkpointing (for the LLM Backbone)
    enable_gradient_checkpointing: bool = True

    # Enable Traditional Mixed Precision Training via Torch Native AMP (`autocast`)
    enable_mixed_precision_training: bool = True            # Whether to enable mixed precision training
    reduce_in_full_precision: bool = False                  # Whether to run gradient reduction in FP32

    # fmt: on

    # Help with some version compatibility b/w breaking changes
    #   Easier than updating every config
    def __post_init__(self) -> None:
        if isinstance(self.num_frames, int):
            overwatch.info(f"Inflating num_frames {self.num_frames} from int to list...")
            # Naively inflate to match
            self.num_frames = [self.num_frames for _ in range(len(self.video_backbone_ids))]


# === LLaVa v1.5 Reproduction - Fully Specified Configurations ===
@dataclass
class MERV_Base(ModelConfig):
    model_id: str = "merv-base"
    arch_specifier: str = "no-align+3davg+linear"
    feature_fusion: str = "cross_attention_avg_lq"

    video_backbone_ids: List[str] = field(
        default_factory=lambda: [
            "languagebind-video-noclass",
            "dinov2-video-all-tokens",
            "vivit-google-b-all-no-cls-16frames",
            "siglip-vit-b16-224px-all-no-cls",
        ]
    )
    llm_backbone_id: str = "llama2-7b-pure"
    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 2048
    num_frames: Union[int, List[int]] = field(default_factory=lambda: [16, 16, 32, 16])
    projector_token_length: int = 64
    visual_feature_length: int = 1024

    # Align Stage Optimization Parameters
    align_epochs: int = 1
    align_max_steps: Optional[int] = None
    align_global_batch_size: int = 256
    align_per_device_batch_size: int = 16

    align_learning_rate: float = 1e-3
    align_weight_decay: float = 0.0
    align_max_grad_norm: float = 1.0
    align_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    align_warmup_ratio: float = 0.03

    align_train_strategy: str = "fsdp-shard-grad-op"

    # Finetune Stage Optimization Parameters
    finetune_epochs: int = 1
    finetune_max_steps: Optional[int] = None
    finetune_global_batch_size: int = 128
    finetune_per_device_batch_size: int = 8

    finetune_learning_rate: float = 2e-5
    finetune_weight_decay: float = 0.1
    finetune_max_grad_norm: float = 1.0
    finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    finetune_warmup_ratio: float = 0.03

    finetune_train_strategy: str = "fsdp-full-shard"


@dataclass
class MERV_Full(MERV_Base):
    model_id: str = "merv-full"
    # No align since there's stage 1 and 2, only difference
    arch_specifier: str = "3davg+linear"
    # Full shard as well for LLM training
    align_train_strategy: str = "fsdp-full-shard"
    align_learning_rate: float = 1e-4


# Single Encoder Models
@dataclass
class LanguageBind_Single_Encoder(MERV_Base):
    model_id: str = "languagebind-single"
    video_backbone_ids: List[str] = field(default_factory=lambda: ["languagebind-video-noclass"])
    num_frames: Union[int, List[int]] = field(default_factory=lambda: [16])


@dataclass
class DINOv2_Single_Encoder(MERV_Base):
    model_id: str = "dinov2-single"
    video_backbone_ids: List[str] = field(default_factory=lambda: ["dinov2-video-all-tokens"])
    num_frames: Union[int, List[int]] = field(default_factory=lambda: [16])


@dataclass
class ViViT_Single_Encoder(MERV_Base):
    model_id: str = "vivit-single"
    video_backbone_ids: List[str] = field(default_factory=lambda: ["vivit-google-b-all-no-cls-16frames"])
    num_frames: Union[int, List[int]] = field(default_factory=lambda: [32])


@dataclass
class SigLIP_Single_Encoder(MERV_Base):
    model_id: str = "siglip-single"
    video_backbone_ids: List[str] = field(default_factory=lambda: ["siglip-vit-b16-224px-all-no-cls"])
    num_frames: Union[int, List[int]] = field(default_factory=lambda: [16])


@dataclass
class LLaVa_v15_Reproduction_13B(MERV_Base):
    model_id: str = "reproduction-llava-v15+13b"
    llm_backbone_id: str = "vicuna-v15-13b"


# === Define a Model Registry Enum for Reference & Validation ===
@unique
class ModelRegistry(Enum):
    # === MERV Base ===
    MERV_BASE = MERV_Base
    MERV_FULL = MERV_Full

    # === Single Encoder Models ===
    LANGUAGEBIND_SINGLE = LanguageBind_Single_Encoder
    DINOV2_SINGLE = DINOv2_Single_Encoder
    VIVIT_SINGLE = ViViT_Single_Encoder
    SIGLIP_SINGLE = SigLIP_Single_Encoder

    @property
    def model_id(self) -> str:
        return self.value.model_id


# Register Models in Choice Registry
for model_variant in ModelRegistry:
    ModelConfig.register_subclass(model_variant.model_id, model_variant.value)


if __name__ == "__main__":
    m = ModelRegistry
