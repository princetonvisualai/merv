"""
fsdp.py

Core class definition for a strategy implementing Torch native Fully Sharded Data Parallel Training (with support for
fine-grained control over wrapping policies and mixed precision per component).
"""

import re
import math
import shutil
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from merv.models.vidlms import MERV
from merv.overwatch import initialize_overwatch
from merv.training.strategies.base_strategy import TrainingStrategy

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class FSDPStrategy(TrainingStrategy):
    def __init__(
        self,
        vidlm: MERV,
        device_id: int,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        save_checkpoint_after: int = 12,
        resume_from_checkpoint: Optional[str] = None,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        sharding_strategy: str = "shard-grad-op",
        state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT,
    ) -> None:
        super().__init__(
            vidlm=vidlm,
            device_id=device_id,
            epochs=epochs,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            save_checkpoint_after=save_checkpoint_after,
            resume_from_checkpoint=resume_from_checkpoint,
            enable_mixed_precision_training=enable_mixed_precision_training,
            reduce_in_full_precision=reduce_in_full_precision,
            mixed_precision_dtype=mixed_precision_dtype,
            worker_init_fn=worker_init_fn,
        )

        # FSDP-Specific Parameters
        if sharding_strategy == "shard-grad-op":
            self.fsdp_sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        elif sharding_strategy == "full-shard":
            self.fsdp_sharding_strategy = ShardingStrategy.FULL_SHARD
        else:
            raise ValueError(f"FSDP Sharding Strategy {sharding_strategy} is not supported!")

        assert state_dict_type == StateDictType.FULL_STATE_DICT, "Sharded state saving is not yet implemented!"
        self.fsdp_state_dict_type = state_dict_type
        self.fsdp_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """Save a checkpoint to the `run_dir` only containing the state_dicts for trainable parameters by default."""
        assert isinstance(self.vidlm, FSDP), "FSDPStrategy.save_checkpoint assumes VidLM is already wrapped in FSDP!"

        # Summon Full State Dictionary =>> Reconstitute from Shards
        with FSDP.state_dict_type(self.vidlm, self.fsdp_state_dict_type, self.fsdp_save_policy):
            full_vidlm_state_dict = self.vidlm.state_dict()
            model_state_dicts = {
                mkey: OrderedDict() for mkey in (self.trainable_module_keys if only_trainable else self.all_module_keys)
            }

            # Iterate through `full_vidlm_state_dict` and split `mkey.{full_dotted_path}` -> `mkey: {full_dotted_path}`
            for key, param in full_vidlm_state_dict.items():
                for mkey in model_state_dicts:
                    if key.startswith(mprefix := f"{mkey}."):
                        model_state_dicts[mkey][key.removeprefix(mprefix)] = param

            # Save on rank zero *only*
            if overwatch.is_rank_zero():
                checkpoint_dir = run_dir / "checkpoints"
                if train_loss is None:
                    checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
                else:
                    checkpoint_path = (
                        checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"
                    )

                # Save Checkpoint & Copy Latest to `latest-checkpoint.pt`
                torch.save({"model": model_state_dicts}, checkpoint_path)
                shutil.copy(checkpoint_path, checkpoint_dir / "latest-checkpoint.pt")
    
    def save_intermediate_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
    ) -> None:
        """Saves a full training checkpoint (model, optimizer, scheduler, step, epoch) to the `run_dir`."""
        assert isinstance(self.vidlm, FSDP), "FSDPStrategy.save_checkpoint assumes VidLM is already wrapped in FSDP!"

        # Summon Full State Dictionary for model and optimizer
        with FSDP.state_dict_type(self.vidlm, self.fsdp_state_dict_type, self.fsdp_save_policy):
            model_state_dict = self.vidlm.state_dict()
            optimizer_state_dict = self.optimizer.state_dict()

            # Save on rank zero *only*
            if overwatch.is_rank_zero():
                checkpoint_dir = run_dir / "checkpoints"
                checkpoint_path = checkpoint_dir / f"step-{global_step:06d}.pt"

                # Consolidate all state into a single dictionary
                full_checkpoint = {
                    "model": model_state_dict,
                    "optimizer": optimizer_state_dict,
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "train_loss": train_loss
                }
                
                # Save Checkpoint & create/update a 'latest.pt' symlink for easy access
                torch.save(full_checkpoint, checkpoint_path)
                # symlink_path = checkpoint_dir / "latest.pt"
                # if os.path.exists(symlink_path):
                #     os.remove(symlink_path)
                # os.symlink(checkpoint_path, symlink_path)
                overwatch.info(f"Saved full FSDP checkpoint to `{checkpoint_path}`")
    
    def load_checkpoint(
        self,
        checkpoint_path: Path,
        only_trainable: bool = True,
    ) -> Tuple[int, int]:
        """
        Load a checkpoint written by `save_checkpoint`, restore the FSDP-wrapped
        VidLM model’s parameters via the FSDP API, and return (global_step, epoch).
        """
        # 1) Load the saved file on CPU
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model_state_dicts: Dict[str, Dict[str, torch.Tensor]] = ckpt["model"]

        # 2) Rebuild a flat state_dict with keys like "mkey.param_name"
        full_sd: Dict[str, torch.Tensor] = {}
        keys = self.trainable_module_keys if only_trainable else self.all_module_keys
        for mkey in keys:
            sub = model_state_dicts.get(mkey, {})
            for name, tensor in sub.items():
                full_sd[f"{mkey}.{name}"] = tensor

        # 3) Sanity check
        assert isinstance(self.vidlm, FSDP), "VidLM must be wrapped in FSDP"

        # 4) Enter the same state_dict_type context you used for saving
        with FSDP.state_dict_type(
            self.vidlm,
            self.fsdp_state_dict_type,
            self.fsdp_save_policy,
        ):
            # 5) Call load_state_dict on the FSDP wrapper itself.
            #    The wrapper will reshape / shard / unflatten as needed.
            load_result = self.vidlm.load_state_dict(full_sd, strict=False)

            # Optional: log any missing / unexpected to see what didn’t map
            if load_result.missing_keys:
                overwatch.info(f"Missing keys: {load_result.missing_keys}", ctx_level=1)
            if load_result.unexpected_keys:
                overwatch.info(f"Unexpected keys: {load_result.unexpected_keys}", ctx_level=1)

        # 6) Parse global_step & epoch from filename
        stem = checkpoint_path.stem
        m = re.match(r"step-(\d+)-epoch-(\d+)-loss=.*", stem)
        if m:
            global_step, epoch = int(m.group(1)), int(m.group(2))
        else:
            global_step, epoch = 0, 0

        # 7) Update your trainer state
        self.global_step = global_step
        self.start_epoch = epoch + 1

        return global_step, epoch
        
    # def load_checkpoint(self, checkpoint_path: Path):
    #     """
    #     Loads a full training checkpoint (model, optimizer, scheduler, step, epoch).
    #     This method must be called AFTER the model is wrapped in FSDP and the optimizer is created.
    #     """
    #     assert isinstance(self.vidlm, FSDP), "VidLM must be wrapped in FSDP before loading a checkpoint."
        
    #     # Load the checkpoint dictionary on CPU to avoid CUDA OOM on any single rank
    #     full_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
    #     # Step 1: Load the model state dict. FSDP handles scattering the full state to all ranks.
    #     self.vidlm.load_state_dict(full_checkpoint["model"])
        
    #     # Step 2: Load the optimizer state dict. FSDP requires a special procedure for this.
    #     # We must first convert the flat state dict to one that FSDP can digest.
    #     optimizer_state_dict = FSDP.optim_state_dict_to_load(
    #         self.vidlm, self.optimizer, full_checkpoint["optimizer"]
    #     )
    #     self.optimizer.load_state_dict(optimizer_state_dict)
        
    #     # Step 3: Load the LR scheduler state
    #     self.lr_scheduler.load_state_dict(full_checkpoint["lr_scheduler"])
        
    #     # Step 4: Return the step and epoch to resume the training loop
    #     start_step = full_checkpoint["global_step"]
    #     start_epoch = full_checkpoint["epoch"]
        
    #     overwatch.info(f"Successfully loaded checkpoint from `{checkpoint_path}`. "
    #                     f"Resuming from Epoch {start_epoch}, Global Step {start_step}.")
        
    #     return start_step, start_epoch

    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        # Iteratively Assemble FSDP Wrapping Policy by fetching the wrapping policies for each backbone/constituent
        vidlm_fsdp_wrapping_policy = self.vidlm.get_fsdp_wrapping_policy()

        # Assemble the Default FSDP Mixed Precision Policy
        if self.enable_mixed_precision_training and self.mixed_precision_dtype == torch.bfloat16:
            # MixedPrecision `param_dtype` specifies *compute* dtype (for forward/backward only)
            #   => Reference: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision
            reduce_buffer_dtype = torch.bfloat16 if not self.reduce_in_full_precision else torch.float32
            fsdp_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16, reduce_dtype=reduce_buffer_dtype, buffer_dtype=reduce_buffer_dtype
            )

            # When running FSDP with a frozen vision backbone --> move to half precision!
            overwatch.info("Casting Video Backbone to *Half Precision* via `.to(dtype=...)`")
            for video_backbone in self.vidlm.video_backbones:
                video_backbone.to(dtype=video_backbone.half_precision_dtype)

        else:
            # If we're not using mixed precision, everything is in default full precision!
            fsdp_precision_policy = MixedPrecision(
                param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32
            )

        # <FSDP> => note that FSDP will automatically take care of device placement (similar to `autocast`)
        self.vidlm = FSDP(
            self.vidlm,
            auto_wrap_policy=vidlm_fsdp_wrapping_policy,
            mixed_precision=fsdp_precision_policy,
            sharding_strategy=self.fsdp_sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=True,
        )

        # Gradient Checkpoint Setup
        if self.enable_gradient_checkpointing:
            # For Gradient Checkpointing under FSDP --> we make the same assumption as in the DDP/other strategies; the
            #   bulk of activation memory is taken up by the LLM activations. However, unlike other strategies, we
            #   cannot rely on the HF Transformers default `gradient_checkpointing_enable()` --> FSDP breaks semantics!
            #
            # Instead, we need to write our own *NO-REENTRANT* wrapper, and apply it to the LLM's Transformer Layer.
            non_reentrant_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

            def check_fn(submodule: nn.Module) -> bool:
                return isinstance(submodule, self.llm_transformer_layer_cls)

            # Note that the terms "activation checkpointing" and "gradient checkpointing" are synonymous!
            apply_activation_checkpointing(self.vidlm, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)

        # Barrier =>> Sharding takes a minute?
        dist.barrier()

        # Create Optimizer and LR Scheduler =>> note that most of the LR Schedulers we use require `max_steps/epochs`
        #   => Optimizer should only operate on parameters that are *unfrozen* / trainable!
        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            n_train_examples = math.ceil(n_train_examples / self.global_batch_size) * self.global_batch_size
            if self.max_steps is None:
                num_training_steps = (n_train_examples * self.epochs) // self.global_batch_size
            else:
                num_training_steps = self.max_steps

            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            # Default AdamW w/ specified LR & Linear Warmup / Cosine Decay & Weight Decay
            #   => Create Parameter Groups --> bias terms, normalization layer parameters shouldn't be decayed!
            decay, no_decay = [], []
            for name, param in self.vidlm.named_parameters():
                if not param.requires_grad:
                    continue

                # Check on any parameters with fewer than 2 dimensions or with "bias" in the name
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)

            # Build Parameter Groups
            groups = [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

            # Create Optimizer & LR Scheduler
            self.optimizer = AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0

        else:
            raise ValueError(f"Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!")

        # Finalize Setup =>> Log!
        overwatch.info(
            "FSDP Full-Shard Strategy =>> Finalized Training Setup:\n"
            f"         |-> Global (Effective) Batch Size = {self.global_batch_size}\n"
            f"         |-> Per-Device Batch Size = {self.per_device_batch_size}\n"
            f"         |-> Distributed World Size = {overwatch.world_size()}\n"
            f"         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n"
            f"         |-> LLM Backbone FSDP Gradient Checkpointing = {self.enable_gradient_checkpointing}\n"
            f"         |-> Use FSDP Mixed Precision = {self.enable_mixed_precision_training}\n"
            f"                 |-> Parameter Precision = {fsdp_precision_policy.param_dtype}\n"
            f"                 |-> Reduction Precision = {fsdp_precision_policy.reduce_dtype}\n"
            f"                 |-> Buffer Precision = {fsdp_precision_policy.buffer_dtype}\n\n"
            f"         |-> Default AdamW LR = {self.learning_rate}\n"
            f"         |-> AdamW Weight Decay = {self.weight_decay}\n"
            f"         |-> LR Scheduler Type = {self.lr_scheduler_type}\n"
            f"         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n"
            f"         |-> Dataset Size = {n_train_examples} Examples\n"
            f"         |-> Max Steps = {num_training_steps}\n"
        )
        
        if self.resume_from_checkpoint is not None:
            overwatch.info("[MERV] Found a checkpoint to load from - This can be an intermediate one.")
            self.start_step, self.start_epoch = self.load_checkpoint(self.resume_from_checkpoint)
        else:
            overwatch.info("[MERV] No intermediate checkpoint found. Starting training from scratch.")

    def clip_grad_norm(self) -> None:
        # Note =>> FSDP uses a custom `clip_grad_norm_` function; requires *uniform grad dtype*
        self.vidlm.clip_grad_norm_(max_norm=self.max_grad_norm)
