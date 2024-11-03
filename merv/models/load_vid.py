"""
load.py

Entry point for loading pretrained VidLMs for inference; exposes functions for listing available models (with canonical
IDs, mappings to paper experiments, and short descriptions), as well as for loading models (from disk or HF Hub).
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import hf_hub_download

from merv.conf import ModelConfig, ModelRegistry
from merv.models.materialize import get_llm_backbone_and_tokenizer, get_video_backbone_and_transform
from merv.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from merv.models.vidlms import MERV
from merv.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === HF Hub Repository ===
HF_HUB_REPO = "tyleryzhu/merv"


# === Available Models ===
def available_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def available_model_names() -> List[str]:
    return list(GLOBAL_REGISTRY.items())


def get_model_description(model_id_or_name: str) -> str:
    if model_id_or_name not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `{model_id_or_name = }; check `merv.available_model_names()`")

    # Print Description & Return
    print(json.dumps(description := GLOBAL_REGISTRY[model_id_or_name]["description"], indent=2))

    return description


# === Load Pretrained Model ===
def load_vid(
    model_id_or_path: Union[str, Path], hf_token: Optional[str] = None, cache_dir: Optional[Union[str, Path]] = None
) -> MERV:
    """Loads a pretrained MERV from either local disk."""
    if os.path.isdir(model_id_or_path):
        overwatch.info(f"Loading from local path `{(run_dir := Path(model_id_or_path))}`")

        # Get paths for `config.json` and pretrained checkpoint
        config_json = run_dir / "config.json"
        checkpoint_pt = run_dir / "checkpoints" / "latest-checkpoint.pt"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert checkpoint_pt.exists(), "Missing checkpoint!"
    else:
        if model_id_or_path not in GLOBAL_REGISTRY:
            raise ValueError(f"Couldn't find `{model_id_or_path = }; check `merv.available_model_names()`")

        overwatch.info(f"Downloading `{(model_id := GLOBAL_REGISTRY[model_id_or_path]['model_id'])} from HF Hub")
        config_json = hf_hub_download(repo_id=HF_HUB_REPO, filename=f"{model_id}/config.json", cache_dir=cache_dir)
        checkpoint_pt = hf_hub_download(
            repo_id=HF_HUB_REPO, filename=f"{model_id}/checkpoints/latest-checkpoint.pt", cache_dir=cache_dir
        )
        checkpoint_pt = Path(checkpoint_pt)

    # Load Model Config from `config.json`
    with open(config_json, "r") as f:
        model_cfg = json.load(f)["model"]
        model_cfg.pop("vidlm_id", None)
        # wasn't even getting used before anyways shrug
        model_cfg.pop("type", None)
        # Doing this so we can get post_init bwd compatbility
        model_cfg = ModelConfig.get_choice_class(ModelRegistry.MERV_BASE.model_id)(**model_cfg)

    # = Load Individual Components necessary for Instantiating a VidLM =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg.model_id}[/] with:\n"
        f"             Video Backbone =>> [bold]{model_cfg.video_backbone_ids}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg.llm_backbone_id}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg.arch_specifier}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # Load Vision Backbone

    # Load Video Backbone
    overwatch.info(f"Loading Video Backbone [bold]{model_cfg.video_backbone_ids}[/]")
    video_backbones, _ = get_video_backbone_and_transform(
        model_cfg.video_backbone_ids,
        image_resize_strategy=model_cfg.image_resize_strategy,
        num_frames=model_cfg.num_frames,
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg.llm_backbone_id}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg.llm_backbone_id,
        llm_max_length=model_cfg.llm_max_length,
        hf_token=hf_token,
        inference_mode=True,
    )

    # Load VidLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VidLM [bold blue]{model_cfg.model_id}[/] from Checkpoint; Freezing Weights 🥶")

    vidlm = MERV.from_pretrained(
        checkpoint_pt,
        model_cfg.model_id,
        video_backbones,
        llm_backbone,
        arch_specifier=model_cfg.arch_specifier,
        adapter=model_cfg.feature_fusion,
        visual_feature_length=model_cfg.visual_feature_length,
        projector_token_length=model_cfg.projector_token_length,
    )

    return vidlm
