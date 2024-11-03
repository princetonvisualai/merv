"""
registry.py

Exhaustive list of pretrained VidLMs (with full descriptions / links to corresponding names and sections of paper).
"""

# === Pretrained Model Registry ===
# fmt: off
MODEL_REGISTRY = {
    # === Base MERV Config ===
    "merv-base": {
        "model_id": "merv-base",
        "names": ["MERV Base"],
        "description": {
            "name": "MERV Base",
            "optimization_procedure": "single-stage",
            "visual_representation": "LanguageBind, DINO, SigLIP, ViViT",
            "image_processing": "Letterbox",
            "language_model": "Llama 2 7B",
            "datasets": ["Video-LLaVA"],
            "train_epochs": 1,
        }
    },
    "merv-full": {
        "model_id": "merv-full",
        "names": ["MERV Full"],
        "description": {
            "name": "MERV Full",
            "optimization_procedure": "multi-stage",
            "visual_representation": "LanguageBind, DINO, SigLIP, ViViT",
            "image_processing": "Letterbox",
            "language_model": "Llama 2 7B",
            "datasets": ["Video-LLaVA"],
            "train_epochs": 1,
        }
    },
    # === Single Encoder Models ===
    "languagebind-single": {
        "model_id": "languagebind-single",
        "names": ["LanguageBind Single Encoder"],
        "description": {
            "name": "LanguageBind Single Encoder",
            "optimization_procedure": "single-stage",
            "visual_representation": "LanguageBind",
            "image_processing": "Letterbox",
            "language_model": "Llama 2 7B",
            "datasets": ["Video-LLaVA"],
            "train_epochs": 1,
        }
    },
    "dinov2-single": {
        "model_id": "dinov2-single",
        "names": ["DINOv2 Single Encoder"],
        "description": {
            "name": "DINOv2 Single Encoder",
            "optimization_procedure": "single-stage",
            "visual_representation": "DINO",
            "image_processing": "Letterbox",
            "language_model": "Llama 2 7B",
            "datasets": ["Video-LLaVA"],
            "train_epochs": 1,
        }
    },
    "vivit-single": {
        "model_id": "vivit-single",
        "names": ["ViViT Single Encoder"],
        "description": {
            "name": "ViViT Single Encoder",
            "optimization_procedure": "single-stage",
            "visual_representation": "ViViT",
            "image_processing": "Letterbox",
            "language_model": "Llama 2 7B",
            "datasets": ["Video-LLaVA"],
            "train_epochs": 1,
        }
    },
    "siglip-single": {
        "model_id": "siglip-single",
        "names": ["SigLIP Single Encoder"],
        "description": {
            "name": "SigLIP Single Encoder",
            "optimization_procedure": "single-stage",
            "visual_representation": "SigLIP",
            "image_processing": "Letterbox",
            "language_model": "Llama 2 7B",
            "datasets": ["Video-LLaVA"],
            "train_epochs": 1,
        }
    },
}

# Build Global Registry (Model ID, Name) -> Metadata
GLOBAL_REGISTRY = {name: v for k, v in MODEL_REGISTRY.items() for name in [k] + v["names"]}

# fmt: on
