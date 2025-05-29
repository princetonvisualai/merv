"""
qwen2.py

Class definition for all LLMs derived from Qwen2ForCausalLM.
"""

from typing import Optional, Type

import torch
from torch import nn as nn
from transformers import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from merv.models.backbones.llm.base_llm import HFCausalLLMBackbone
from merv.models.backbones.llm.prompting import (
    Qwen2PromptBuilder,
    PromptBuilder,
    PurePromptBuilder,
)

# Registry =>> Support LLaMa-3 Models (from HF Transformers)
# fmt: off
QWEN2_MODELS = {
    "qwen2.5-7b-instruct": {
        "llm_family": "qwen2", "llm_cls": Qwen2ForCausalLM, "hf_hub_path": "Qwen/Qwen2.5-7B-Instruct"
    },
    "qwen2.5-3b-instruct": {
        "llm_family": "qwen2", "llm_cls": Qwen2ForCausalLM, "hf_hub_path": "Qwen/Qwen2.5-3B-Instruct"
    },
}
# fmt: on

class Qwen2LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            **QWEN2_MODELS[llm_backbone_id],
        )
        self.tokenizer.pad_token = "<|finetune_right_pad_id|>"
        self.tokenizer.pad_token_id = 128004

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        return Qwen2PromptBuilder

    # below two copied from above.
    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return Qwen2DecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """Qwen2 was trained in FP16; see https://huggingface.co/docs/transformers/main/model_doc/qwen2."""
        return torch.float16
