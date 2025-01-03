"""
llama3.py

Class definition for all LLMs derived from LlamaForCausalLM.
"""

from typing import Optional, Type

import torch
from torch import nn as nn
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from merv.models.backbones.llm.base_llm import HFCausalLLMBackbone
from merv.models.backbones.llm.prompting import (
    LLaMa2ChatPromptBuilder,
    LLaMa31PromptBuilder,
    PromptBuilder,
    PurePromptBuilder,
)

# Registry =>> Support LLaMa-3 Models (from HF Transformers)
# fmt: off
LLAMA3_MODELS = {
    # === Pure Meta LLaMa-3 (non-instruct/chat-tuned) Models ===
    "llama3-8b-pure": {
        "llm_family": "llama3", "llm_cls": LlamaForCausalLM, "hf_hub_path": "meta-llama/Meta-Llama-3-8B"
    },

    # === Meta LLaMa-3 Chat Models ===
    "llama3-8b-chat": {
        "llm_family": "llama3", "llm_cls": LlamaForCausalLM, "hf_hub_path": "meta-llama/Meta-Llama-3-8B-Instruct"
    },

    # === Meta LLaMa-3.1 Chat Models ===
    "llama3.1-8b-chat": {
        "llm_family": "llama3", "llm_cls": LlamaForCausalLM, "hf_hub_path": "meta-llama/Meta-Llama-3.1-8B-Instruct"
    },
}
# fmt: on


class LLaMa3LLMBackbone(HFCausalLLMBackbone):
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
            **LLAMA3_MODELS[llm_backbone_id],
        )

        # [Special Case] LLaMa-3 PAD Token Handling --> for clarity, we add an extra token (and resize)
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        if self.identifier.startswith("llama3-") and self.identifier.endswith("-pure"):
            return PurePromptBuilder

        elif self.identifier.startswith("llama3-") and self.identifier.endswith("-chat"):
            return LLaMa2ChatPromptBuilder

        raise ValueError(f"No PromptBuilder defined for LLM Backbone `{self.identifier}`")

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return LlamaDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """LLaMa-2 was trained in BF16; see https://huggingface.co/docs/transformers/main/model_doc/llama2."""
        return torch.bfloat16


class LLaMA31LLMBackbone(HFCausalLLMBackbone):
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
            **LLAMA3_MODELS[llm_backbone_id],
        )
        self.tokenizer.pad_token = "<|finetune_right_pad_id|>"
        self.tokenizer.pad_token_id = 128004

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        return LLaMa31PromptBuilder

    # below two copied from above.
    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return LlamaDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """LLaMa-2 was trained in BF16; see https://huggingface.co/docs/transformers/main/model_doc/llama2."""
        return torch.bfloat16
