"""
llama2_prompter.py

Defines a PromptBuilder for building LLaMa-2 Chat Prompts --> not sure if this is "optimal", but this is the pattern
that's used by HF and other online tutorials.

Reference: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
"""

# ruff: noqa: E501

from typing import Optional

from merv.models.backbones.llm.prompting.base_prompter import PromptBuilder

# Default System Prompt for merv Models
SYS_PROMPTS = {
    "merv": (
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language."
    ),
}


def format_system_prompt(system_prompt: str) -> str:
    return f"<<SYS>\n{system_prompt.strip()}\n<</SYS>>\n\n"


class LLaMa2ChatPromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)
        self.system_prompt = format_system_prompt(
            SYS_PROMPTS[self.model_family] if system_prompt is None else system_prompt
        )

        # LLaMa-2 Specific
        self.bos, self.eos = "<s>", "</s>"

        # Get role-specific "wrap" functions
        self.wrap_human = lambda msg: f"{self.bos}[INST] {msg} [/INST] "
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            sys_message = self.wrap_human(self.system_prompt + message)
            wrapped_message = sys_message
        elif (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1

        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, user_msg: str) -> None:
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            sys_message = self.wrap_human(self.system_prompt + user_msg)
            prompt_copy += sys_message

        else:
            human_message = self.wrap_human(user_msg)
            prompt_copy += human_message

        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        # Remove prefix <bos> because it gets auto-inserted by tokenizer!
        return self.prompt.removeprefix(self.bos).rstrip()


class LLaMa31PromptBuilder(PromptBuilder):
    def __init__(self, model_family: str) -> None:
        self.model_family = model_family

        # this can be lot easier if we use, apply_chat_template but 'return_assistant_tokens_mask' is broken in llama3.1
        # as of now.
        self.system_prompt = SYS_PROMPTS[self.model_family]
        self.turn_count = 0

        self.prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{self.system_prompt}<|eot_id|>"
        # <|begin_of_text|> is not added cuz tokenizer will add automatically.

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")

        if role == "human":
            wrapped_message = f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif role == "gpt":
            wrapped_message = f"{message}<|eot_id|>"
        else:
            raise ValueError(f"Invalid role: {role}")

        self.prompt += wrapped_message
        self.turn_count += 1
        return wrapped_message

    def get_potential_prompt(self, user_msg: str) -> None:

        assert NotImplementedError("I don't think this is ever called...")

    def get_prompt(self) -> str:

        return self.prompt
