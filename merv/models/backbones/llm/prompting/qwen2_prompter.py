from merv.models.backbones.llm.prompting.base_prompter import PromptBuilder

SYS_PROMPTS = {
    "merv": (
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language."
    ),
}

class Qwen2PromptBuilder(PromptBuilder):
    def __init__(self, model_family: str) -> None:
        self.model_family = model_family
        # this can be lot easier if we use, apply_chat_template but 'return_assistant_tokens_mask' is broken in llama3.1
        # as of now.
        self.system_prompt = SYS_PROMPTS[self.model_family]
        self.turn_count = 0

        self.prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        # <|begin_of_text|> is not added cuz tokenizer will add automatically.

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")

        if role == "human":
            # wrapped_message = f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            wrapped_message = f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
        elif role == "gpt":
            wrapped_message = f"{message}<|im_end|>"
        else:
            raise ValueError(f"Invalid role: {role}")

        self.prompt += wrapped_message
        self.turn_count += 1
        return wrapped_message

    def get_potential_prompt(self, user_msg: str) -> None:
        assert NotImplementedError("I don't think this is ever called...")

    def get_prompt(self) -> str:
        return self.prompt