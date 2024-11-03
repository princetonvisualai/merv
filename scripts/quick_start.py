from pathlib import Path

import torch

from merv import load_vid

hf_token = Path(".hf_token").read_text().strip()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Or a local path if the models are already downloaded
vidlm = load_vid("merv-full", hf_token=hf_token)
vidlm.to(device, dtype=torch.bfloat16)

# Run on example Perception Test video and specify a prompt
video_path = "./assets/video_10336_short.mp4"
user_prompt = "Describe what is happening in this video."

# Build prompt
prompt_builder = vidlm.get_prompt_builder()
prompt_builder.add_turn(role="human", message=user_prompt)
prompt_text = prompt_builder.get_prompt()

# Generate!
generated_text = vidlm.generate(
    video_path,
    prompt_text,
    num_frames=[16, 16, 32, 16],  # get from model config
    do_sample=True,
    temperature=0.4,
    max_new_tokens=512,
    min_length=1,
)

print(generated_text)
