# python scripts/eval.py --model_path "videollava-clip-dinovid-bs64" \
#     --eval_dataset MSVDsmall --num_chunks 4 --chunk_idx 0
# run like above
import glob
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import torch
from tqdm.auto import tqdm

from merv.conf.models import ModelConfig, ModelRegistry
from merv.models.load_vid import load_vid
from merv.overwatch import initialize_overwatch
from merv.util import get_statistics


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class EvalConfig:
    # fmt: off
    model_path: Union[str, Path] = "merv-full"                    # Path to Pretrained VidLM (on disk or HF Hub)
                                                                  # e.g. "merv-full" for HuggingFace Hub
                                                                  # or "trainID" for "runs/trainID/checkpoints/latest-checkpoint.pt"
    hf_token: Union[str, Path] = Path(".hf_token")                # Environment variable or Path to HF Token

    # Default Generation Parameters =>> subscribes to HuggingFace's GenerateMixIn API
    do_sample: bool = False
    temperature: float = 1.0
    max_new_tokens: int = 512
    min_length: int = 1
    eval_dataset: str = "MSVD"
    num_chunks: int = 1                        # Number of parallelization
    chunk_idx: int = 0                         # Parallelization index
    filename_question: str = 'test_q'          # filename for the question
    filename_answer: str = 'test_a'            # filename for the answer

@draccus.wrap()
def evaluate(cfg: EvalConfig) -> None:
    #### LOAD EVAL DATASET ####
    cfg.model_path = Path(cfg.model_path)
    os.makedirs("./eval_result" / cfg.model_path, exist_ok=True)

    benchmark = cfg.eval_dataset.replace("_token", "")
    filename_q = cfg.filename_question
    filename_a = cfg.filename_answer

    questions = json.load(open(f"./eval_data/{benchmark}/{filename_q}.json"))
    print(f"Number of Questions in {benchmark}: {len(questions)}")
    all_questions_id = set([item["question_id"] for item in questions])
    questions = get_chunk(questions, cfg.num_chunks, cfg.chunk_idx)
    print(f"Number of Questions in {benchmark} that this machine has to run: {len(questions)}")

    answers = json.load(open(f"./eval_data/{benchmark}/{filename_a}.json"))
    answers_dict = {item["question_id"]: item for item in answers}

    #### LOAD MODEL ####
    hf_token = Path(".hf_token").read_text().strip()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if os.path.exists(os.path.join('runs', cfg.model_path)):
        
        cfg.full_path_ckpt = "runs" / cfg.model_path
        print('Loading checkpoint from local path', str(cfg.full_path_ckpt))

        loaded_cfg = json.load(open(cfg.full_path_ckpt / "config.json", "r"))
        loaded_cfg["model"].pop("type", None)
        loaded_cfg["model"].pop("vidlm_id", None)
        model_cfg = ModelConfig.get_choice_class(ModelRegistry.MERV_BASE.model_id)(**loaded_cfg["model"])
        vidlm = load_vid(cfg.full_path_ckpt, hf_token=hf_token)

    else:
        print('Loading checkpoint from HuggingFace Hub', str(cfg.model_path))
        vidlm, model_cfg = load_vid(str(cfg.model_path), hf_token=hf_token, get_model_cfg=True)

    vidlm.to(device, dtype=torch.bfloat16)

    #### Get statistics like FLOPs, Params ####
    if not os.path.exists("./eval_result" / cfg.model_path / "flops.json"):
        overwatch.info("Getting model statistics: FLOPs + Params")
        macs, params = get_statistics(vidlm=vidlm, num_frames=model_cfg.num_frames)
        overwatch.info(f"Model FLOPs: {macs}, Params: {params}")
        json.dump({"macs": macs, "params": params}, open("./eval_result" / cfg.model_path / "flops.json", "w"))

    #### CONTINUE EVAL ####
    if os.path.exists(
        "./eval_result" / cfg.model_path / f"{cfg.eval_dataset}_pred_{cfg.num_chunks}_{cfg.chunk_idx}_done.jsonl"
    ):  # if the chunk is already done, add the video id to the done list
        done = [
            line
            for line in open(
                "./eval_result" / cfg.model_path / f"{cfg.eval_dataset}_pred_{cfg.num_chunks}_{cfg.chunk_idx}_done.jsonl"
            ).readlines()
        ]
        done_ids = set([json.loads(item)["question_id"] for item in done])
        questions = [item for item in questions if (item["question_id"] not in done_ids)]

    elif os.path.exists(
        "./eval_result" / cfg.model_path / f"{cfg.eval_dataset}_pred_{cfg.num_chunks}_{cfg.chunk_idx}.jsonl"
    ): # if its not done, but file exists, then add the video id to the done list
        done = [
            line
            for line in open(
                "./eval_result" / cfg.model_path / f"{cfg.eval_dataset}_pred_{cfg.num_chunks}_{cfg.chunk_idx}.jsonl"
            ).readlines()
        ]
        done_ids = set([json.loads(item)["question_id"] for item in done])
        questions = [item for item in questions if (item["question_id"] not in done_ids)]

    else:
        # if there are files with different chunk_num, (e.g., Previously ran with 4 chunks, now running with 8 chunks)
        # go through all of them and add the video id to the done list. 
        previous_jsonls = set(
            glob.glob(str("./eval_result" / cfg.model_path / f"{cfg.eval_dataset}_pred_*.jsonl"))
        ) - set(glob.glob(str("./eval_result" / cfg.model_path / f"{cfg.eval_dataset}_pred_{cfg.num_chunks}_*.jsonl")))

        done = []  # list of strings
        for path in previous_jsonls:
            done += [line for line in open(path).readlines()]

        # ruff: noqa: E722
        try:
            done_ids = set([json.loads(item)["question_id"] for item in done])
            done_dict = {json.loads(item)["question_id"]: item for item in done}
        except:
            done_ids = set()
            done_dict = {}

        # questions that are already "done"
        not_done = [item for item in questions if (item["question_id"] not in done_ids)]
        # among dones, get the ones that are in current chunk
        done = [done_dict[item["question_id"]] for item in questions if (item["question_id"] in done_ids)]
        questions = not_done

    with open(
        "./eval_result" / cfg.model_path / f"{cfg.eval_dataset}_pred_{cfg.num_chunks}_{cfg.chunk_idx}.jsonl", "w"
    ) as f:
        for line in done:
            f.write(line)

        f.flush()

        for i, question in enumerate(tqdm(questions, desc=f"{cfg.eval_dataset}_{cfg.num_chunks}_{cfg.chunk_idx}")):
            prompt_builder = vidlm.llm_backbone.prompt_builder_fn(model_family="merv")

            message = question["question"] + ("\n<video>" if "_token" in cfg.eval_dataset else "")
            prompt_builder.add_turn(role="human", message=message)
            prompt_text = prompt_builder.get_prompt()

            video_name = glob.glob(f"./eval_data/{benchmark}/videos/{question['video_name']}.*")[0]

            try:
                generated_text = vidlm.generate(
                    video_name,
                    prompt_text,
                    do_sample=cfg.do_sample,
                    temperature=cfg.temperature,
                    max_new_tokens=cfg.max_new_tokens,
                    min_length=cfg.min_length,
                    num_frames=model_cfg.num_frames,
                )

                question["pred"] = generated_text
                question["message"] = message
                question = {**question, **answers_dict[question["question_id"]]}

                f.write(json.dumps(question) + "\n")
            except Exception as e:  # if video loading has an issue
                print(e)
                print(f"Issue when evaluating {video_name}")
                continue

            if i % 100 == 99:
                f.flush()

    os.rename(
        "./eval_result" / cfg.model_path / f"{cfg.eval_dataset}_pred_{cfg.num_chunks}_{cfg.chunk_idx}.jsonl",
        "./eval_result" / cfg.model_path / f"{cfg.eval_dataset}_pred_{cfg.num_chunks}_{cfg.chunk_idx}_done.jsonl",
    )

    all_jsonls = glob.glob(
        str("./eval_result" / cfg.model_path / f"{cfg.eval_dataset}_pred_{cfg.num_chunks}_*_done.jsonl")
    )

    all_done_items = {
        item["question_id"]: item
        for jsonl in all_jsonls
        for line in open(jsonl).readlines()
        if (item := json.loads(line))
    }

    if len(all_questions_id - set(all_done_items.keys())) == 0:
        print("merging")
        with open("./eval_result" / cfg.model_path / f"{cfg.eval_dataset}_pred_merge.jsonl", "w") as f:
            for item in all_done_items.values():
                f.write(json.dumps(item) + "\n")
        for jsonl in all_jsonls:
            os.remove(jsonl)


if __name__ == "__main__":
    evaluate()
