# python scripts/eval_mcq.py --model_path "merv-full" --eval_dataset Perception --num_chunks 4 --chunk_idx 0

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

from merv.conf import ModelConfig, ModelRegistry
from merv.models.load_vid import load_vid
from merv.overwatch import initialize_overwatch


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
    eval_dataset: str = "Perception" 
    num_chunks: int = 1                        # Number of parallelization
    chunk_idx: int = 0                         # Parallelization index
    strategy: str = 'naive'                    # Strategy for extracting the MCQ option from the response
    filename_question: str = 'test_q'          # filename for the question
    filename_answer: str = 'test_a'            # filename for the answer


def prepare_mcqa_question(sample, gt_answer, cfg):
    """
    Formats the input question based on the strategy requested.
    """
    if cfg.strategy == "naive":
        question = sample["question"]
        choice_list = sample["options"]
        mapping = ["A. ", "B. ", "C. ", "D. ", "E. "]
        num_answers = sample["num_option"]
        choices = "\n".join([mapping[i] + c for i, c in enumerate(choice_list)])
        letters = ", ".join([mapping[c][0] for c in range(num_answers)])

        # ruff: noqa
        prompt = f"""{question} Select the correct answer from the following options. Write your answer as only one of {letters} and nothing else.

    {choices}"""
        answer = mapping[gt_answer["answer_id"]][0]
        return prompt, answer


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

    #### OUTPUT FILE NAMES ####
    temp_output_path = "./eval_result" / cfg.model_path\
        / f"{cfg.eval_dataset}_{cfg.strategy}_pred_{cfg.num_chunks}_{cfg.chunk_idx}.jsonl"
    done_output_path = "./eval_result" / cfg.model_path\
        / f"{cfg.eval_dataset}_{cfg.strategy}_pred_{cfg.num_chunks}_{cfg.chunk_idx}_done.jsonl"
    done_outputs_path = "./eval_result" / cfg.model_path\
        / f"{cfg.eval_dataset}_{cfg.strategy}_pred_{cfg.num_chunks}_*_done.jsonl"
    done_output_merged_path = "./eval_result" / cfg.model_path / f"{cfg.eval_dataset}_{cfg.strategy}_pred_merge.jsonl"
    accuracyfile_path = "./eval_result" / cfg.model_path / f"{cfg.eval_dataset}_{cfg.strategy}_gpt.json"

    #### RUN EVAL AND PRINT ACC ####
    with open(temp_output_path, "w") as f:
        for i, question in enumerate(tqdm(questions, desc=f"{cfg.eval_dataset}_{cfg.num_chunks}_{cfg.chunk_idx}")):
            prompt_builder = vidlm.llm_backbone.prompt_builder_fn(model_family="merv")

            question_text, answer_char = prepare_mcqa_question(question, answers_dict[question["question_id"]], cfg)

            if "_token" in cfg.eval_dataset:
                question_text = "<video>\n" + question_text

            prompt_builder.add_turn(role="human", message=question_text)
            prompt_text = prompt_builder.get_prompt()

            if os.path.isdir(f"./eval_data/{benchmark}/videos/{question['video_name']}"):
                video_name = f"./eval_data/{benchmark}/videos/{question['video_name']}"
            else:
                video_name = glob.glob(f"./eval_data/{benchmark}/videos/{question['video_name']}.*")[0]

            clip_start_sec = question["time"][0] if "time" in question else 0.0
            clip_end_sec = question["time"][1] if "time" in question else None
            end_frame = question["end_frame"] if "end_frame" in question else None

            print(video_name)
            generated_text = vidlm.generate(
                video_name,
                prompt_text,
                do_sample=cfg.do_sample,
                temperature=cfg.temperature,
                max_new_tokens=cfg.max_new_tokens,
                min_length=cfg.min_length,
                num_frames=model_cfg.num_frames,
                clip_start_sec=clip_start_sec,
                clip_end_sec=clip_end_sec,
                end_frame=end_frame,
            )

            question["pred"] = generated_text

            question = {**question, **answers_dict[question["question_id"]]}
            question["question_text"] = question_text
            question["answer_char"] = answer_char

            f.write(json.dumps(question) + "\n")

            if i % 100 == 99:
                f.flush()
    os.rename(temp_output_path, done_output_path)


    # check if all the jsonls of this num_chunks contains all the answers.
    # If so, it means this index is executed last, merge the files and stuff
    all_jsonls = glob.glob(str(done_outputs_path))
    all_done_items = {
        item["question_id"]: item
        for jsonl in all_jsonls
        for line in open(jsonl).readlines()
        if (item := json.loads(line))
    }

    if len(all_questions_id - set(all_done_items.keys())) == 0:
        with open(done_output_merged_path, "w") as f:
            for item in all_done_items.values():
                f.write(json.dumps(item) + "\n")
        for jsonl in all_jsonls:
            os.remove(jsonl)

    if os.path.exists(done_output_merged_path):
        all_done_items = {
            item["question_id"]: item
            for line in open(
                done_output_merged_path
            ).readlines()
            if (item := json.loads(line))
        }

        new_pred_contents = list(all_done_items.values())

        # Generating list of id's and corresponding files
        [x["question_id"] for x in new_pred_contents]

        # Preparing dictionary of question-answer sets
        completed_files = {}

        yes_count = 0
        no_count = 0
        for sample in new_pred_contents:
            sample["acc"] = sample["pred"].lower()[:1] == sample["answer_char"].lower()
            completed_files[sample["question_id"]] = [{"pred": "yes" if sample["acc"] else "no", "score": 0}, sample]

            if sample["acc"]:
                yes_count += 1
            else:
                no_count += 1

        json.dump(
            completed_files, open(accuracyfile_path, "w")
        )

        accuracy = yes_count / (yes_count + no_count)
        print("Yes count:", yes_count)
        print("No count:", no_count)
        print("Accuracy:", accuracy)

if __name__ == "__main__":
    evaluate()
