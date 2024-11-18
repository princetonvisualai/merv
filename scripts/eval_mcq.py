# python scripts/eval_mcq.py --model_path "videollava-clip-dinovid-bs64" \
#    --eval_dataset Perception --num_chunks 4 --chunk_idx 0 --strategy naive
# run like above

import glob
import json
import math
import os
import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import numpy as np
import torch
from tqdm.auto import tqdm

from merv.conf import ModelConfig, ModelRegistry
from merv.models.load_vid import load_vid
from merv.overwatch import initialize_overwatch


def process_punctuation(inText):
    import re

    outText = inText
    punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]
    commaStrip = re.compile("(\d)(,)(\d)")  # noqa: W605
    periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")  # noqa: W605
    for p in punct:
        if (p + " " in inText or " " + p in inText) or (re.search(commaStrip, inText) is not None):
            outText = outText.replace(p, "")
        else:
            outText = outText.replace(p, " ")
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    articles = ["a", "an", "the"]
    manualMap = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    contractions = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = " ".join(outText)
    return outText


def process_answer(answer):
    """Taken from MMBench."""
    answer = answer.replace("\n", " ")
    answer = answer.replace("\t", " ")
    answer = answer.strip()
    answer = process_punctuation(answer)
    answer = process_digit_article(answer)
    return answer


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_secret_test_output(new_pred_contents, path):
    """This gives the predictions in the required format to upload and check on official test set where we do not know ground truths"""
    all_answer = []
    for sample in new_pred_contents:
        a_dic = {"example_id": sample["question_id"]}
        p = sample["pred"].lower()[:1]
        if p.lower() == "a":
            a_dic["pred_ans"] = 0
        else:
            a_dic["pred_ans"] = 1
        all_answer.append(a_dic)

    with open(f"{path}_secret_test_answer.jsonl", "w") as f:
        for entry in all_answer:
            json.dump(entry, f)
            f.write("\n")


def get_NExTQA_breakdown(preds):
    """Takes as input in the format of NExT-QA_naive_gpt"""
    group = {"CW": [], "CH": [], "TN": [], "TC": [], "DC": [], "DL": [], "DO": []}
    for qns_id, question in preds.items():
        qtype = qns_id.split("_")[0]
        # (combine temporal qns of previous and next as 'TN')
        if qtype == "TP":
            qtype = "TN"
        group[qtype].append(qns_id)

    group_acc, group_cnt = {"CW": 0, "CH": 0, "TN": 0, "TC": 0, "DC": 0, "DL": 0, "DO": 0}, {
        "CW": 0,
        "CH": 0,
        "TN": 0,
        "TC": 0,
        "DC": 0,
        "DL": 0,
        "DO": 0,
    }
    overall_acc, overall_cnt = {"C": 0, "T": 0, "D": 0}, {"C": 0, "T": 0, "D": 0}
    all_acc, all_cnt = 0, 0
    for qtype, qns_ids in group.items():
        cnt, acc = 0, 0
        for qid in qns_ids:
            cnt += 1
            if preds[qid][0]["pred"] == "yes":
                acc += 1

        group_cnt[qtype] = cnt
        group_acc[qtype] += acc
        overall_acc[qtype[0]] += acc
        overall_cnt[qtype[0]] += cnt
        all_acc += acc
        all_cnt += cnt

    for qtype, value in overall_acc.items():
        group_acc[qtype], group_cnt[qtype] = value, overall_cnt[qtype]

    final_group_acc = {}
    for qtype, acc in group_acc.items():
        if group_cnt[qtype] == 0:
            final_group_acc[qtype] = 0.0
        else:
            final_group_acc[qtype] = acc / group_cnt[qtype]

    total_acc = all_acc / all_cnt

    return total_acc, final_group_acc


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class EvalConfig:
    # fmt: off
    model_path: Union[str, Path] = (                                    # Path to Pretrained VidLM (on disk or HF Hub)
        "videollava-clip-dinovid-bs64"
    )

    hf_token: Union[str, Path] = Path(".hf_token")                      # Environment variable or Path to HF Token

    # Default Generation Parameters =>> subscribes to HuggingFace's GenerateMixIn API
    do_sample: bool = False
    temperature: float = 1.0
    max_new_tokens: int = 512
    min_length: int = 1
    eval_dataset: str = "MSVD"  # -1 to run all, 0~3 select specific
    num_chunks: int = 1
    chunk_idx: int = 0
    strategy: str = 'naive'
    filename_question: str = 'test_q'
    filename_answer: str = 'test_a'
    full_path_ckpt: Union[str, Path] = (
        "videollava-clip-dinovid-bs64"
    )


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
    cfg.model_path = Path(cfg.model_path)
    cfg.full_path_ckpt = Path(cfg.full_path_ckpt)
    print(cfg)

    os.makedirs("./eval_result" / cfg.model_path, exist_ok=True)

    # load saved cfg. (This is done inside load_vid, but we do again)
    # loaded_cfg = json.load(open("runs" / cfg.model_path / "config.json", "r"))
    loaded_cfg = json.load(open(cfg.full_path_ckpt / "config.json", "r"))
    loaded_cfg["model"].pop("type", None)
    loaded_cfg["model"].pop("vidlm_id", None)
    model_cfg = ModelConfig.get_choice_class(ModelRegistry.MERV_BASE.model_id)(**loaded_cfg["model"])

    benchmark = cfg.eval_dataset.replace("_token", "")
    filename_q = cfg.filename_question
    filename_a = cfg.filename_answer

    questions = json.load(open(f"./eval_data/{benchmark}/{filename_q}.json"))
    all_questions_id = set([item["question_id"] for item in questions])
    questions = get_chunk(questions, cfg.num_chunks, cfg.chunk_idx)

    answers = json.load(open(f"./eval_data/{benchmark}/{filename_a}.json"))
    answers_dict = {item["question_id"]: item for item in answers}

    # if there is jsonl with same num_chunk and chunk_idx, continue from it.

    print(f"Original Size: {len(questions)}")
    if os.path.exists(
        "./eval_result"
        / cfg.model_path
        / f"{cfg.eval_dataset}_{cfg.strategy}_pred_{cfg.num_chunks}_{cfg.chunk_idx}.jsonl"
    ) or os.path.exists(
        "./eval_result"
        / cfg.model_path
        / f"{cfg.eval_dataset}_{cfg.strategy}_pred_{cfg.num_chunks}_{cfg.chunk_idx}_done.jsonl"
    ):
        if os.path.exists(
            "./eval_result"
            / cfg.model_path
            / f"{cfg.eval_dataset}_{cfg.strategy}_pred_{cfg.num_chunks}_{cfg.chunk_idx}_done.jsonl"
        ):
            print(
                "previous "
                + str(
                    "./eval_result"
                    / cfg.model_path
                    / f"{cfg.eval_dataset}_{cfg.strategy}_pred_{cfg.num_chunks}_{cfg.chunk_idx}_done.jsonl"
                )
                + " exists"
            )
            done = [
                line
                for line in open(
                    "./eval_result"
                    / cfg.model_path
                    / f"{cfg.eval_dataset}_{cfg.strategy}_pred_{cfg.num_chunks}_{cfg.chunk_idx}_done.jsonl"
                ).readlines()
            ]

        else:
            print(
                "previous "
                + str(
                    "./eval_result"
                    / cfg.model_path
                    / f"{cfg.eval_dataset}_{cfg.strategy}_pred_{cfg.num_chunks}_{cfg.chunk_idx}.jsonl"
                )
                + " exists"
            )
            done = [
                line
                for line in open(
                    "./eval_result"
                    / cfg.model_path
                    / f"{cfg.eval_dataset}_{cfg.strategy}_pred_{cfg.num_chunks}_{cfg.chunk_idx}.jsonl"
                ).readlines()
            ]
        # done_ids = set([json.loads(item)["question_id"] for item in done])
        # some lines are broken, due to IO issues. Only get the none-broken ones.
        done_ids = set()
        done_dict = {}
        for item in done:
            try:
                j = json.loads(item)
                assert "video_name" in j
                assert "num_option" in j
                assert "question_id" in j
                assert "question" in j
                assert "pred" in j
                assert "answer_id" in j
                assert "answer" in j
                assert "question_text" in j
                assert "answer_char" in j
                done_ids.add(j["question_id"])
                done_dict[j["question_id"]] = item
            except:
                pass
        done = [done_dict[item["question_id"]] for item in questions if (item["question_id"] in done_ids)]
        questions = [item for item in questions if (item["question_id"] not in done_ids)]

    else:
        # if there is jsonl with different num_chunk, get all
        previous_jsonls = set(
            glob.glob(str("./eval_result" / cfg.model_path / f"{cfg.eval_dataset}_{cfg.strategy}_pred_*.jsonl"))
        ) - set(
            glob.glob(
                str(
                    "./eval_result" / cfg.model_path / f"{cfg.eval_dataset}_{cfg.strategy}_pred_{cfg.num_chunks}_*.jsonl"
                )
            )
        )
        print(previous_jsonls)
        print("exists")

        done = []  # list of strings
        for path in previous_jsonls:
            done += [line for line in open(path).readlines()]
        # done_ids = set([json.loads(item)["question_id"] for item in done])
        # done_dict = {json.loads(item)["question_id"]: item for item in done}

        done_ids = set()
        done_dict = {}
        for item in done:
            try:
                j = json.loads(item)
                assert "video_name" in j
                assert "num_option" in j
                assert "question_id" in j
                assert "question" in j
                assert "pred" in j
                assert "answer_id" in j
                assert "answer" in j
                assert "question_text" in j
                assert "answer_char" in j
                done_ids.add(j["question_id"])
                done_dict[j["question_id"]] = item

            except:
                pass

        # questions that are already "done"
        not_done = [item for item in questions if (item["question_id"] not in done_ids)]
        # among dones, get the ones that are in current chunk
        done = [done_dict[item["question_id"]] for item in questions if (item["question_id"] in done_ids)]
        questions = not_done

    print(f"New Size: {len(questions)}")

    if len(questions) > 0:
        # load model
        hf_token = Path(".hf_token").read_text().strip()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        vidlm = load_vid(cfg.full_path_ckpt, hf_token=hf_token)
        vidlm = vidlm.to(device, dtype=torch.bfloat16)

    with open(
        "./eval_result"
        / cfg.model_path
        / f"{cfg.eval_dataset}_{cfg.strategy}_pred_{cfg.num_chunks}_{cfg.chunk_idx}.jsonl",
        "w",
    ) as f:
        for line in done:
            f.write(line)
        f.flush()

        for i, question in enumerate(tqdm(questions, desc=f"{cfg.eval_dataset}_{cfg.num_chunks}_{cfg.chunk_idx}")):
            prompt_builder = vidlm.llm_backbone.prompt_builder_fn(model_family="merv")

            question_text, answer_char = prepare_mcqa_question(question, answers_dict[question["question_id"]], cfg)

            if "_token" in cfg.eval_dataset:
                question_text = "<video>\n" + question_text

            prompt_builder.add_turn(role="human", message=question_text)
            prompt_text = prompt_builder.get_prompt()

            if benchmark in ["VLEPdev", "VLEPdevsmall", "VLEP_secret_test_set"]:
                video_name = glob.glob(f"./eval_data/{benchmark}/videos/{question['video_name']}")[0]
            else:
                video_name = glob.glob(f"./eval_data/{benchmark}/videos/{question['video_name']}.*")[0]

            clip_start_sec = question["time"][0] if "time" in question else 0.0
            clip_end_sec = question["time"][1] if "time" in question else None

            print(video_name)
            generated_text = vidlm.generate(
                video_name,
                prompt_text,
                do_sample=False,
                temperature=1.0,
                max_new_tokens=512,
                min_length=1,
                num_frames=model_cfg.num_frames,
                clip_start_sec=clip_start_sec,
                clip_end_sec=clip_end_sec,
            )

            question["pred"] = generated_text

            question = {**question, **answers_dict[question["question_id"]]}
            question["question_text"] = question_text
            question["answer_char"] = answer_char

            f.write(json.dumps(question) + "\n")

            if i % 100 == 99:
                f.flush()

    os.rename(
        "./eval_result"
        / cfg.model_path
        / f"{cfg.eval_dataset}_{cfg.strategy}_pred_{cfg.num_chunks}_{cfg.chunk_idx}.jsonl",
        "./eval_result"
        / cfg.model_path
        / f"{cfg.eval_dataset}_{cfg.strategy}_pred_{cfg.num_chunks}_{cfg.chunk_idx}_done.jsonl",
    )

    # check if all the jsonls of this num_chunks contains all the answers.
    # If so, it means this index is executed last, merge the files and stuff
    all_jsonls = glob.glob(
        str("./eval_result" / cfg.model_path / f"{cfg.eval_dataset}_{cfg.strategy}_pred_{cfg.num_chunks}_*_done.jsonl")
    )
    all_done_items = {
        item["question_id"]: item
        for jsonl in all_jsonls
        for line in open(jsonl).readlines()
        if (item := json.loads(line))
    }

    if len(all_questions_id - set(all_done_items.keys())) == 0:
        with open("./eval_result" / cfg.model_path / f"{cfg.eval_dataset}_{cfg.strategy}_pred_merge.jsonl", "w") as f:
            for item in all_done_items.values():
                f.write(json.dumps(item) + "\n")
        for jsonl in all_jsonls:
            os.remove(jsonl)

    if os.path.exists("./eval_result" / cfg.model_path / f"{cfg.eval_dataset}_{cfg.strategy}_pred_merge.jsonl"):
        all_done_items = {
            item["question_id"]: item
            for line in open(
                "./eval_result" / cfg.model_path / f"{cfg.eval_dataset}_{cfg.strategy}_pred_merge.jsonl"
            ).readlines()
            if (item := json.loads(line))
        }

        new_pred_contents = list(all_done_items.values())

        # Generating list of id's and corresponding files
        [x["question_id"] for x in new_pred_contents]
        # caption_files = [f"{id}.json" for id in id_list]

        if "secret_test" in benchmark:
            if "VLEP" in benchmark or "TVQA" in benchmark:
                get_secret_test_output(new_pred_contents, os.path.join("./eval_result", cfg.model_path, f"{benchmark}"))
                exit(0)
            else:
                raise NotImplementedError

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
            completed_files, open("./eval_result" / cfg.model_path / f"{cfg.eval_dataset}_{cfg.strategy}_gpt.json", "w")
        )

        accuracy = yes_count / (yes_count + no_count)
        print("Yes count:", yes_count)
        print("No count:", no_count)
        print("Accuracy:", accuracy)

        if "NExT-QA" in benchmark:
            total_acc, final_group_acc = get_NExTQA_breakdown(completed_files)

            print("NExT-QA breakdown:", final_group_acc)
            print(total_acc)


if __name__ == "__main__":
    evaluate()
