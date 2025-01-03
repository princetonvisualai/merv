import argparse
import ast
import json
import os
import time
import openai
from tqdm import tqdm
import io
import asyncio

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-4o-mini")
    parser.add_argument("--ckpt-name", default=r"", help="Name of checkpoint.")
    parser.add_argument("--benchmark", default=r"", help="Benchmark.")
    args = parser.parse_args()
    return args

async def run(message, qa_set, completed_files, key, client):
    try:
        completion = client.chat.completions.create(model="gpt-4o-mini-2024-07-18", messages=message, max_tokens=500)

        response = await completion
        response = response.dict()

        response_message = response["choices"][0]["message"]["content"]
        response_dict = ast.literal_eval(response_message)

        assert isinstance(response_dict, dict)
        assert "score" in response_dict and "pred" in response_dict
        assert isinstance(response_dict["score"], int) or isinstance(response_dict["score"], float)
        assert response_dict["pred"] in ["yes", "no"]
        result_qa_pair = [response_dict, qa_set]

        # Save the question-answer pairs to a json file.
        completed_files[key] = result_qa_pair
        # with open(saveloc, "w") as f:
        #     json.dump(result_qa_pair, f)

    except SyntaxError as e:
        print(f"Error processing file '{key}': {e}")
    except ValueError as e:
        print(f"Error processing file '{key}': {e}")

    except openai.BadRequestError as e:
        print(f"Error processing file '{key}': {e}")

        if "invalid_prompt" in e.code:
            print(message)
            return key
        if "content_filter" in e.code:
            print(message)
            return key

    except Exception as e:
        print(type(e))
        print(f"Error processing file '{key}': {e}")

async def annotate(client, prediction_set, caption_files, completed_files, args):
    cs = []
    skip_due_to_contentfilters = set()
    for file in tqdm(caption_files):
        key = file
        qa_set = prediction_set[key]
        question = qa_set["q"]
        answer = qa_set["a"]
        pred = qa_set["pred"]

        messages = [
            {
                "role": "system",
                "content": "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Evaluate the correctness of the prediction compared to the answer.",
            },
            {
                "role": "user",
                "content": "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}.",
            },
        ]
        cs.append(asyncio.create_task(run(messages, qa_set, completed_files, key, client)))

        if len(cs) > 500:
            rs = await asyncio.gather(*cs)
            cs = []

            skip_due_to_contentfilters = skip_due_to_contentfilters.union(set(rs))

    if len(cs) > 0:
        rs = await asyncio.gather(*cs)
        skip_due_to_contentfilters = skip_due_to_contentfilters.union(set(rs))

    return skip_due_to_contentfilters

def main():
    
    args = parse_args()

    pred_path = f"eval_result/{args.ckpt_name}/{args.benchmark}_pred_merge.jsonl"
    gpt_batch_output_path = f"eval_result/{args.ckpt_name}/{args.benchmark}_gpt.json"

    with open(".openai_key") as f:
        OPENAI_KEY = f.read().strip()
    if len(OPENAI_KEY) == 0:
        raise ValueError("No OpenAI API keys found in .openai_key")
    client = openai.AsyncOpenAI(api_key=OPENAI_KEY)
    if not os.path.exists(pred_path):
        print("File", pred_path, "does not exist")
        exit()


    file = open(pred_path)
    new_pred_contents = [eval(i.strip()) for i in file.readlines()]
    file.close()

    # Generating list of id's and corresponding files
    id_list = [x["question_id"] for x in new_pred_contents]

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in new_pred_contents:
        q_id = sample["question_id"]
        question = sample["question"]
        answer = sample["answer"]
        pred = sample["pred"]
        qa_set = {"q": question, "a": answer, "pred": pred}
        prediction_set[q_id] = qa_set

    
    num_incomplete_files = []
    skip_due_to_contentfilters = set()
    while True:
        # Files that have not been processed yet.
        print(f"content filter files: {len(skip_due_to_contentfilters)}")
        completed_files = json.load(open(gpt_batch_output_path)) if os.path.exists(gpt_batch_output_path) else {}
        json.dump(completed_files, open(gpt_batch_output_path, "w"))
        print(f"completed_files: {len(completed_files)}")

        # Files that have not been processed yet.
        incomplete_files = [f for f in id_list if ((f not in completed_files) and (f not in skip_due_to_contentfilters))]
        print(f"incomplete_files: {len(incomplete_files)}")

        # Break the loop when there are no incomplete files
        if len(incomplete_files) == 0:
            break

        rs = asyncio.run(annotate(client, prediction_set, incomplete_files, completed_files, args))
        skip_due_to_contentfilters = skip_due_to_contentfilters.union(rs)

        # if incomplete_file is repeatly the same, break the loop. 
        num_incomplete_files.append(len(incomplete_files))
        if len(num_incomplete_files) > 10:
            if all([i == len(incomplete_files) for i in num_incomplete_files[-10:]]):
                break

        json.dump(completed_files, open(gpt_batch_output_path, "w"))


    combined_contents = {}
    combined_contents = json.load(open(gpt_batch_output_path))

    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    for key, result in combined_contents.items():
        # Computing score
        count += 1
        try:
            score_match = result[0]["score"]
        except:
            print(result)
            exit()
        score = int(score_match)
        score_sum += score

        # Computing accuracy
        pred = result[0]["pred"]
        if "yes" in pred.lower():
            yes_count += 1
        elif "no" in pred.lower():
            no_count += 1

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    print("\nYes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)
    print("Average score:", average_score)

    
if __name__ == "__main__":
    main()