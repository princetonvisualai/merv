import argparse
import ast
import asyncio
import json
import os
import random
import time

import openai
import yaml
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-4o-mini")
    parser.add_argument("--ckpt-name", default=r"", help="Name of checkpoint.")
    parser.add_argument("--benchmark", default=r"", help="Benchmark.")
    # parser.add_argument("--pred_path", default=r"", help="The path to file containing prediction.")
    # parser.add_argument("--output_file", default=r"", help="The path to save annotation json files.")
    args = parser.parse_args()
    return args


async def run(message, qa_set, completed_files, key, client, openai_modelname):
    try:
        completion = client.chat.completions.create(model=openai_modelname, messages=message, max_tokens=500)

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


def query_gpt(prediction_set, caption_files, gpt_batch_input_path, gpt_batch_output_path):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """


    openai_modelname = "gpt-4o-mini-2024-07-18"

    for key in tqdm(caption_files):
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
                "For example, your response should look like this: {'pred': 'yes', 'score': 4}.",
            },
        ]

        payload = {
            "custom_id": key,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": openai_modelname,
                "messages": messages,
                "max_tokens": 500,
            },
        }
        with open(gpt_batch_input_path, "a") as f:
            f.write(json.dumps(payload) + "\n")

    client = openai.OpenAI(api_key=OPENAI_KEY)
    batch_input_file = client.files.create(
        file=open(gpt_batch_input_path, "rb"),
        purpose="batch",
    ) 

    batch_input_file_id = batch_input_file.id

    submitted_batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
        }
    )

    return submitted_batch


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    pred_path = f"eval_result/{args.ckpt_name}/{args.benchmark}_pred_merge.jsonl"
    gpt_batch_input_path = f"eval_result/{args.ckpt_name}/{args.benchmark}_gpt_batch_input.jsonl"
    gpt_batch_output_path = f"eval_result/{args.ckpt_name}/{args.benchmark}_gpt_return.jsonl"

    # Set the OpenAI API key.
    with open(".openai_key") as f:
        OPENAI_KEY = f.read().strip()
    if len(OPENAI_KEY) == 0:
        raise ValueError("No OpenAI API keys found in .openai_key")
    client = openai.OpenAI(api_key=OPENAI_KEY)

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

    ### Need to create a batch file at eval_result/{ckpt}/{benchmark}_batch.jsonl
    submitted_batch = query_gpt(prediction_set, id_list, gpt_batch_input_path, gpt_batch_output_path)

    ### Monitor now for return of the batch
    while True:
        batch_retrieved = client.batches.retrieve(submitted_batch.id)
        batch_status = batch_retrieved.status
        print(f"Batch status: {batch_status}")
        if batch_status == "completed":
            break 
        elif batch_status in ["in_progress", "validating", "finalizing"]:
            print("sleeping for 60 seconds")
            time.sleep(60)
        else:
            print("Batch failed")
            exit()

    file_response = client.files.content(batch_retrieved.output_file_id)
    with open(gpt_batch_output_path, "wb") as f:
        f.write(file_response)
    
    # Combine all the processed files into one
    combined_contents = []
    with open(gpt_batch_output_path) as f:
        for line in f:
            combined_contents.append(json.loads(line))

    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    for key, result in tqdm(combined_contents.items()):
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
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)
    print("Average score:", average_score)


if __name__ == "__main__":
    main()
