import argparse
import ast
import asyncio
import json
import os
import random

import openai
import yaml
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", default=r"", help="The path to file containing prediction.")
    parser.add_argument("--output_file", default=r"", help="The path to save annotation json files.")
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


async def annotate(prediction_set, caption_files, completed_files, args):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    # Set the OpenAI API key.
    openai_api_info = yaml.safe_load(open(".oai_keys.yaml"))
    if len(openai_api_info["api_keys"]) == 0:
        raise ValueError("No OpenAI API keys found in .oai_keys.yaml")

    # Pick a random api key for load balancing
    api = random.choice(openai_api_info["api_keys"])

    openai.api_key = api["api_key"]
    client = openai.AsyncAzureOpenAI(
        api_key=api["api_key"],
        api_version=api["api_version"],
        azure_endpoint=api["azure_endpoint"],
    )

    openai_modelname = "gpt-35-turbo"
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

        cs.append(asyncio.create_task(run(messages, qa_set, completed_files, key, client, openai_modelname)))

        if len(cs) > 20:
            rs = await asyncio.gather(*cs)
            cs = []

            skip_due_to_contentfilters = skip_due_to_contentfilters.union(set(rs))

    if len(cs) > 0:
        rs = await asyncio.gather(*cs)
        skip_due_to_contentfilters = skip_due_to_contentfilters.union(set(rs))

    return skip_due_to_contentfilters


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    if not os.path.exists(args.pred_path):
        print("File", args.pred_path, "does not exist")
        exit()

    file = open(args.pred_path)
    new_pred_contents = [eval(i.strip()) for i in file.readlines()]

    # Generating list of id's and corresponding files
    id_list = [x["question_id"] for x in new_pred_contents]

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in new_pred_contents:
        id = sample["question_id"]
        question = sample["question"]
        answer = sample["answer"]
        pred = sample["pred"]
        qa_set = {"q": question, "a": answer, "pred": pred}
        prediction_set[id] = qa_set

    # While loop to ensure that all captions are processed.

    num_incomplete_files = []
    skip_due_to_contentfilters = set()
    while True:
        # Files that have not been processed yet.
        print(f"content filter files: {len(skip_due_to_contentfilters)}")
        completed_files = json.load(open(args.output_file)) if os.path.exists(args.output_file) else {}
        json.dump(completed_files, open(args.output_file, "w"))
        print(f"completed_files: {len(completed_files)}")

        # Files that have not been processed yet.
        incomplete_files = [f for f in id_list if ((f not in completed_files) and (f not in skip_due_to_contentfilters))]
        print(f"incomplete_files: {len(incomplete_files)}")

        # Break the loop when there are no incomplete files
        if len(incomplete_files) == 0:
            break

        rs = asyncio.run(annotate(prediction_set, incomplete_files, completed_files, args))

        skip_due_to_contentfilters = skip_due_to_contentfilters.union(rs)

        # if incomplete_file is
        num_incomplete_files.append(len(incomplete_files))

        if len(num_incomplete_files) > 10:
            if all([i == len(incomplete_files) for i in num_incomplete_files[-10:]]):
                break

        json.dump(completed_files, open(args.output_file, "w"))

    # Combine all the processed files into one
    combined_contents = {}
    combined_contents = json.load(open(args.output_file))

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
