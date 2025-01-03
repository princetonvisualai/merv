import argparse
import ast
import json
import os
import time
import openai
from tqdm import tqdm
import io

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-4o-mini")
    parser.add_argument("--ckpt_name", default=r"", help="Name of checkpoint.")
    parser.add_argument("--benchmark", default=r"", help="Benchmark.")
    args = parser.parse_args()
    return args

def query_gpt(client, prediction_set, caption_files, gpt_batch_batch_id, ckpt_name, benchmark):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """


    openai_modelname = "gpt-4o-mini-2024-07-18"

    jsonl = []
    for key in tqdm(caption_files, desc='Generating JSONL'):
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

        jsonl.append(json.dumps(payload))

    #jsonl cannot have more than 50K items, and 200M in size
    batchfile = []
    size = 0
    ind = 0
    batch_input_files = []
    for l in tqdm(jsonl, desc='Uploading to OpenAI'):
        this_size = len(l)/1024/1024
        if size  + this_size > 190 or len(batchfile) > 49000:
            fileio = io.BytesIO("\n".join(batchfile).encode('utf-8'))
            fileio.name = '{}_{}_{}.jsonl'.format(ckpt_name, benchmark, ind)
            batch_input_file = client.files.create(
                file=fileio,
                purpose="batch"
            )
            batch_input_files.append(batch_input_file)

            batchfile = [l]
            size = this_size
            ind += 1
        else:
            batchfile += [l]
            size += this_size
    
    if len(batchfile) > 0:
        
        fileio = io.BytesIO("\n".join(batchfile).encode('utf-8'))
        fileio.name = '{}_{}_{}.jsonl'.format(ckpt_name, benchmark, ind)
        batch_input_file = client.files.create(
            file=fileio,
            purpose="batch"
        )
        batch_input_files.append(batch_input_file)
        ind+=1

    print(f'Uploaded {ind} files to OpenAI')

    # we should check if file status is 'precessed' but its now deprecated... hmm
    submitted_batchs = []
    for batch_input_file in batch_input_files:
        submitted_batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "nightly eval job"
            }
        )
        submitted_batchs.append(submitted_batch)

    with open(gpt_batch_batch_id, 'w') as f:
        for submitted_batch in submitted_batchs:
            f.write(json.dumps(submitted_batch.json())+"\n")

    return submitted_batchs
        
def save_evaluation_result(jsonl, prediction_set, gpt_batch_output_path):

    gpt_outputs = {}
    for line in jsonl.split("\n"):
        if len(line) == 0:
            continue

        linej = json.loads(line)

        try:
            prediction = ast.literal_eval(linej['response']['body']['choices'][0]['message']['content'])
            assert 'pred' in prediction and 'score' in prediction
        except:
            prediction = {'pred': 'no', 'score': 0}
        gpt_outputs[linej['custom_id']] = prediction

    with open(gpt_batch_output_path, "w") as f:
        gpt_outputs_with_missing = {}
        for k, v in prediction_set.items():
            if k in gpt_outputs:
                gpt_outputs_with_missing[k] = ([gpt_outputs[k], v])
            else:
                gpt_outputs_with_missing[k] = ([{'pred': 'no', 'score': 0}, v])

        json.dump(gpt_outputs_with_missing, f)
   
def print_calculated_performance(gpt_batch_output_path):

    with open(gpt_batch_output_path) as f:
        gpt_outputs = json.load(f)

    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0

    for k, v in gpt_outputs.items():
        count += 1
        score_sum += v[0]['score']
        if 'yes' in v[0]['pred'].lower():
            yes_count += 1
        else:
            no_count += 1

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)
    print("Average score:", average_score)

    return accuracy, average_score

def print_token_usage(jsonl):
    """
    Prints token usage of the OpenAI API
    """
    prompt_tokens = 0
    completion_tokens = 0
    for line in jsonl.split("\n"):

        if len(line) == 0:
            continue

        linej = json.loads(line)

        prompt_tokens += linej['response']['body']['usage']['prompt_tokens']
        completion_tokens += linej['response']['body']['usage']['completion_tokens']

    print(f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")
    if linej['response']['body']['model'] == 'gpt-4o-mini-2024-07-18':
        print("Total Cost on {}".format(linej['response']['body']['model']))
        print('${:.10f}'.format(0.075*prompt_tokens/1000000 + 0.3 * completion_tokens/1000000))
    else:
        print('IDK cost for this model')
    print(f"")

def main():
    args = parse_args()

    pred_path = f"eval_result/{args.ckpt_name}/{args.benchmark}_pred_merge.jsonl"
    gpt_batch_batch_id = f"eval_result/{args.ckpt_name}/{args.benchmark}_gpt_batch_id.jsonl"
    gpt_batch_return_path = f"eval_result/{args.ckpt_name}/{args.benchmark}_gpt_return.jsonl"
    gpt_batch_output_path = f"eval_result/{args.ckpt_name}/{args.benchmark}_gpt.json"

    if os.path.exists(gpt_batch_output_path):
        print('File', gpt_batch_output_path, 'already exists. Evaluation was already done!')
        print_calculated_performance(gpt_batch_output_path)
        exit()

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

    
    if not os.path.exists(gpt_batch_batch_id):
        submitted_batchs = query_gpt(client, prediction_set, id_list, gpt_batch_batch_id, args.ckpt_name, args.benchmark)
        print('We submitted the batch job. Please check back later.')
    else:
        print('Batch job already submitted. Checking if finetuning is finished.')

    with open(gpt_batch_batch_id) as f:
        submitted_batch_ids = []
        with open(gpt_batch_batch_id, 'r') as f:

            for line in f.readlines():
                gpt_batch_id = json.loads(line.strip())
                submitted_batch_ids.append(json.loads(gpt_batch_id)['id'])


    print('Requesting OpenAI server every 60 seconds...')
    bar = tqdm(total=10)
    while True:
        batches_retrieved = [client.batches.retrieve(submitted_batch_id) for submitted_batch_id in submitted_batch_ids]
        batches_status = [batch_retrieved.status for batch_retrieved in batches_retrieved]
        bar.desc = f"Batches status: {batches_status}"
        bar.refresh()
        if all([batch_status == "completed" for batch_status in batches_status]):
            break 
        elif any([batch_status in ["in_progress", "validating", "finalizing"] for batch_status in batches_status]):
            

            total = sum([batch_retrieved.request_counts.total for batch_retrieved in batches_retrieved 
                            if batch_retrieved.status in ["in_progress", 'finalizing']])
            done = sum([batch_retrieved.request_counts.completed for batch_retrieved in batches_retrieved
                            if batch_retrieved.status in ["in_progress", 'finalizing']])
            bar.total = total
            bar.desc = f"Batches status: {batches_status}"
            bar.refresh()
            bar.update(done-bar.n)
            
            time.sleep(60)
        else:
            print("Batch failed. They have to be either completed, in_progress, validating, or finalizing")
            exit()
    
    file_responses = "\n".join([client.files.content(batch_retrieved.output_file_id).text for batch_retrieved in batches_retrieved])
    file_responses = "\n".join([l for l in file_responses.split("\n") if len(l)!=0])
    with open(gpt_batch_return_path, "w") as f:
        f.write(file_responses)

    print(gpt_batch_return_path)
    print_token_usage(file_responses)
    save_evaluation_result(file_responses, prediction_set, gpt_batch_output_path)
    print_calculated_performance(gpt_batch_output_path)

if __name__ == "__main__":
    main()