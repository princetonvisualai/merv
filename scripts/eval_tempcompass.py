import torch
import argparse, json, os
from tqdm import tqdm
from pathlib import Path

from merv import load_vid
from merv.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

def inference_single_video(video_path, inp, vidlm, num_frames=[16,16,32,16]):
    prompt_builder = vidlm.get_prompt_builder()
    prompt_builder.add_turn(role="human", message=inp)
    prompt_text = prompt_builder.get_prompt()

    # video_tensor = video_processor(video_path, return_tensors='pt')['pixel_values']
    # if type(video_tensor) is list:
    #     tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    # else:
    #     tensor = video_tensor.to(model.device, dtype=torch.float16)
    # key = ['video']

    # print(f"{roles[1]}: {inp}")
    # inp = DEFAULT_X_TOKEN['VIDEO'] + '\n' + inp
    # conv.append_message(conv.roles[0], inp)
    # conv.append_message(conv.roles[1], None)
    # prompt = conv.get_prompt()
    # input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0).cuda()
    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # keywords = [stop_str]
    # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        outputs = vidlm.generate(
            video_path,
            prompt_text,
            num_frames=num_frames,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=128,
            use_cache=True
        )
            # stopping_criteria=[stopping_criteria])

    # outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace('</s>', '')
    return outputs

answer_prompt = {
    # "multi-choice": "\nBest Option:",     # The old version
    "multi-choice": "\nPlease directly give the best option:",
    "yes_no": "\nPlease answer yes or no:",
    # "caption_matching": "\nBest Option:",     #The old version
    "caption_matching": "\nPlease directly give the best option:",
    "captioning": ""    # The answer "Generated Caption:" is already contained in the question
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--input_path', default='/n/fs/tz-ego4d/projectB/TempCompass')     
    parser.add_argument('--video_path', default='videos')     
    parser.add_argument('--output_path', default='eval_results')     
    parser.add_argument('--task_type', default='multi-choice', choices=['multi-choice', 'captioning', 'caption_matching', 'yes_no'])     
    parser.add_argument('--model_name', default='merv-full')
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path) / args.model_name
    hf_token = Path(".hf_token").read_text().strip()

    # Loading questions
    question_path = input_path / f"questions/{args.task_type}.json"
    with open(question_path, 'r') as f:
        input_datas = json.load(f)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    pred_file = output_path / f"{args.task_type}.json"
    # Loading existing predictions
    if os.path.isfile(pred_file):
        with open(pred_file, 'r') as f:
            predictions = json.load(f)
    else:
        predictions = {}

    # Loading Video-LLaVA
    # model_path = 'LanguageBind/Video-LLaVA-7B'
    # model_path = f'merv/{args.model_name}'
    device = 'cuda'
    vidlm = load_vid(args.model_name, hf_token=hf_token)
    vidlm.to(device, dtype=torch.bfloat16)
    # load_4bit, load_8bit = True, False
    # model_name = get_model_name_from_path(model_path)
    # tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)
    num_frames = {
        "merv-base": [16, 16, 32, 16],
        "merv-full": [16, 16, 32, 16],
        "languagebind-single": [16],
        "dinov2-single": [16],
        "vivit-single": [32],
        "siglip-single": [16],
    }
    nf = num_frames[args.model_name]

    for vid, data in tqdm(input_datas.items()):
        if vid not in predictions:
            predictions[vid] = {}
            video_path = input_path / args.video_path / f'{vid}.mp4'
            for dim, questions in data.items():
                predictions[vid][dim] = []
                for question in questions:
                    inp = question['question'] + answer_prompt[args.task_type]
                    video_llm_pred = inference_single_video(str(video_path), inp, vidlm, num_frames=nf)
                    predictions[vid][dim].append({'question': question['question'], 'answer': question['answer'], 'prediction': video_llm_pred})
            with open(pred_file, 'w') as f:
                json.dump(predictions, f, indent=4)