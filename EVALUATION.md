# Evaluation

We evaluate on a diverse set of tasks.
* MSVD, MSRVTT, TGIF, and ActivityNet preparation follow that of [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/TRAIN_AND_VALIDATE.md).
* Perception Test can be found [here](https://github.com/google-deepmind/perception_test).
* NExT-QA, VLEP, TVQA preparation follow that of [SeViLA](https://github.com/Yui010206/SeViLA?tab=readme-ov-file).

To help you understand the evaluation data structure, please take a look at [dummy_mcq](eval_data/dummy_mcq) and [dummy_openended](eval_data/dummy_openended).

## Open-ended Evaluation

We follow the Video-ChatGPT protocol for evaluation, but our prompts are the same as Video-LLaVA for consistency in comparison.
Note that the API model is always subject to change; we used ``gpt-3.5-turbo-0613`` for GPT evaluation on our paper, but this is no longer available through OpenAI.
Instead, we provide API usage of ``gpt-4o-mini-2024-07-18`` as per OpenAI's recommendation.


```sh
# In parallel, run inference jobs. E.g, for parallel of 4
CKPT_NAME=merv-full
BENCHMARK=dummy_openended
python scripts/eval_openended.py --model_path ${CKPT_NAME} --eval_dataset ${BENCHMARK} \
      --num_chunks 4 \
      --chunk_idx 0 # 0,1,2, or 3

# ... wait for all jobs to finish ...

# Then run GPT on the results; API keys taken from .openai_key
python scripts/eval_gpt_batch.py \
    --ckpt_name ${CKPT_NAME} \
    --benchmark ${BENCHMARK}
```

## MCQ Evaluation

For MCQ based tasks, we use the following script:

```sh
CKPT_NAME=merv-full
BENCHMARK=dummy_mcq
python scripts/eval_mcq.py --model_path ${CKPT_NAME} --eval_dataset ${BENCHMARK} \
      --num_chunks 1 \
      --chunk_idx 0
```
