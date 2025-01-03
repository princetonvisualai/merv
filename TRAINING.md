# Training

## Data Preparation
Our training data mirrors that of Video-LLaVA, so follow their instructions from [here](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/TRAIN_AND_VALIDATE.md) to download the data.

After Downloading them, the data structure should be as follows:

```Shell
data/download/videollava
├── valley_llavaimage.json
├── videochatgpt_llavaimage_tune.json
├── valley
│   ├── 000001_000050
│   └── ...
├── llava_image_tune
│   ├── coco
│   ├── gqa
│   ├── ocr_vqa
│   ├── textvqa
│   └── vg
└── videochatgpt_tune
    ├── v_---9CpRcKoU.mp4
    └── ...
```

## Example Training Script
```sh
ID="merv-run"

torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain_video.py \
  --run_id $ID \
  --model.model_id $ID \
  --model.type "merv-base" \
  --dataset.type "videollava" \
  --stage finetune 
```

TODO: add more settings.
