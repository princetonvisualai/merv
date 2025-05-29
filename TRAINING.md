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
Here is a bare-bones shell script one can use to launch reproduction of our method.
```sh
ID="merv-run"

torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain_video.py \
  --run_id $ID \
  --model.model_id $ID \
  --model.type "merv-base" \
  --dataset.type "videollava" \
  --stage finetune 
```

To modify some parameters, feel free to adjust any of the configs outlined in ```merv/conf/models.py```.
For example, to swap out the encoders and adjust the projector token length from 64 to 16, you can run 

```sh
ID="merv-novel"

# Visual_feature_length = Projector_token_length x temporal_resolution (i.e. # of frames after encoding)
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain_video.py \
  --run_id $ID \
  --model.model_id $ID \
  --model.type "merv-base" \
  --model.video_backbone_ids ['languagebind-video-noclass','dinov2-video-all-tokens','hiera-base-plus-video','siglip-vit-b16-224px-all-no-cls'] \
  --model.num_frames [16,16,32,16] \
  --model.projector_token_length 16 \
  --model.visual_feature_length 256 \
  --dataset.type "videollava" \
  --stage finetune 
```

To use different backbone try following:

```sh
ID="merv-run-qwen2.5"

torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain_video.py \
  --run_id $ID \
  --model.model_id $ID \
  --model.finetune_per_device_batch_size 2 \
  --model.llm_backbone_id "qwen2.5-7b-instruct" \
  --model.type "merv-base" \
  --dataset.type "videollava" \
  --stage finetune 
```

```sh
ID="merv-run-llama3.1"

torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain_video.py \
  --run_id $ID \
  --model.model_id $ID \
  --model.llm_backbone_id "llama3.1-8b-chat" \
  --model.finetune_per_device_batch_size 4 \
  --model.type "merv-base" \
  --dataset.type "videollava" \
  --stage finetune 
```