ID="hiera_merv_base"

# WANDB_MODE=disabled \
torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/pretrain_video.py \
  --run_id $ID \
  --model.model_id $ID \
  --model.type "merv-base" \
  --model.video_backbone_ids ['languagebind-video-noclass','dinov2-video-all-tokens',"hiera-base-plus-video",'siglip-vit-b16-224px-all-no-cls'] \
  --model.num_frames [16,16,32,16] \
  --model.visual_feature_length 1024 \
  --dataset.type "videollava" \
  --stage finetune 