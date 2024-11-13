ID="hiera_merv-noft_single"

# WANDB_MODE=disabled \
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain_video.py \
  --run_id $ID \
  --model.model_id $ID \
  --model.type "merv-base" \
  --model.video_backbone_ids ["hiera-base-plus-video-noft"] \
  --model.num_frames [32] \
  --dataset.type "videollava" \
  --stage finetune 