ID="hiera_merv_base_five"

WANDB_MODE=disabled \
torchrun --standalone --nnodes 1 --nproc-per-node 4 scripts/pretrain_video.py \
  --run_id $ID \
  --model.model_id $ID \
  --model.type "merv-base" \
  --model.video_backbone_ids ['languagebind-video-noclass','dinov2-video-all-tokens',"vivit-google-b-all-no-cls-16frames",'hiera-base-plus-video','siglip-vit-b16-224px-all-no-cls'] \
  --model.num_frames [16,16,32,32,16] \
  --dataset.type "videollava" \
  --stage finetune 