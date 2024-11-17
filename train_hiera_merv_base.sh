ID="hiera_merv_base"

# WANDB_MODE=disabled \
# torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain_video.py \
#   --run_id $ID \
#   --model.model_id $ID \
#   --model.type "merv-base" \
#   --model.video_backbone_ids ['languagebind-video-noclass','dinov2-video-all-tokens',"hiera-base-plus-video",'siglip-vit-b16-224px-all-no-cls'] \
#   --model.num_frames [16,16,32,16] \
#   --model.visual_feature_length 1024 \
#   --dataset.type "videollava" \
#   --stage finetune 

sbatch -J ${ID}_MSRVTT_full --array "[0-3]" --gres=gpu:1 eval_1gpu_multi.sh $ID MSRVTT_full 4
sbatch -J ${ID}_MSVD_full --array "[0-3]" --gres=gpu:1 eval_1gpu_multi.sh $ID MSVD_full 4
sbatch -J ${ID}_TGIF_full --array "[0-2]" --gres=gpu:1 eval_1gpu_multi.sh $ID TGIF_full 3
sbatch --job-name="Perception_naive_${ID}" --gres=gpu:1 --array "[0-3]" eval_mcq.sh ${ID} "Perception" 4 "test_q" "test_a" "runs/${ID}"
sbatch -J ${ID}_ActivityNet-QAdev --array "[0-7]" --gres=gpu:1 eval_1gpu_multi.sh $ID ActivityNet-QAdev 8
