#!/bin/sh
#SBATCH -o "slurm/eval_gpt/outfile_%A"    # $A for the job id, $a for the task id, ie one of [1 - 100]
#SBATCH -e "slurm/eval_gpt/errfile_%A"    # $A for the job id, $a for the task id, ie one of [1 - 100]
#SBATCH --time 40:00:00
#SBATCH --gres gpu:0
#SBATCH -c 4
#SBATCH --mem=10G

CKPT_NAME=$1
BENCHMARK=$2

################################################### Run GPT for evaluation.
# case $BENCHMARK in
#     'MSVD'|'MSVD_full'|'ActivityNet-QAdev')
#         api_key="b8913a3a9fa34b8c8da2c62185f73e83"
#         ;;
#     'MSRVTT'|'MSRVTT_full')
#         api_key='f713862f4b174ddaaf7a230e34bc2fc3'
#         ;;
#     'TGIF'|'TGIF_full')
#         api_key='46d8daced544461980a81686f6a2d3a5'
#         ;;
#     *)
#         api_key='f713862f4b174ddaaf7a230e34bc2fc3'
#         ;;
# esac

# this code will not run if merged file does not exist. 
output_file=eval_result/${CKPT_NAME}/${BENCHMARK}_pred_merge.jsonl

# python scripts/eval_gpt.py \
#     --pred_path $output_file \
#     --output_file eval_result/${CKPT_NAME}/${BENCHMARK}_gpt.json
