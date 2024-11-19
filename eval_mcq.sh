#!/bin/sh
#SBATCH -o "slurm/eval/outfile_%A_%a"    # $A for the job id, $a for the task id, ie one of [1 - 100]
#SBATCH -e "slurm/eval/errfile_%A_%a"    # $A for the job id, $a for the task id, ie one of [1 - 100]
#SBATCH --time 40:00:00
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#### ex) sbatch --array "[0-7]" eval_perception.sh "videoonlyllava-clip-dinovid" "Perception" 8

CKPT_NAME=$1
BENCHMARK=$2
CHUNKS=$3

source ~/.bashrc
conda activate merv


HF_HOME=/u/jc5933/.cache/huggingface \
python scripts/eval_mcq.py --model_path ${CKPT_NAME} --eval_dataset ${BENCHMARK} \
      --num_chunks $CHUNKS \
      --chunk_idx $SLURM_ARRAY_TASK_ID