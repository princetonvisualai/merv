#!/bin/sh
#SBATCH -o "slurm/eval/outfile_%A_%a"    # $A for the job id, $a for the task id, ie one of [1 - 100]
#SBATCH -e "slurm/eval/errfile_%A_%a"    # $A for the job id, $a for the task id, ie one of [1 - 100]
#SBATCH --time 40:00:00
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem=60G


#### ex) sbatch -J jobname --array [0-7] eval_1gpu_multi.sh query4_proj16_multi MSVDsmall 8

# if there is error in one of the code below, the whole script stops.
set -e

CKPT_NAME=$1
BENCHMARK=$2
CHUNKS=$3

source ~/.bashrc
conda activate merv

# In parallel, run evaluation jobs.
python scripts/eval_openended.py --model_path ${CKPT_NAME} --eval_dataset ${BENCHMARK} \
      --num_chunks $CHUNKS \
      --chunk_idx $SLURM_ARRAY_TASK_ID
# when done, this will generate a file with _done appended.
# if number of done is same as number of chunks, it will merge into single file.

sbatch eval_1gpu_multi2.sh ${CKPT_NAME} $BENCHMARK
