#!/bin/sh
#SBATCH -o "slurm/eval/outfile_%A_%a"    # $A for the job id, $a for the task id, ie one of [1 - 100]
#SBATCH -e "slurm/eval/errfile_%A_%a"    # $A for the job id, $a for the task id, ie one of [1 - 100]
#SBATCH --time 40:00:00
#SBATCH -c 8
#SBATCH --mem=60G
set -e

#### ex) sbatch --array "[0-7]" eval_perception.sh "videoonlyllava-clip-dinovid" "Perception" 8
#### ex) sbatch --array "[1-7]" eval_perception.sh "lb-llama3" "Perception" 8

# if there is error in one of the code below, the whole script stops.
handle_error() {
    echo "An error occurred on line $1"
    exit 1
}

CKPT_NAME=$1
BENCHMARK=$2
FILENAMEQUESTION=$4
FILENAMEANSWER=$5
FULLPATH=$6
STRATEGY=naive

source ~/.bashrc
conda activate merv


python scripts/eval_mcq.py --model_path ${CKPT_NAME} --eval_dataset ${BENCHMARK} \
      --num_chunks $3 \
      --chunk_idx $SLURM_ARRAY_TASK_ID \
      --filename_question ${FILENAMEQUESTION}\
      --filename_answer ${FILENAMEANSWER} \
      --full_path_ckpt ${FULLPATH} \
      --strategy ${STRATEGY}

# No need for GPT thing. its included in eval_mcq now.
