#!/bin/bash

# The SBATCH directives must appear before any executable line in this script.

##SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=8            # tasks requested
#SBATCH --cpus-per-task=8      # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:8          # the number of GPUs requested
#SBATCH --mem=400G            # memory
#SBATCH -o slurm/%N_%j_%t.out # send stdout to outfile
#SBATCH -e slurm/%N_%j_%t.err # send stderr to errfile
##SBATCH --nodelist=node028,node029 # specify the nodes you want to run on
##SBATCH --exclude=neu[301,323,328]
#SBATCH -t 3-0:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all # choice between begin, end, all to notify you via email
#SBATCH --mail-user=tz9136@princeton.edu

# You can also just run the script from the directory you want to be in.
# SBATCH -D /home/tyler/scale-mae-detection-xview

# Uncomment to control the output files. By default stdout and stderr go to
# the same place, but if you use both commands below they'll be split up.
# %N is the hostname (if used, will create output(s) per node).
# %j is jobid.

# Print some info for context.
pwd
hostname
date

echo "Starting job..."

source ~/.bashrc
conda activate merv

# Python will buffer output of your script unless you set this.
# If you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed.
export PYTHONUNBUFFERED=1

# Do all the research.
# python scripts/eval_tempcompass.py --task_type $1 --model_name $2
bash $1

# Print completion time.
date
