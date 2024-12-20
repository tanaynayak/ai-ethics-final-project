#!/bin/bash

#SBATCH -A ia1               # Set the account name, assuming this is correct
#SBATCH --partition debug                 # Use the 8 GPU partition
#SBATCH --gres=gpu:2             # Request one GPU
#SBATCH --nodes=1                      # Request one node
#SBATCH --ntasks-per-node=1              # Run one task per node
#SBATCH --time=23:00:00                 # Memory per CPU
#SBATCH --job-name="t2i-bias"    # Set the job name
#SBATCH --output=slurm-%j.out            # Standard output and error log
#SBATCH --mail-user=tnayak2@jh.edu
#SBATCH --mail-type=ALL

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
eval "$__conda_setup"
else
if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then . "/opt/anaconda3/etc/profile.d/conda.sh"
else
export PATH="/opt/anaconda3/bin:$PATH"
fi
fi
unset __conda_setup
# <<< conda initialize <<<

source ~/.bashrc
conda activate tanayai2 # activate the Python environment

# runs your code
python3 run_koala_pipeline.py