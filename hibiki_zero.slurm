#!/bin/bash
#
#SBATCH -J hibiki_zero
#SBATCH -t 08:00:00
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:1

#pipeline return status is not only dependent on the status of the last command
#https://www.gnu.org/software/bash/manual/bash.html, last visit 20.10.2026
set -euo pipefail

#Removing all current software packages of the environment
module purge
#Adding Python to the running environment
module load python/3.12.8-gcc-12.2.0-4y5tbpr
#Adding FFmpeg to the running environment
module load ffmpeg/4.4.1-gcc-12.2.0-d54gcg6

#Activating pre built Python Virtual Environment
source "$DATA/venvs/hibiki-zero/bin/activate"


#Redirecting all the data that is produced in a model run to the $DATA directory because of full disk space in the other VSC5 directories.
#export makes variables available in the environment
#https://www.gnu.org/software/bash/manual/bash.html
export XDG_CACHE_HOME="$DATA/.cache"
export HF_HOME="$DATA/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$DATA/.cache/torch"

mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME"

# Compiler caches.
COMPILER_CACHE_DIRECTORY="$DATA/.cache/hibiki_zero_compiler"

export TMPDIR="$COMPILER_CACHE_DIRECTORY/tmp/$SLURM_JOB_ID"
export TRITON_CACHE_DIR="$COMPILER_CACHE_DIRECTORY/triton"
export TORCHINDUCTOR_CACHE_DIR="$COMPILER_CACHE_DIRECTORY/torchinductor"
export CUDA_CACHE_PATH="$COMPILER_CACHE_DIRECTORY/cuda"

mkdir -p \
    "$TMPDIR" \
    "$TRITON_CACHE_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" \
    "$CUDA_CACHE_PATH"

# Prints out the environment variables into the slurm-xxxxxx.out file for debugging
echo "Hugging Face cache: $HF_HUB_CACHE"
echo "Compiler cache: $COMPILER_CACHE_DIRECTORY"

PROJECT_DIRECTORY="$DATA/hibiki_zero"

#Task
python -u "$PROJECT_DIRECTORY/hibiki_zero_script.py"
