#!/bin/bash
#SBATCH --job-name=conceptdrift
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_short
#SBATCH --mem=16G
#SBATCH --mail-type=END
#SBATCH --mail-user=bogdan.floris@gmail.com

module purge
module load 2019
module load CUDA/10.1.243
module load Python/3.7.5-foss-2018b

cp -r "$HOME"/detecting-and-addressing-change/ "$TMPDIR"
# shellcheck disable=SC2164
cd "$TMPDIR"/detecting-and-addressing-change/

RUN_NAME="$(date +%d_%m_%Y_%H_%M_%S)";
python3 run.py >> "output_${RUN_NAME}.txt"

rsync -a ./detecting-and-addresing-change/ "$HOME"/detecting-and-addressing-change/
