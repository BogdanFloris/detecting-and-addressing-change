#!/bin/bash
# SBATCH --job-name=conceptdrift
# SBATCH --nodes=1
# SBATCH --time=1:00:00
# SBATCH --ntasks-per-node=1
# SBATCH --mem=16G
# SBATCH --mail-type=END
# SBATCH --mail-user=bogdan.floris@gmail.com

module purge
module load 2019
module load Python/3.7.5-foss-2018b

cp -r "$HOME"/detecting-and-addressing-change/ "$TMPDIR"/
cd "$TMPDIR"/detecting-and-addressing-change/ || return

RUN_NAME="$(date +%d_%m_%Y_%H_%M_%S)";
python3 run.py >> "output_${RUN_NAME}.txt"

rsync -a ./ "$HOME"/detecting-and-addressing-change/