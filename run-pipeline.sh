#!/bin/bash

#SBATCH --account=park
#SBATCH -c 2
#SBATCH -p priority
#SBATCH -t 2-00:00
#SBATCH --mem 64G
#SBATCH -o logs/%j_pipeline.log
#SBATCH -e logs/%j_pipeline.log

module unload python
module load gcc conda2 slurm-drmaa/1.1.1

# shellcheck source=/dev/null
source "$HOME/.bashrc"
conda activate ppl-comp-smk

snakemake \
    --cores 1 \
    --restart-times 0 \
    --latency-wait 120 \
    --use-conda \
    --keep-going \
    --printshellcmds

conda deactivate
exit 0
