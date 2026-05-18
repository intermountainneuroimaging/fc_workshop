#!/bin/bash
#SBATCH --job-name=msc_fc_ready
#SBATCH --output=/scratch/alpine/jade6100/openneuro/hcp_scripts/logs/msc_fc_ready_%j.out
#SBATCH --error=/scratch/alpine/jade6100/openneuro/hcp_scripts/logs/msc_fc_ready_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=normal
#SBATCH --partition=amilan

set -euo pipefail

export USERNAME="jade6100"
export RAW_BIDS="/scratch/alpine/${USERNAME}/openneuro/ds000224"
export STUDY_DIR="/scratch/alpine/${USERNAME}/openneuro/hcp_outputs"
export FC_DIR="/scratch/alpine/${USERNAME}/openneuro/fc_ready"

export SUBJECTS="MSC01 MSC02 MSC03 MSC04 MSC05"
export FUNC_SESSIONS="ses-func01"

export DROP_TRS="5"
export TR="2.2"

mkdir -p /scratch/alpine/${USERNAME}/openneuro/hcp_scripts/logs

bash /scratch/alpine/${USERNAME}/openneuro/hcp_scripts/05_msc_hcp_style_to_fc_ready.sh
