#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 03_verify_msc_download.sh
# Verify that MSC files downloaded correctly and are readable by FSL.
###############################################################################

export USERNAME="${USERNAME:-jade6100}"
export MSC_BIDS_DIR="${MSC_BIDS_DIR:-/scratch/alpine/${USERNAME}/openneuro/ds000224}"
export FSLDIR="${FSLDIR:-/projects/ics/software/fsl/6.0.7}"

export PATH="${FSLDIR}/bin:${FSLDIR}/share/fsl/bin:${PATH}"
source "${FSLDIR}/etc/fslconf/fsl.sh" 2>/dev/null || true

if ! command -v fslinfo >/dev/null 2>&1; then
  echo "ERROR: fslinfo was not found."
  echo "FSLDIR=${FSLDIR}"
  echo "Check this path:"
  echo "${FSLDIR}/share/fsl/bin/fslinfo"
  exit 1
fi

if [[ ! -d "${MSC_BIDS_DIR}" ]]; then
  echo "ERROR: MSC_BIDS_DIR does not exist:"
  echo "${MSC_BIDS_DIR}"
  echo "Run 02_download_msc_openneuro.sh first."
  exit 1
fi

cd "${MSC_BIDS_DIR}"

echo "Current folder:"
pwd
echo

echo "Dataset-level files:"
ls -lah dataset_description.json participants.tsv task-rest_bold.json 2>/dev/null || true
echo

echo "T1w count:"
find . -name "*_T1w.nii.gz" | sort | tee /tmp/msc_t1w_files.txt | wc -l
echo

echo "Rest BOLD count:"
find . -name "*task-rest_bold.nii.gz" | sort | tee /tmp/msc_rest_bold_files.txt | wc -l
echo

n_t1w=$(wc -l < /tmp/msc_t1w_files.txt)
n_bold=$(wc -l < /tmp/msc_rest_bold_files.txt)

if [[ "${n_t1w}" -eq 0 ]]; then
  echo "ERROR: No T1w files found."
  echo "Run the download script first."
  exit 1
fi

if [[ "${n_bold}" -eq 0 ]]; then
  echo "ERROR: No rest BOLD files found."
  echo "Run the download script first."
  exit 1
fi

echo "First T1w files:"
head -20 /tmp/msc_t1w_files.txt
echo

echo "First rest BOLD files:"
head -20 /tmp/msc_rest_bold_files.txt
echo

echo "Inspect first T1w with fslinfo:"
first_t1w=$(head -1 /tmp/msc_t1w_files.txt)
fslinfo "${first_t1w}"
echo

echo "Inspect first rest BOLD with fslinfo:"
first_bold=$(head -1 /tmp/msc_rest_bold_files.txt)
fslinfo "${first_bold}"
echo

echo "BOLD volume count:"
fslnvols "${first_bold}"
echo

echo "Optional input audit:"
echo "T2w files:"
find . -iname "*T2w*.nii.gz" | sort | head -50
echo

echo "Fieldmap-like files:"
find . \( -path "*/fmap/*" -o -iname "*fieldmap*" -o -iname "*magnitude*" -o -iname "*phasediff*" -o -iname "*phase*" \) | sort | head -100
echo

echo "SBRef or spin echo files:"
find . \( -iname "*sbref*" -o -iname "*SpinEcho*" -o -iname "*se*" \) | sort | head -100
echo

echo "Verification complete."
