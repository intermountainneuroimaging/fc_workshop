#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 02_download_msc_openneuro.sh
# Download Midnight Scan Club sample data from OpenNeuro.
#
# Default:
#   5 subjects, sub-MSC01 through sub-MSC05
#   dataset-level files
#   T1w anatomy files
#   rest BOLD files
#   JSON sidecars
#   scans.tsv files
#
# Uses public unsigned S3 access, so no AWS account is required.
###############################################################################

export USERNAME="${USERNAME:-jade6100}"
export MSC_ROOT="${MSC_ROOT:-/scratch/alpine/${USERNAME}/openneuro}"
export MSC_BIDS_DIR="${MSC_BIDS_DIR:-${MSC_ROOT}/ds000224}"
export MSC_SUBJECTS="${MSC_SUBJECTS:-sub-MSC01 sub-MSC02 sub-MSC03 sub-MSC04 sub-MSC05}"

export PATH="/scratch/alpine/${USERNAME}/software/bin:${HOME}/.local/bin:${PATH}"

if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: aws was not found."
  echo "Run 01_install_aws_cli.sh first, then run this script again."
  echo "Expected location:"
  echo "/scratch/alpine/${USERNAME}/software/bin/aws"
  exit 1
fi

echo "AWS CLI:"
which aws
aws --version
echo

mkdir -p "${MSC_ROOT}"
cd "${MSC_ROOT}"
mkdir -p ds000224

echo "Testing public OpenNeuro access..."
# Do not let head create a strict-mode broken pipe failure.
set +o pipefail
aws s3 ls --no-sign-request s3://openneuro.org/ds000224/ | head
set -o pipefail
echo

echo "Downloading dataset-level metadata..."
aws s3 sync --no-sign-request s3://openneuro.org/ds000224 ./ds000224 \
  --exclude "*" \
  --include ".bidsignore" \
  --include "dataset_description.json" \
  --include "README" \
  --include "CHANGES" \
  --include "participants.tsv" \
  --include "participants.json" \
  --include "T1w.json" \
  --include "task-rest_bold.json"

echo
echo "Downloading selected MSC subjects..."
for sub in ${MSC_SUBJECTS}; do
  echo
  echo "============================================================"
  echo "Downloading ${sub}"
  echo "============================================================"

  aws s3 sync --no-sign-request s3://openneuro.org/ds000224 ./ds000224 \
    --exclude "*" \
    --include "${sub}/ses-struct*/anat/*_T1w.nii.gz" \
    --include "${sub}/ses-struct*/anat/*_T1w.json" \
    --include "${sub}/ses-struct*/${sub}_ses-struct*_scans.tsv" \
    --include "${sub}/ses-func*/func/*task-rest_bold.nii.gz" \
    --include "${sub}/ses-func*/func/*task-rest_bold.json" \
    --include "${sub}/ses-func*/func/*task-rest_events.tsv" \
    --include "${sub}/ses-func*/${sub}_ses-func*_scans.tsv"
done

echo
echo "Download complete."
echo "BIDS directory: ${MSC_BIDS_DIR}"
echo

echo "Counts:"
echo -n "T1w count: "
find "${MSC_BIDS_DIR}" -name "*_T1w.nii.gz" | sort | wc -l

echo -n "Rest BOLD count: "
find "${MSC_BIDS_DIR}" -name "*task-rest_bold.nii.gz" | sort | wc -l
