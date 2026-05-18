#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 04_setup_hcp_fsl_env.sh
# Load FSL, FreeSurfer, HCP Pipelines, and Workbench.
###############################################################################

export USERNAME="${USERNAME:-jade6100}"

export FSLDIR="${FSLDIR:-/projects/ics/software/fsl/6.0.7}"
export FREESURFER_HOME="${FREESURFER_HOME:-/projects/ics/software/freesurfer/6.0.1}"
export FS_LICENSE="${FS_LICENSE:-${FREESURFER_HOME}/license.txt}"
export HCPPIPEDIR="${HCPPIPEDIR:-/projects/ics/software/hcp_pipeline/HCP_Pipelines-3.4.0}"
export CARET7DIR="${CARET7DIR:-/projects/ics/software/hcp_pipeline/caret_workbench/bin_rh_linux64}"

export PATH="${HCPPIPEDIR}/global/scripts:${HCPPIPEDIR}/PreFreeSurfer:${HCPPIPEDIR}/FreeSurfer:${HCPPIPEDIR}/PostFreeSurfer:${HCPPIPEDIR}/fMRIVolume:${HCPPIPEDIR}/fMRISurface:${CARET7DIR}:${FSLDIR}/bin:${FSLDIR}/share/fsl/bin:${FREESURFER_HOME}/bin:${PATH}"

source "${FSLDIR}/etc/fslconf/fsl.sh"

# FreeSurfer 6 setup can fail under strict bash settings because of internal
# grep checks. Temporarily relax strict mode while sourcing it.
export FS_FREESURFERENV_NO_OUTPUT=1
set +e
set +u
set +o pipefail
source "${FREESURFER_HOME}/SetUpFreeSurfer.sh"
FS_SETUP_STATUS=$?
set -e
set -u
set -o pipefail

if [[ "${FS_SETUP_STATUS}" -ne 0 ]]; then
  echo "ERROR: FreeSurfer setup failed."
  echo "FREESURFER_HOME=${FREESURFER_HOME}"
  exit 1
fi

echo "Environment loaded."
echo
echo "FSL:"
which fslinfo
which fslmaths
which mcflirt
which flirt
which bet
which robustfov
echo
echo "FreeSurfer:"
which recon-all
echo
echo "HCP Pipeline scripts:"
find "${HCPPIPEDIR}" -type f \( \
  -name "PreFreeSurferPipeline.sh" -o \
  -name "FreeSurferPipeline.sh" -o \
  -name "PostFreeSurferPipeline.sh" -o \
  -name "GenericfMRIVolumeProcessingPipeline.sh" -o \
  -name "GenericfMRISurfaceProcessingPipeline.sh" \
  \) | sort
echo
echo "Workbench:"
command -v wb_command || true
