#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 07_full_hcp_own_data_template.sh
# Template for running the official HCP scripts on your own data.
#
# Edit the file paths under "EDIT THESE FOR YOUR OWN DATA" before running.
###############################################################################

export USERNAME="${USERNAME:-jade6100}"

export FSLDIR="${FSLDIR:-/projects/ics/software/fsl/6.0.7}"
export FREESURFER_HOME="${FREESURFER_HOME:-/projects/ics/software/freesurfer/6.0.1}"
export FS_LICENSE="${FS_LICENSE:-${FREESURFER_HOME}/license.txt}"
export HCPPIPEDIR="${HCPPIPEDIR:-/projects/ics/software/hcp_pipeline/HCP_Pipelines-3.4.0}"
export CARET7DIR="${CARET7DIR:-/projects/ics/software/hcp_pipeline/caret_workbench/bin_rh_linux64}"
export PATH="${HCPPIPEDIR}/global/scripts:${HCPPIPEDIR}/PreFreeSurfer:${HCPPIPEDIR}/FreeSurfer:${HCPPIPEDIR}/PostFreeSurfer:${HCPPIPEDIR}/fMRIVolume:${HCPPIPEDIR}/fMRISurface:${CARET7DIR}:${FSLDIR}/bin:${FSLDIR}/share/fsl/bin:${FREESURFER_HOME}/bin:${PATH}"

source "${FSLDIR}/etc/fslconf/fsl.sh"

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
  exit 1
fi

###############################################################################
# EDIT THESE FOR YOUR OWN DATA
###############################################################################

export StudyFolder="${StudyFolder:-/scratch/alpine/${USERNAME}/my_hcp_project/hcp_outputs}"
export Subject="${Subject:-sub-01}"

export T1wInputImage="${T1wInputImage:-/path/to/sub-01_T1w.nii.gz}"
export T2wInputImage="${T2wInputImage:-/path/to/sub-01_T2w.nii.gz}"

export fMRIName="${fMRIName:-rfMRI_REST01_LR}"
export fMRIInput="${fMRIInput:-/path/to/sub-01_task-rest_bold.nii.gz}"
export SBRefInput="${SBRefInput:-/path/to/sub-01_task-rest_sbref.nii.gz}"

export SpinEchoPhaseEncodeNegative="${SpinEchoPhaseEncodeNegative:-/path/to/spin_echo_negative.nii.gz}"
export SpinEchoPhaseEncodePositive="${SpinEchoPhaseEncodePositive:-/path/to/spin_echo_positive.nii.gz}"

export PhaseEncodingDir="${PhaseEncodingDir:-y}"
export DwellTime="${DwellTime:-0.00058}"
export DistortionCorrection="${DistortionCorrection:-TOPUP}"
export GradientDistortionCoeffs="${GradientDistortionCoeffs:-NONE}"
export TopUpConfig="${TopUpConfig:-${HCPPIPEDIR}/global/config/b02b0.cnf}"

export FinalFMRIResolution="${FinalFMRIResolution:-2}"
export LowResMesh="${LowResMesh:-32}"
export GrayordinatesResolution="${GrayordinatesResolution:-2}"

mkdir -p "${StudyFolder}/${Subject}/unprocessed/3T/T1w_MPR1"
mkdir -p "${StudyFolder}/${Subject}/unprocessed/3T/T2w_SPC1"
mkdir -p "${StudyFolder}/${Subject}/unprocessed/3T/${fMRIName}"

cp -f "${T1wInputImage}" "${StudyFolder}/${Subject}/unprocessed/3T/T1w_MPR1/${Subject}_3T_T1w_MPR1.nii.gz"
cp -f "${T2wInputImage}" "${StudyFolder}/${Subject}/unprocessed/3T/T2w_SPC1/${Subject}_3T_T2w_SPC1.nii.gz"
cp -f "${fMRIInput}" "${StudyFolder}/${Subject}/unprocessed/3T/${fMRIName}/${Subject}_3T_${fMRIName}.nii.gz"
cp -f "${SBRefInput}" "${StudyFolder}/${Subject}/unprocessed/3T/${fMRIName}/${Subject}_3T_${fMRIName}_SBRef.nii.gz"
cp -f "${SpinEchoPhaseEncodeNegative}" "${StudyFolder}/${Subject}/unprocessed/3T/${fMRIName}/${Subject}_3T_SpinEchoFieldMap_LR.nii.gz"
cp -f "${SpinEchoPhaseEncodePositive}" "${StudyFolder}/${Subject}/unprocessed/3T/${fMRIName}/${Subject}_3T_SpinEchoFieldMap_RL.nii.gz"

echo "Running HCP PreFreeSurfer..."
"${HCPPIPEDIR}/PreFreeSurfer/PreFreeSurferPipeline.sh" \
  --path="${StudyFolder}" \
  --subject="${Subject}" \
  --t1="${StudyFolder}/${Subject}/unprocessed/3T/T1w_MPR1/${Subject}_3T_T1w_MPR1.nii.gz" \
  --t2="${StudyFolder}/${Subject}/unprocessed/3T/T2w_SPC1/${Subject}_3T_T2w_SPC1.nii.gz" \
  --t1template="${FSLDIR}/data/standard/MNI152_T1_0.7mm.nii.gz" \
  --t1templatebrain="${FSLDIR}/data/standard/MNI152_T1_0.7mm_brain.nii.gz" \
  --t1template2mm="${FSLDIR}/data/standard/MNI152_T1_2mm.nii.gz" \
  --t2template="${FSLDIR}/data/standard/MNI152_T1_0.7mm.nii.gz" \
  --t2templatebrain="${FSLDIR}/data/standard/MNI152_T1_0.7mm_brain.nii.gz" \
  --t2template2mm="${FSLDIR}/data/standard/MNI152_T1_2mm.nii.gz" \
  --templatemask="${FSLDIR}/data/standard/MNI152_T1_0.7mm_brain_mask.nii.gz" \
  --template2mmmask="${FSLDIR}/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz" \
  --brainsize="150" \
  --fnirtconfig="${HCPPIPEDIR}/global/config/T1_2_MNI152_2mm.cnf" \
  --fmapmag="NONE" \
  --fmapphase="NONE" \
  --echospacing="NONE" \
  --seunwarpdir="NONE" \
  --gdcoeffs="${GradientDistortionCoeffs}"

echo "Running HCP FreeSurfer..."
"${HCPPIPEDIR}/FreeSurfer/FreeSurferPipeline.sh" \
  --subject="${Subject}" \
  --subjectDIR="${StudyFolder}/${Subject}/T1w" \
  --t1="${StudyFolder}/${Subject}/T1w/T1w_acpc_dc_restore.nii.gz" \
  --t1brain="${StudyFolder}/${Subject}/T1w/T1w_acpc_dc_restore_brain.nii.gz" \
  --t2="${StudyFolder}/${Subject}/T1w/T2w_acpc_dc_restore.nii.gz"

echo "Running HCP PostFreeSurfer..."
"${HCPPIPEDIR}/PostFreeSurfer/PostFreeSurferPipeline.sh" \
  --path="${StudyFolder}" \
  --subject="${Subject}" \
  --surfatlasdir="${HCPPIPEDIR}/global/templates/standard_mesh_atlases" \
  --grayordinatesdir="${HCPPIPEDIR}/global/templates/91282_Greyordinates" \
  --grayordinatesres="${GrayordinatesResolution}" \
  --hiresmesh="164" \
  --lowresmesh="${LowResMesh}" \
  --subcortgraylabels="${HCPPIPEDIR}/global/config/FreeSurferSubcorticalLabelTableLut.txt" \
  --freesurferlabels="${HCPPIPEDIR}/global/config/FreeSurferAllLut.txt" \
  --refmyelinmaps="NONE"

echo "Running HCP fMRI volume processing..."
"${HCPPIPEDIR}/fMRIVolume/GenericfMRIVolumeProcessingPipeline.sh" \
  --path="${StudyFolder}" \
  --subject="${Subject}" \
  --fmriname="${fMRIName}" \
  --fmritcs="${StudyFolder}/${Subject}/unprocessed/3T/${fMRIName}/${Subject}_3T_${fMRIName}.nii.gz" \
  --fmriscout="${StudyFolder}/${Subject}/unprocessed/3T/${fMRIName}/${Subject}_3T_${fMRIName}_SBRef.nii.gz" \
  --SEPhaseNeg="${StudyFolder}/${Subject}/unprocessed/3T/${fMRIName}/${Subject}_3T_SpinEchoFieldMap_LR.nii.gz" \
  --SEPhasePos="${StudyFolder}/${Subject}/unprocessed/3T/${fMRIName}/${Subject}_3T_SpinEchoFieldMap_RL.nii.gz" \
  --echospacing="${DwellTime}" \
  --unwarpdir="${PhaseEncodingDir}" \
  --fmrires="${FinalFMRIResolution}" \
  --dcmethod="${DistortionCorrection}" \
  --gdcoeffs="${GradientDistortionCoeffs}" \
  --topupconfig="${TopUpConfig}"

echo "Full HCP template finished."
echo "Check outputs under:"
echo "${StudyFolder}/${Subject}/MNINonLinear/Results/${fMRIName}"
