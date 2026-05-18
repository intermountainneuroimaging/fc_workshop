#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 05_msc_hcp_style_to_fc_ready.sh
# Prepare MSC resting-state data for volume-based functional connectivity.
#
# Output:
#   ${FC_DIR}/${SUBJECT}/${SESSION}/bold_fc_ready_mni.nii.gz
###############################################################################

export USERNAME="${USERNAME:-jade6100}"
export RAW_BIDS="${RAW_BIDS:-/scratch/alpine/${USERNAME}/openneuro/ds000224}"
export STUDY_DIR="${STUDY_DIR:-/scratch/alpine/${USERNAME}/openneuro/hcp_outputs}"
export FC_DIR="${FC_DIR:-/scratch/alpine/${USERNAME}/openneuro/fc_ready}"

export SUBJECTS="${SUBJECTS:-MSC01}"
export FUNC_SESSIONS="${FUNC_SESSIONS:-ses-func01}"

export DROP_TRS="${DROP_TRS:-5}"
export TR="${TR:-2.2}"
export HIGHPASS_SIGMA="${HIGHPASS_SIGMA:-22.7}"
export LOWPASS_SIGMA="${LOWPASS_SIGMA:-2.27}"

export FSLDIR="${FSLDIR:-/projects/ics/software/fsl/6.0.7}"
export FREESURFER_HOME="${FREESURFER_HOME:-/projects/ics/software/freesurfer/6.0.1}"
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

MNI_HEAD="${FSLDIR}/data/standard/MNI152_T1_2mm.nii.gz"
MNI_BRAIN="${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz"
MNI_MASK="${FSLDIR}/data/standard/MNI152_T1_2mm_brain_mask.nii.gz"

preflight() {
  echo "============================================================"
  echo "PREFLIGHT"
  echo "============================================================"
  echo "RAW_BIDS=${RAW_BIDS}"
  echo "STUDY_DIR=${STUDY_DIR}"
  echo "FC_DIR=${FC_DIR}"

  test -d "${RAW_BIDS}"
  mkdir -p "${STUDY_DIR}" "${FC_DIR}"

  echo
  echo "Tool checks:"
  which fslinfo
  which fslmaths
  which mcflirt
  which flirt
  which bet
  which robustfov
  which fsl_regfilt
  which fslnvols

  echo
  echo "Input counts:"
  echo -n "T1w count: "
  find "${RAW_BIDS}" -name "*_T1w.nii.gz" | sort | wc -l
  echo -n "Rest BOLD count: "
  find "${RAW_BIDS}" -name "*task-rest_bold.nii.gz" | sort | wc -l
}

process_subject_session() {
  local sub="$1"
  local ses="$2"
  local bids_sub="sub-${sub}"
  local run_label
  run_label=$(echo "${ses}" | sed 's/ses-func//')
  local fmri_name="rfMRI_REST${run_label}_LR"

  local hcp_sub_dir="${STUDY_DIR}/${sub}"
  local unproc_dir="${hcp_sub_dir}/unprocessed/3T"
  local t1w_unproc="${unproc_dir}/T1w_MPR1/${sub}_3T_T1w_MPR1.nii.gz"
  local fmri_unproc_dir="${unproc_dir}/${fmri_name}"
  local bold_unproc="${fmri_unproc_dir}/${sub}_3T_${fmri_name}.nii.gz"

  local t1w_src
  t1w_src=$(find "${RAW_BIDS}/${bids_sub}" -path "*/anat/*_T1w.nii.gz" | sort | head -1 || true)
  local bold_src="${RAW_BIDS}/${bids_sub}/${ses}/func/${bids_sub}_${ses}_task-rest_bold.nii.gz"

  if [[ -z "${t1w_src}" ]]; then
    echo "ERROR: no T1w found for ${bids_sub}"
    return 1
  fi

  if [[ ! -f "${bold_src}" ]]; then
    echo "WARNING: missing BOLD, skipping ${bids_sub} ${ses}"
    return 0
  fi

  echo
  echo "============================================================"
  echo "PROCESSING ${sub} ${ses}"
  echo "============================================================"

  mkdir -p "${unproc_dir}/T1w_MPR1" "${fmri_unproc_dir}"

  echo "Staging T1w:"
  echo "${t1w_src}"
  cp -f "${t1w_src}" "${t1w_unproc}"

  echo "Staging BOLD:"
  echo "${bold_src}"
  cp -f "${bold_src}" "${bold_unproc}"

  local t1w_dir="${hcp_sub_dir}/T1w"
  mkdir -p "${t1w_dir}"

  echo "Structural prep..."
  cp -f "${t1w_unproc}" "${t1w_dir}/T1w.nii.gz"
  robustfov -i "${t1w_dir}/T1w.nii.gz" -r "${t1w_dir}/T1w_acpc_dc.nii.gz"
  bet "${t1w_dir}/T1w_acpc_dc.nii.gz" "${t1w_dir}/T1w_acpc_dc_brain.nii.gz" -R -f 0.25 -g 0 -m
  flirt -in "${t1w_dir}/T1w_acpc_dc_brain.nii.gz" -ref "${MNI_BRAIN}" -omat "${t1w_dir}/T1w_to_MNI.mat" -out "${t1w_dir}/T1w_acpc_dc_brain_mni.nii.gz" -dof 12
  convert_xfm -omat "${t1w_dir}/MNI_to_T1w.mat" -inverse "${t1w_dir}/T1w_to_MNI.mat"

  local out_dir="${FC_DIR}/${sub}/${ses}"
  mkdir -p "${out_dir}"

  echo "Functional prep..."
  fslroi "${bold_unproc}" "${out_dir}/bold_drop.nii.gz" "${DROP_TRS}" -1
  fslmaths "${out_dir}/bold_drop.nii.gz" -Tmean "${out_dir}/bold_mean.nii.gz"
  bet "${out_dir}/bold_mean.nii.gz" "${out_dir}/bold_mean_brain.nii.gz" -f 0.3 -m
  mcflirt -in "${out_dir}/bold_drop.nii.gz" -out "${out_dir}/bold_mc.nii.gz" -plots -mats -reffile "${out_dir}/bold_mean_brain.nii.gz"
  fslmaths "${out_dir}/bold_mc.nii.gz" -mas "${out_dir}/bold_mean_brain_mask.nii.gz" "${out_dir}/bold_mc_brain.nii.gz"

  echo "Registering BOLD to T1w and MNI..."
  flirt -in "${out_dir}/bold_mean_brain.nii.gz" -ref "${t1w_dir}/T1w_acpc_dc_brain.nii.gz" -omat "${out_dir}/bold_to_T1w.mat" -out "${out_dir}/bold_mean_to_T1w.nii.gz" -dof 6
  convert_xfm -omat "${out_dir}/bold_to_MNI.mat" -concat "${t1w_dir}/T1w_to_MNI.mat" "${out_dir}/bold_to_T1w.mat"
  flirt -in "${out_dir}/bold_mc_brain.nii.gz" -ref "${MNI_HEAD}" -applyxfm -init "${out_dir}/bold_to_MNI.mat" -out "${out_dir}/bold_mc_mni.nii.gz" -interp trilinear

  echo "Motion cleanup..."
  local par="${out_dir}/bold_mc.nii.gz.par"
  if [[ ! -f "${par}" ]]; then
    par="${out_dir}/bold_mc.par"
  fi

  if [[ -f "${par}" ]]; then
    cp -f "${par}" "${out_dir}/motion.par"
    fsl_regfilt -i "${out_dir}/bold_mc_mni.nii.gz" -d "${out_dir}/motion.par" -f "1,2,3,4,5,6" -o "${out_dir}/bold_mc_mni_motionreg.nii.gz"
  else
    echo "WARNING: motion parameter file not found. Continuing without motion regression."
    cp -f "${out_dir}/bold_mc_mni.nii.gz" "${out_dir}/bold_mc_mni_motionreg.nii.gz"
  fi

  echo "Temporal filtering..."
  fslmaths "${out_dir}/bold_mc_mni_motionreg.nii.gz" \
    -bptf "${HIGHPASS_SIGMA}" "${LOWPASS_SIGMA}" \
    "${out_dir}/bold_fc_ready_mni.nii.gz"

  fslmaths "${out_dir}/bold_fc_ready_mni.nii.gz" -mas "${MNI_MASK}" "${out_dir}/bold_fc_ready_mni.nii.gz"
  fslmaths "${out_dir}/bold_fc_ready_mni.nii.gz" -Tmean "${out_dir}/bold_fc_ready_mni_mean.nii.gz"
  fslnvols "${out_dir}/bold_fc_ready_mni.nii.gz" > "${out_dir}/n_volumes_after_drop.txt"

  echo "Finished ${sub} ${ses}"
  echo "Output: ${out_dir}/bold_fc_ready_mni.nii.gz"
}

preflight

for sub in ${SUBJECTS}; do
  for ses in ${FUNC_SESSIONS}; do
    process_subject_session "${sub}" "${ses}"
  done
done

echo
echo "All requested processing complete."
find "${FC_DIR}" -name "bold_fc_ready_mni.nii.gz" | sort
