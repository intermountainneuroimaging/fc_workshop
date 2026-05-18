# Functional Connectivity Tutorial

A self-contained Jupyter notebook for computing seed-to-parcellation **functional connectivity (FC)** matrices from **fMRIPrep** preprocessed data.

---

## What the notebook does

1. Loads a preprocessed BOLD image and its fMRIPrep confounds TSV
2. Optionally loads a BIDS events file for task-based inspection
3. Fetches a parcellation atlas (Schaefer-200 by default, or your own NIfTI)
4. Extracts parcel-averaged time series with confound regression + bandpass filtering
5. Computes Pearson correlations and applies Fisher r → z transform
6. Saves the FC matrix as a labelled CSV

---

## Required inputs

| File | Description |
|------|-------------|
| `*_desc-preproc_bold.nii.gz` | fMRIPrep preprocessed BOLD (MNI space) |
| `*_desc-confounds_timeseries.tsv` | fMRIPrep confounds file for the same run |
| `*_events.tsv` | BIDS events file *(resting-state: set `EVENTS_PATH = None`)* |

All paths are set in the **Configuration** cell at the top of the notebook.

---

## Setup — conda environment

### Option A: create from the provided YAML (recommended)

```bash
conda env create -f environment.yml
conda activate fmri-fc
python -m ipykernel install --user --name fmri-fc --display-name "Python 3 (fmri-fc)"
jupyter lab
```

### Option B: manual install into an existing environment

```bash
conda create -n fmri-fc python=3.11
conda activate fmri-fc

conda install -c conda-forge \
    numpy pandas scipy matplotlib seaborn \
    nibabel nilearn \
    jupyter jupyterlab ipykernel ipywidgets tqdm

python -m ipykernel install --user --name fmri-fc --display-name "Python 3 (fmri-fc)"
jupyter lab
```

---

## Package versions (tested)

| Package | Min version | Notes |
|---------|-------------|-------|
| Python  | 3.10+ | |
| numpy   | 1.24+ | |
| pandas  | 2.0+  | |
| scipy   | 1.11+ | used for `pearsonr` (validation only) |
| matplotlib | 3.7+ | |
| seaborn | 0.13+ | heatmap plotting |
| nibabel | 5.1+  | NIfTI I/O |
| nilearn | 0.10+ | `NiftiLabelsMasker`, atlas fetching, plotting |
| jupyter / jupyterlab | any recent | |

---

## Key configuration options

Open the **Configuration** cell in the notebook and adjust:

```python
# Paths
FMRIPREP_DIR   = '/path/to/fmriprep'
SUBJECT        = 'sub-01'
SESSION        = 'ses-01'    # None if no session level
TASK           = 'rest'
RUN            = 'run-01'    # None if no run level

# Parcellation
ATLAS          = 'schaefer'  # 'schaefer' | 'destrieux' | 'custom'
N_ROIS         = 200         # Schaefer: 100, 200, 300 … 1000
ATLAS_PATH     = None        # path to custom NIfTI when ATLAS='custom'

# Confound regression
CONFOUND_COLS  = [...]       # list of column names from confounds TSV
SCRUB_THRESHOLD = 0.5        # FD threshold in mm; None = no scrubbing

# Acquisition
TR             = 2.0         # seconds
```

---

## Output

The notebook writes a CSV to `./fc_output/` with rows and columns labelled by ROI name:

```
sub-01_ses-01_task-rest_atlas-schaefer200_fc-fisherz.csv
```

Each cell contains the Fisher z-transformed correlation between that pair of parcels. The diagonal is set to 0.

---

## Downloading the MSC dataset (ds000224)

The Midnight Scan Club dataset is publicly available on AWS S3 — no AWS account or credentials required.

### Quick download (AWS CLI one-liner)

Install the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), then:

```bash
# Full dataset (~200 GB)
aws s3 sync --no-sign-request \
    s3://openneuro.org/ds000224 \
    ./ds000224

# Single subject only
aws s3 sync --no-sign-request \
    s3://openneuro.org/ds000224/sub-MSC01 \
    ./ds000224/sub-MSC01

# Functional data only (all subjects)
aws s3 sync --no-sign-request \
    s3://openneuro.org/ds000224 \
    ./ds000224 \
    --exclude "*" --include "*/func/*"
```

### Python script (recommended for fine-grained control)

`download_msc.py` uses **boto3** with anonymous access and supports subject/session/datatype filtering, parallel downloads, and resume-on-interrupt.

**Install dependencies:**

```bash
pip install boto3 tqdm
# or, if using the conda environment:
conda activate fmri-fc && pip install boto3 tqdm
```

**Run:**

```bash
# Preview what would be downloaded
python download_msc.py --dry-run

# List all matching S3 keys and sizes
python download_msc.py --list

# Download (uses CONFIG block inside the script)
python download_msc.py
```

**Key CONFIG options in `download_msc.py`:**

```python
CONFIG = dict(
    output_dir  = './ds000224',         # where to save files

    subjects    = ['sub-MSC01'],        # None = all 10 subjects
    sessions    = ['ses-func01'],       # None = all sessions
    datatypes   = ['func', 'anat'],     # None = all modalities
    suffixes    = ['.nii.gz', '.tsv', '.json'],  # None = all files

    include_derivatives = False,        # skip fMRIPrep derivatives
    skip_existing       = True,         # resume interrupted downloads
    n_workers           = 4,            # parallel threads
)
```

### Dataset structure

```
ds000224/
├── participants.tsv          # subject metadata (age, sex, handedness)
├── dataset_description.json
├── sub-MSC01/
│   ├── ses-func01/           # 10 functional sessions per subject
│   │   ├── func/
│   │   │   ├── *_task-rest_bold.nii.gz
│   │   │   ├── *_task-rest_bold.json
│   │   │   └── *_task-rest_events.tsv
│   │   └── fmap/
│   └── ses-struct01/         # 2 structural sessions per subject
│       └── anat/
│           ├── *_T1w.nii.gz
│           └── *_T2w.nii.gz
├── sub-MSC02/ … sub-MSC10/
└── derivatives/
    └── fmriprep/             # preprocessed data (~150 GB)
```

### Approximate sizes

| Scope | Size |
|-------|------|
| Raw BOLD only (`func/`) | ~80 GB |
| Raw + structural | ~100 GB |
| fMRIPrep derivatives | ~150 GB |
| Full dataset | ~250 GB |

---

## Notes on confound strategy

The default `CONFOUND_COLS` list implements a **24-parameter motion** model (6 motion parameters + derivatives + quadratics + their derivatives) plus **white matter** and **CSF** signals. This is a conservative but standard approach. Common alternatives:

- **6-parameter**: just `trans_x/y/z`, `rot_x/y/z`
- **aCompCor**: replace `white_matter`/`csf` with `a_comp_cor_00` … `a_comp_cor_05`
- **ICA-AROMA**: use `aroma_motion_*` columns (requires fMRIPrep `--use-aroma`)

Change `CONFOUND_COLS` in the configuration cell to switch strategies.

---

## Extending the notebook

- **Group analysis**: loop over subjects, load each FC CSV, stack into a 3D array `(n_subs, n_rois, n_rois)`, and run a paired t-test or GLM across the upper triangle.
- **Task FC**: use `EVENTS_PATH` to identify condition blocks, extract time series windows per condition, and compute separate FC matrices per condition before contrasting.
- **Graph theory**: pass `fc_matrix` to `networkx` or `bct` (Brain Connectivity Toolbox) for modularity, clustering coefficient, or hub detection.
