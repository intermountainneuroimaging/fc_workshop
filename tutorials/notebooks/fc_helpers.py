"""
fc_helpers.py
=============
Helper functions for the Functional Connectivity Tutorial.

Import everything into your notebook with:
    from fc_helpers import *

Then call  fc_help()              — list all available functions
           fc_help('load_confounds')  — show detailed docs for one function

Available functions
-------------------
load_confounds      Load and clean fMRIPrep confound regressors
get_parcellation    Fetch a nilearn atlas or load a custom NIfTI
check_alignment     Check that BOLD and atlas share the same template space
plot_alignment      Visually overlay the atlas on the mean BOLD image
extract_time_series Extract parcel-averaged BOLD time series
compute_fc_matrix   Compute Pearson correlations + Fisher z-transform
save_fc_matrix      Save an FC matrix as a labelled CSV
plot_fc_matrix      Plot a connectivity matrix heatmap
plot_time_series    Plot a sample of parcel time series
plot_fd_trace       Plot the framewise displacement trace
summarise_fc        Print summary statistics of an FC matrix
"""

import os
import glob as _glob
import textwrap
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import nibabel as nib
from pathlib import Path
from nilearn import datasets, plotting
from nilearn.maskers import NiftiLabelsMasker


# =============================================================================
# Help system
# =============================================================================

_REGISTRY = {}   # populated by the decorator below
_logger   = None # FCLogger instance; None until init_log() is called
_config   = {}   # pipeline config; populated by init_log()
_UNSET    = object()  # sentinel: distinguishes "not passed" from None


def _register(fn):
    """Decorator: add a function to the help registry."""
    _REGISTRY[fn.__name__] = fn
    return fn


def fc_help(name=None):
    """
    Display help for the fc_helpers module.

    Parameters
    ----------
    name : str or None
        If None, print a one-line summary of every available function.
        If a function name (str), print its full docstring.

    Examples
    --------
    >>> fc_help()                      # list all functions
    >>> fc_help('load_confounds')      # detailed docs for one function
    >>> fc_help('compute_fc_matrix')
    """
    if name is None:
        _print_index()
    else:
        if name not in _REGISTRY:
            print(f"  No function named '{name}' in fc_helpers.")
            print("  Call fc_help() with no arguments to see available functions.")
            return
        fn = _REGISTRY[name]
        doc = fn.__doc__ or '  (no docstring)'
        header = f'  fc_helpers.{name}'
        print('=' * 70)
        print(header)
        print('=' * 70)
        print(textwrap.dedent(doc))


def _print_index():
    width = 70
    print('=' * width)
    print('  fc_helpers — Functional Connectivity Utility Functions')
    print('=' * width)
    print('  Call fc_help("<name>") for detailed docs on any function.\n')
    groups = [
        ('Logging',        ['init_log', 'build_fmriprep_paths']),
        ('Data loading',   ['load_confounds', 'get_parcellation',
                            'check_alignment', 'plot_alignment']),
        ('FC Analysis',    ['extract_time_series', 'compute_fc_matrix']),
        ('gPPI Analysis',  ['load_events', 'build_hrf_regressors',
                            'build_gppi_design', 'compute_gppi_matrix',
                            'plot_gppi_seed_profile']),
        ('Output',         ['save_fc_matrix']),
        ('Visualisation',  ['plot_fc_matrix', 'plot_time_series',
                            'plot_fd_trace']),
        ('Utilities',      ['summarise_fc']),
    ]
    for group_name, names in groups:
        print(f'  ── {group_name} ' + '─' * (width - len(group_name) - 6))
        for n in names:
            fn = _REGISTRY.get(n)
            if fn is None:
                continue
            # Extract first sentence of docstring for the summary line
            first = (fn.__doc__ or '').strip().split('\n')[0]
            print(f'  {n:<25}  {first}')
        print()
    print('=' * width)


# =============================================================================
# Module-level docstring update for gPPI
# (appended to the Available functions table in the module docstring)
# =============================================================================
# load_events          Load a BIDS events TSV for task-based analysis
# build_hrf_regressors Convolve task timing with a haemodynamic response function
# build_gppi_design    Assemble the gPPI design matrix for one seed ROI
# compute_gppi_matrix  Compute a full ROI × ROI gPPI connectivity matrix
# plot_gppi_seed_profile  Bar chart of strongest gPPI connections for one seed


# =============================================================================
# Pipeline logger
# =============================================================================

class FCLogger:
    """
    Lightweight pipeline logger for functional connectivity runs.

    Records every configuration parameter and every computational step
    taken during a pipeline run, then writes a human-readable ``.log``
    file alongside the output CSV.

    Do not instantiate directly — call ``init_log(config)`` instead.
    """

    def __init__(self, config: dict):
        self._start    = datetime.datetime.now()
        self._config   = dict(config)
        self._steps    = []          # list of (timestamp, step_name, details)

    # ------------------------------------------------------------------
    def log_step(self, step_name: str, **kwargs):
        """Record a pipeline step with arbitrary key-value details."""
        ts = datetime.datetime.now()
        self._steps.append((ts, step_name, dict(kwargs)))

    # ------------------------------------------------------------------
    def save(self, path):
        """
        Write the accumulated log to *path* (.log file).

        Called automatically by ``save_fc_matrix`` — you rarely need to
        call this yourself.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        W = 72

        # ── Header ──────────────────────────────────────────────────────
        lines.append('=' * W)
        lines.append('  Functional Connectivity Pipeline Log')
        lines.append('=' * W)
        lines.append(f'  Run started : {self._start.strftime("%Y-%m-%d  %H:%M:%S")}')
        lines.append(f'  Run ended   : {datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")}')
        elapsed = datetime.datetime.now() - self._start
        lines.append(f'  Elapsed     : {str(elapsed).split(".")[0]}')
        lines.append(f'  Log saved   : {path}')
        lines.append('')

        # ── Configuration ────────────────────────────────────────────────
        lines.append('─' * W)
        lines.append('  CONFIGURATION PARAMETERS')
        lines.append('─' * W)
        col_w = max((len(k) for k in self._config), default=10) + 2
        for k, v in self._config.items():
            lines.append(f'  {k:<{col_w}}: {v}')
        lines.append('')

        # ── Steps ─────────────────────────────────────────────────────────
        lines.append('─' * W)
        lines.append('  COMPUTATIONAL STEPS')
        lines.append('─' * W)
        for i, (ts, name, details) in enumerate(self._steps, start=1):
            ts_str = ts.strftime('%H:%M:%S')
            lines.append(f'  [{ts_str}]  STEP {i} — {name}')
            if details:
                det_w = max(len(k) for k in details) + 2
                for k, v in details.items():
                    lines.append(f'    {k:<{det_w}}: {v}')
            lines.append('')

        if not self._steps:
            lines.append('  (no steps recorded)')
            lines.append('')

        lines.append('=' * W)
        lines.append('  END OF LOG')
        lines.append('=' * W)

        with open(path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')

        print(f'  [log] Saved → {path}')
        return str(path)


@_register
def init_log(config: dict):
    """
    Initialise the pipeline logger with the current configuration.

    Call this once at the top of your pipeline (in the configuration cell)
    after setting all your CONFIG variables.  From that point on, every
    fc_helpers function automatically records its parameters and outputs.
    The log is written to the same directory as the FC matrix CSV, with
    the same filename stem and a ``.log`` extension.

    Parameters
    ----------
    config : dict
        A dictionary of all configuration parameters for this run.
        Typical keys: ``SUBJECT``, ``SESSION``, ``TASK``, ``RUN``,
        ``ATLAS``, ``N_ROIS``, ``TR``, ``SCRUB_THRESHOLD``,
        ``USE_GLOBAL_SIGNAL``, ``CONFOUND_COLS``, ``BOLD_PATH``,
        ``CONFOUNDS_PATH``, ``OUTPUT_DIR``, ``OUTPUT_FILENAME``.

    Returns
    -------
    logger : FCLogger
        The active logger instance (also stored as module-level state).

    Notes
    -----
    - After calling ``init_log()``, logging is fully automatic — you do
      not need to call any other logging functions.
    - The log file is written when ``save_fc_matrix()`` is called.  If
      you never call ``save_fc_matrix()``, call ``_logger.save(path)``
      manually at the end of your pipeline.
    - Only one logger is active at a time; calling ``init_log()`` again
      starts a fresh log.

    Examples
    --------
    Typical use in the notebook configuration cell:

    >>> init_log({
    ...     'SUBJECT'          : SUBJECT,
    ...     'SESSION'          : SESSION,
    ...     'TASK'             : TASK,
    ...     'RUN'              : RUN,
    ...     'ATLAS'            : ATLAS,
    ...     'N_ROIS'           : N_ROIS,
    ...     'TR'               : TR,
    ...     'SCRUB_THRESHOLD'  : SCRUB_THRESHOLD,
    ...     'USE_GLOBAL_SIGNAL': USE_GLOBAL_SIGNAL,
    ...     'CONFOUND_COLS'    : CONFOUND_COLS,
    ...     'BOLD_PATH'        : BOLD_PATH,
    ...     'CONFOUNDS_PATH'   : CONFOUNDS_PATH,
    ...     'OUTPUT_DIR'       : OUTPUT_DIR,
    ...     'OUTPUT_FILENAME'  : OUTPUT_FILENAME,
    ... })

    See also
    --------
    save_fc_matrix : Triggers automatic log save alongside the CSV.
    """
    global _logger, _config
    _config = dict(config)
    _logger = FCLogger(config)
    n = len(config)
    print(f'  [log] Pipeline logger initialised  ({n} config params recorded)')
    print(f'  [log] Log will be saved alongside the FC matrix CSV (.log)')
    return _logger


@_register
def build_fmriprep_paths(fmriprep_dir=_UNSET, subject=_UNSET, session=_UNSET,
                         task=_UNSET, run=_UNSET, space=_UNSET,
                         debug=False, **filters):
    """
    Locate standard fMRIPrep derivative files by searching the directory.

    Rather than constructing a hardcoded path, this function globs the
    ``func/`` directory and returns the first file whose name contains all
    required BIDS entities.  Optional entities such as ``res``, ``acq``,
    ``dir``, and ``echo`` are passed as keyword arguments and used to
    narrow the search; if omitted, any value matches.

    Any positional argument left at its default is looked up from the
    config stored by ``init_log()``; explicitly passed values (including
    ``None`` for session/run) always take priority.

    Parameters
    ----------
    fmriprep_dir : str or None
        Root directory of the fMRIPrep derivatives tree.
        Falls back to ``_config['FMRIPREP_DIR']``.

    subject : str or None
        BIDS subject label, e.g. ``'sub-MSC01'``.
        Falls back to ``_config['SUBJECT']``.

    session : str or None
        BIDS session label, e.g. ``'ses-func01'``.  Pass ``None`` for
        datasets without a session level.
        Falls back to ``_config['SESSION']``.

    task : str or None
        BIDS task label, e.g. ``'rest'``.
        Falls back to ``_config['TASK']``.

    run : str or None
        BIDS run label, e.g. ``'run-01'``.  Pass ``None`` when there is
        no run level.  Falls back to ``_config['RUN']``.

    space : str or None
        Template space label, e.g. ``'MNI152NLin2009cAsym'``.
        Falls back to ``_config['SPACE']``; default
        ``'MNI152NLin2009cAsym'``.

    **filters : str
        Optional BIDS key-value pairs used to narrow the glob.  Any
        ``key=value`` pair is translated to the token ``key-value`` and
        must appear in the filename.  Common keys:

        ``res``   resolution label, e.g. ``res='2'``
        ``acq``   acquisition label, e.g. ``acq='mb'``
        ``dir``   phase-encoding direction, e.g. ``dir='AP'``
        ``echo``  echo number for multi-echo data, e.g. ``echo='1'``

        Omitting a key matches *any* value for that entity.

    Returns
    -------
    paths : dict with keys
        ``'BOLD_PATH'``      — preprocessed BOLD NIfTI (or ``None`` if not found)
        ``'CONFOUNDS_PATH'`` — fMRIPrep confounds TSV  (or ``None`` if not found)
        ``'EVENTS_PATH'``    — BIDS events TSV          (or ``None`` if not found)

    Notes
    -----
    - When multiple files match the pattern a warning is printed and the
      first alphabetically is returned.  Use ``**filters`` to disambiguate.
    - The BOLD search requires ``space-{space}`` and ``desc-preproc`` in
      the filename; confounds requires ``desc-confounds_timeseries``;
      events requires ``_events.tsv``.
    - If no BOLD file is found the function retries without the space
      constraint so you still get a useful warning rather than silence.

    Examples
    --------
    After ``init_log()``, no arguments needed:

    >>> paths = build_fmriprep_paths()
    >>> BOLD_PATH, CONFOUNDS_PATH = paths['BOLD_PATH'], paths['CONFOUNDS_PATH']

    Restrict to 2 mm resolution explicitly:

    >>> paths = build_fmriprep_paths(res='2')

    Override subject in a loop, keeping everything else from config:

    >>> for sub in SUBJECTS:
    ...     paths = build_fmriprep_paths(subject=sub)

    Fully explicit, no ``init_log`` required:

    >>> paths = build_fmriprep_paths(
    ...     fmriprep_dir = '/data/fmriprep',
    ...     subject      = 'sub-MSC01',
    ...     session      = 'ses-func01',
    ...     task         = 'rest',
    ...     run          = None,
    ...     res          = '2',
    ... )

    See also
    --------
    init_log : Stores config so positional arguments can be omitted here.
    """
    # ── Resolve arguments (sentinel → _config → default) ─────────────────────
    def _resolve(val, key, default=None):
        return val if val is not _UNSET else _config.get(key, default)

    fmriprep_dir = _resolve(fmriprep_dir, 'FMRIPREP_DIR')
    subject      = _resolve(subject,      'SUBJECT')
    session      = _resolve(session,      'SESSION')
    task         = _resolve(task,         'TASK')
    run          = _resolve(run,          'RUN')
    space        = _resolve(space,        'SPACE', 'MNI152NLin2009cAsym')

    if fmriprep_dir is None or subject is None or task is None:
        raise ValueError(
            'fmriprep_dir, subject, and task are required. '
            'Either pass them explicitly or call init_log(config) first.'
        )

    # ── func/ directory for this subject/session ──────────────────────────────
    func_dir = os.path.join(
        fmriprep_dir,
        *([subject, session, 'func'] if session else [subject, 'func'])
    )

    # ── Glob prefix: only the entities that are always first in BIDS order ────
    # sub → ses → task.  We stop here because optional entities (dir, acq, ce,
    # rec) can appear between task and run in any dataset, breaking a rigid
    # prefix that includes run.  run and all **filters are handled as token
    # filters post-glob so they work regardless of entity ordering.
    prefix_tokens = [subject]
    if session: prefix_tokens.append(session)
    prefix_tokens.append(f'task-{task}')
    prefix = '_'.join(prefix_tokens)

    # ── Token filters: run + any **filters (e.g. res='2' → 'res-2') ──────────
    # run is the full BIDS label ('run-01'), not just the number — use as-is
    run_token = [run] if run else []
    filter_tokens = run_token + [f'{k}-{v}' for k, v in filters.items()]

    def _find(required_tokens, glob_suffix):
        """
        Glob broadly then filter to files that contain every required token.
        Returns the first alphabetical match, or None.
        """
        pattern = os.path.join(func_dir, f'{prefix}*{glob_suffix}')
        candidates = sorted(_glob.glob(pattern))

        if debug:
            print(f'\n  [debug] glob pattern : {pattern}')
            print(f'  [debug] func_dir exists: {os.path.isdir(func_dir)}')
            if os.path.isdir(func_dir):
                all_files = sorted(os.listdir(func_dir))
                print(f'  [debug] files in func_dir ({len(all_files)}):')
                for f in all_files:
                    print(f'            {f}')
            print(f'  [debug] glob candidates: {[os.path.basename(c) for c in candidates]}')
            print(f'  [debug] required tokens: {required_tokens}')

        # Keep only files that contain every required token in their basename
        matches = [
            c for c in candidates
            if all(tok in os.path.basename(c) for tok in required_tokens)
        ]

        if debug:
            print(f'  [debug] after token filter: {[os.path.basename(m) for m in matches]}')

        if not matches:
            return None
        if len(matches) > 1:
            print(f'  [paths] WARNING: {len(matches)} files match — using first.')
            print(f'          Pass **filters (e.g. res="2") to disambiguate:')
            for m in matches:
                print(f'            {os.path.basename(m)}')
        return matches[0]

    # ── BOLD — apply space + all **filters ────────────────────────────────────
    # filter_tokens (e.g. ['res-2', 'acq-mb']) only make sense for BOLD;
    # confounds and events filenames never carry these entities.
    bold_required = [f'space-{space}', 'desc-preproc'] + filter_tokens
    bold_path = _find(bold_required, '_desc-preproc_bold.nii.gz')

    if bold_path is None and space:
        # Retry without space constraint so we surface what actually exists
        bold_path = _find(['desc-preproc'] + filter_tokens, '_desc-preproc_bold.nii.gz')
        if bold_path:
            print(f'  [paths] WARNING: no BOLD found for space-{space}; '
                  f'found without space constraint — check your SPACE setting.')

    # ── Confounds — run token only (res/acq/dir not in confounds filenames) ───
    confounds_path = _find(run_token, '_desc-confounds_timeseries.tsv')

    # ── Events — run token only ────────────────────────────────────────────────
    events_path = _find(run_token, '_events.tsv')

    paths = {
        'BOLD_PATH'      : bold_path,
        'CONFOUNDS_PATH' : confounds_path,
        'EVENTS_PATH'    : events_path,
    }

    w = 12
    print(f'  [paths] {"BOLD":<{w}}: {bold_path      or "NOT FOUND ⚠"}')
    print(f'  [paths] {"Confounds":<{w}}: {confounds_path or "NOT FOUND ⚠"}')
    print(f'  [paths] {"Events":<{w}}: {events_path    or "not found (resting-state?)"}')
    return paths


# =============================================================================
# Data loading
# =============================================================================

@_register
def load_confounds(confounds_path, cols, scrub_threshold=None):
    """
    Load and clean confound regressors from an fMRIPrep confounds TSV file
    or an FSL mcflirt ``*.par`` motion parameter file.

    fMRIPrep writes a TSV with one row per volume and one column per
    confound variable.  FSL mcflirt writes a 6-column whitespace-delimited
    file with no header (rotations in radians, translations in mm).

    This function selects the requested columns, replaces NaN values
    (which appear in the first row of derivative columns) with 0, and
    optionally adds spike regressors for high-motion volumes identified
    by framewise displacement (FD).

    Parameters
    ----------
    confounds_path : str or Path
        Path to either:

        - An fMRIPrep confounds TSV
          (``*_desc-confounds_timeseries.tsv``), or
        - An FSL mcflirt parameter file (``*.par``).

        The format is detected automatically from the file extension.

    cols : list of str
        Column names to include.  Any name not present in the file is
        silently skipped (a warning is printed).  Common choices:

        Motion (6 DOF) — available in **both** TSV and ``.par`` formats:
            'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'
        Derivatives (6 DOF) — fMRIPrep TSV only:
            'trans_x_derivative1', ... , 'rot_z_derivative1'
        Quadratics + their derivatives (12 columns) — fMRIPrep TSV only:
            'trans_x_power2', ... , 'rot_z_derivative1_power2'
        Tissue signals — fMRIPrep TSV only:
            'white_matter', 'csf', 'global_signal'
        aCompCor (first 5) — fMRIPrep TSV only:
            'a_comp_cor_00' ... 'a_comp_cor_04'

        For ``.par`` files only the 6 standard motion column names are
        available; any other requested columns will be skipped with a
        warning.

    scrub_threshold : float or None
        Framewise displacement threshold in mm.  Volumes with FD above
        this value receive a binary spike regressor (one column per
        volume).  ``None`` disables scrubbing.  Typical value: 0.5 mm.

        For ``.par`` files, FD is computed automatically from the motion
        parameters using the Power et al. (2012) formula
        (r = 50 mm head radius for rotation → mm conversion).

    Returns
    -------
    confounds_array : np.ndarray, shape (T, n_regressors)
        Confound matrix ready to pass to ``extract_time_series``.
    scrubbed_idx : list of int
        Indices of volumes that were scrubbed (empty list if
        ``scrub_threshold`` is None or no volumes exceeded it).

    Notes
    -----
    - FSL mcflirt ``.par`` column order:
      rot_x (rad), rot_y (rad), rot_z (rad),
      trans_x (mm), trans_y (mm), trans_z (mm).
    - The first row of derivative columns is NaN in fMRIPrep output
      (no derivative can be computed for volume 0).  These are set to 0.
    - Scrubbing spike regressors are appended as additional columns after
      the regular confounds.
    - For resting-state data the 24-parameter motion model (6 + 6
      derivatives + 6 quadratics + 6 derivative-quadratics) plus white
      matter and CSF signals is a conservative but standard strategy.
    - For task data consider leaving out the global signal to avoid
      signal cancellation across task-active regions.

    Examples
    --------
    fMRIPrep TSV — 24-parameter model with scrubbing:

    >>> CONFOUND_COLS = [
    ...     'trans_x', 'trans_y', 'trans_z',
    ...     'rot_x',   'rot_y',   'rot_z',
    ...     'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
    ...     'rot_x_derivative1',   'rot_y_derivative1',   'rot_z_derivative1',
    ...     'trans_x_power2',      'trans_y_power2',      'trans_z_power2',
    ...     'rot_x_power2',        'rot_y_power2',        'rot_z_power2',
    ...     'trans_x_derivative1_power2', 'trans_y_derivative1_power2',
    ...     'trans_z_derivative1_power2', 'rot_x_derivative1_power2',
    ...     'rot_y_derivative1_power2',   'rot_z_derivative1_power2',
    ...     'white_matter', 'csf',
    ... ]
    >>> conf, scrubbed = load_confounds(
    ...     CONFOUNDS_PATH, CONFOUND_COLS, scrub_threshold=0.5
    ... )
    >>> print(conf.shape)   # (T, 26 + n_scrubbed_volumes)

    FSL ``.par`` file — only 6-DOF motion available:

    >>> conf, scrubbed = load_confounds(
    ...     'sub-01_task-rest_mc.par',
    ...     cols=['trans_x', 'trans_y', 'trans_z',
    ...           'rot_x',   'rot_y',   'rot_z'],
    ...     scrub_threshold=0.5,
    ... )

    See also
    --------
    plot_fd_trace       : Visualise the framewise displacement trace.
    extract_time_series : Pass the returned array as ``confounds_array``.
    """
    confounds_path = Path(confounds_path)
    _PAR_COLS = ['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']

    if confounds_path.suffix == '.par':
        # ── FSL mcflirt *.par format ──────────────────────────────────────
        # 6 whitespace-delimited columns, no header
        # Order: rot_x(rad), rot_y(rad), rot_z(rad),
        #        trans_x(mm), trans_y(mm), trans_z(mm)
        df = pd.read_csv(confounds_path, sep=r'\s+', header=None,
                         names=_PAR_COLS, engine='python')
        fmt = 'fsl_par'
        print(f'  [confounds] Detected FSL .par format '
              f'({len(df)} volumes, 6 DOF)')

        # Compute FD using Power et al. (2012): |Δtrans| + r*|Δrot|, r=50 mm
        _r    = 50.0
        _diff = df[_PAR_COLS].diff().fillna(0).abs()
        _diff[['rot_x', 'rot_y', 'rot_z']] *= _r
        df['framewise_displacement'] = _diff.sum(axis=1)
    else:
        # ── fMRIPrep TSV format ───────────────────────────────────────────
        df  = pd.read_csv(confounds_path, sep='\t')
        fmt = 'fmriprep_tsv'

    available = [c for c in cols if c in df.columns]
    missing   = [c for c in cols if c not in df.columns]
    if missing:
        print(f'  [confounds] Columns not found, skipping: {missing}')

    confounds = df[available].copy().fillna(0)

    scrubbed_idx = []
    if scrub_threshold is not None and 'framewise_displacement' in df.columns:
        fd = df['framewise_displacement'].fillna(0)
        scrubbed_idx = list(np.where(fd > scrub_threshold)[0])
        print(f'  [confounds] Scrubbing {len(scrubbed_idx)} volumes '
              f'(FD > {scrub_threshold} mm)')
        for idx in scrubbed_idx:
            col = np.zeros(len(df))
            col[idx] = 1.0
            confounds[f'scrub_{idx:04d}'] = col

    print(f'  [confounds] Format: {fmt}')
    print(f'  [confounds] Matrix shape: {confounds.shape}  (T × regressors)')

    if _logger:
        _logger.log_step(
            'load_confounds',
            confounds_path   = str(confounds_path),
            format           = fmt,
            cols_requested   = len(cols),
            cols_found       = len(available),
            cols_skipped     = len(missing),
            scrub_threshold  = scrub_threshold,
            volumes_scrubbed = len(scrubbed_idx),
            scrubbed_indices = scrubbed_idx if len(scrubbed_idx) <= 20
                               else f'{len(scrubbed_idx)} volumes (list truncated)',
            output_shape     = f'{confounds.shape[0]} volumes × {confounds.shape[1]} regressors',
        )

    return confounds.values, scrubbed_idx


@_register
def get_parcellation(atlas='schaefer', n_rois=200, yeo_networks=7,
                     atlas_path=None):
    """
    Fetch a parcellation atlas image and return its ROI labels.

    Supports three built-in nilearn atlases and a custom NIfTI option.
    The returned image can be passed directly to ``extract_time_series``.

    Parameters
    ----------
    atlas : {'schaefer', 'destrieux', 'custom'}
        Which parcellation to load.

        ``'schaefer'``
            Schaefer 2018 cortical parcellation (Kong et al. 2022).
            Yeo-7 or Yeo-17 network assignment. Resolution: 1 or 2 mm.
        ``'destrieux'``
            FreeSurfer Destrieux atlas (148 cortical parcels, 2009).
        ``'custom'``
            Load any NIfTI parcellation; requires ``atlas_path``.  ROI
            labels are auto-generated as ``ROI_001`` … ``ROI_N``.

    n_rois : int
        Number of parcels for Schaefer atlas.  Must be a multiple of 100
        in the range 100–1000.  Ignored for other atlases.

    yeo_networks : {7, 17}
        Yeo network assignment for the Schaefer atlas.  Ignored for other
        atlases.

    atlas_path : str or Path or None
        Path to a NIfTI label image when ``atlas='custom'``.  Each unique
        non-zero integer in the image defines one parcel.

    Returns
    -------
    atlas_img : nibabel.Nifti1Image
        4-D or 3-D integer label image in MNI152 space.
    labels : list of str
        ROI names, one per parcel (length == n_rois).

    Notes
    -----
    - All built-in atlases are downloaded automatically by nilearn on
      first use and cached in ``~/nilearn_data/``.
    - For custom atlases provide a sidecar JSON or TSV with ROI names and
      pass them in manually if you need meaningful labels.
    - The image is in MNI152 2 mm space.  ``extract_time_series`` will
      resample the BOLD data to match.

    Examples
    --------
    Schaefer 400-parcel, 17-network:

    >>> atlas_img, labels = get_parcellation(
    ...     atlas='schaefer', n_rois=400, yeo_networks=17
    ... )
    >>> print(len(labels))   # 400

    Destrieux:

    >>> atlas_img, labels = get_parcellation(atlas='destrieux')

    Custom atlas:

    >>> atlas_img, labels = get_parcellation(
    ...     atlas='custom', atlas_path='/path/to/my_atlas.nii.gz'
    ... )

    See also
    --------
    extract_time_series : Pass ``atlas_img`` as the second argument.
    plot_fc_matrix      : Pass ``labels`` as ROI tick labels.
    """
    if atlas == 'schaefer':
        data = datasets.fetch_atlas_schaefer_2018(
            n_rois=n_rois, yeo_networks=yeo_networks, resolution_mm=2)
        atlas_img = nib.load(data.maps)
        labels = list(data.labels)
        if labels and isinstance(labels[0], bytes):
            labels = [l.decode() for l in labels]
        # Some nilearn versions include a label entry for the background parcel
        # (atlas integer value 0) in addition to the n_rois real parcel labels.
        # np.unique returns sorted integer values, so labels[i] corresponds to
        # unique_ints[i].  If 0 is in that list, labels[i where unique==0] is
        # the background label — drop it by name regardless of its position.
        unique_ints = sorted(int(v) for v in np.unique(atlas_img.get_fdata()))
        if 0 in unique_ints and len(labels) == len(unique_ints):
            bg_idx  = unique_ints.index(0)   # index of value 0 in sorted list
            bg_name = labels[bg_idx]
            labels  = [l for l in labels if l != bg_name]
            print(f'  [atlas] NOTE: dropped background label '
                  f'"{bg_name}" (atlas value 0)')
        print(f'  [atlas] Schaefer-{n_rois} '
              f'(Yeo-{yeo_networks}): {len(labels)} parcels')

    elif atlas == 'destrieux':
        data = datasets.fetch_atlas_destrieux_2009()
        atlas_img = nib.load(data.maps)
        labels = [str(l[1]) for l in data.labels if l[0] != 0]
        print(f'  [atlas] Destrieux 2009: {len(labels)} parcels')

    elif atlas == 'custom':
        if atlas_path is None:
            raise ValueError(
                'atlas_path must be provided when atlas="custom".')
        atlas_img = nib.load(str(atlas_path))
        unique = np.unique(atlas_img.get_fdata())
        n = int((unique != 0).sum())
        labels = [f'ROI_{i+1:03d}' for i in range(n)]
        print(f'  [atlas] Custom ({atlas_path}): {len(labels)} parcels')

    else:
        raise ValueError(
            f'Unknown atlas "{atlas}". '
            'Choose from: schaefer, destrieux, custom.')

    if _logger:
        _logger.log_step(
            'get_parcellation',
            atlas      = atlas,
            n_rois     = len(labels),
            atlas_path = str(atlas_path) if atlas_path else 'nilearn built-in',
            img_shape  = str(atlas_img.shape),
            vox_size_mm= str(np.round(
                np.sqrt((atlas_img.affine[:3, :3] ** 2).sum(axis=0)), 2
            ).tolist()),
        )

    return atlas_img, labels


# =============================================================================
# Analysis
# =============================================================================

@_register
def extract_time_series(bold_path, atlas_img, confounds_array, t_r,
                        standardize=True, detrend=True,
                        low_pass=0.10, high_pass=0.01):
    """
    Extract parcel-averaged BOLD time series with confound regression.

    Wraps ``nilearn.maskers.NiftiLabelsMasker``.  In a single call it:
    resamples the parcellation to BOLD resolution, averages all voxels
    within each parcel, regresses out the confound matrix, and applies
    bandpass filtering.

    Parameters
    ----------
    bold_path : str or Path
        Path to the preprocessed BOLD NIfTI (fMRIPrep space-MNI output).

    atlas_img : nibabel.Nifti1Image
        Parcellation label image.  Obtain from ``get_parcellation()``.

    confounds_array : np.ndarray, shape (T, n_regressors)
        Confound matrix to regress out.  Obtain from ``load_confounds()``.
        Pass ``None`` to skip confound regression (not recommended).

    t_r : float
        Repetition time in seconds.  Required for bandpass filtering.

    standardize : bool
        Z-score each parcel's time series after extraction (mean 0, SD 1).
        Recommended when computing correlations.

    detrend : bool
        Remove the linear trend from each time series before regression.
        Should almost always be ``True``.

    low_pass : float or None
        Low-pass filter cutoff in Hz.  Removes high-frequency noise.
        Typical resting-state value: 0.10 Hz.  Set to ``None`` to skip.

    high_pass : float or None
        High-pass filter cutoff in Hz.  Removes slow drift.
        Typical resting-state value: 0.01 Hz.  Set to ``None`` to skip.

    Returns
    -------
    time_series : np.ndarray, shape (T, n_rois)
        Parcel-averaged, confound-regressed, filtered BOLD time series.

    Notes
    -----
    - ``resampling_target='labels'`` means the BOLD image is resampled to
      the atlas resolution (2 mm), not vice versa.  This is slightly
      faster than resampling the atlas to the native BOLD resolution.
    - A nilearn cache is written to ``./nilearn_cache/`` to speed up
      repeated calls with the same data.
    - Bandpass filtering is applied *after* confound regression, which is
      the recommended order (Lindquist et al. 2019).
    - For task fMRI, some researchers set ``low_pass=None`` because the
      task signal contains power above 0.1 Hz.

    Examples
    --------
    Resting-state with default bandpass:

    >>> ts = extract_time_series(BOLD_PATH, atlas_img, conf, TR)
    >>> print(ts.shape)   # (T, 200) for Schaefer-200

    Task fMRI — suppress low-pass to keep task-related high-frequency:

    >>> ts = extract_time_series(
    ...     BOLD_PATH, atlas_img, conf, TR,
    ...     low_pass=None, high_pass=0.01,
    ... )

    No filtering (e.g. for ICA or time-frequency analysis):

    >>> ts = extract_time_series(
    ...     BOLD_PATH, atlas_img, conf, TR,
    ...     low_pass=None, high_pass=None,
    ... )

    See also
    --------
    load_confounds    : Produce the ``confounds_array`` argument.
    get_parcellation  : Produce the ``atlas_img`` argument.
    compute_fc_matrix : Pass ``time_series`` to compute connectivity.
    plot_time_series  : Inspect the extracted time series visually.
    """
    masker = NiftiLabelsMasker(
        labels_img       = atlas_img,
        standardize      = standardize,
        detrend          = detrend,
        low_pass         = low_pass,
        high_pass        = high_pass,
        t_r              = t_r,
        resampling_target= 'labels',
        memory           = './nilearn_cache',
        verbose          = 1,
    )
    time_series = masker.fit_transform(bold_path, confounds=confounds_array)
    print(f'  [extraction] Time series shape: {time_series.shape}  (T × ROIs)')

    if _logger:
        _logger.log_step(
            'extract_time_series',
            bold_path        = str(bold_path),
            n_volumes        = time_series.shape[0],
            n_rois           = time_series.shape[1],
            standardize      = standardize,
            detrend          = detrend,
            low_pass_hz      = low_pass,
            high_pass_hz     = high_pass,
            t_r_s            = t_r,
            resampling_target= 'labels',
            confounds_shape  = (str(confounds_array.shape)
                                if confounds_array is not None else 'None'),
        )

    return time_series


@_register
def compute_fc_matrix(time_series, fisher_z=True):
    """
    Compute a parcel × parcel functional connectivity (FC) matrix.

    Calculates the Pearson correlation between every pair of ROI time
    series and optionally applies the Fisher r-to-z transform to
    normalise the distribution for group-level parametric tests.

    Parameters
    ----------
    time_series : np.ndarray, shape (T, n_rois)
        Parcel-averaged BOLD time series.  Obtain from
        ``extract_time_series()``.

    fisher_z : bool
        If ``True`` (default), apply the Fisher transform:
        ``z = arctanh(r)``.

        Why use Fisher z?
          - Pearson r is bounded in [-1, 1] and its sampling distribution
            is skewed, especially for large |r|.
          - arctanh(r) is approximately normally distributed with variance
            1/(T - 3), making it suitable for t-tests and GLMs.
          - Always use Fisher z before group-level statistics.

    Returns
    -------
    fc_matrix : np.ndarray, shape (n_rois, n_rois)
        Symmetric FC matrix.  Diagonal is set to 0 before transformation.

        If ``fisher_z=True``: values are Fisher z-scores (unbounded).
        If ``fisher_z=False``: values are Pearson r (range [-1, 1]).

    Notes
    -----
    - Pearson r values are clipped to ``[-1+ε, 1-ε]`` before ``arctanh``
      to avoid ±∞ (can occur due to floating-point precision).
    - The diagonal is zeroed before the Fisher transform.  This avoids
      ``arctanh(1) = ∞`` on the diagonal.
    - ``np.corrcoef`` operates on the transposed matrix so each *row* is
      treated as a variable.

    Examples
    --------
    Fisher z (recommended for group analysis):

    >>> fc_z = compute_fc_matrix(time_series, fisher_z=True)
    >>> print(fc_z.shape)   # (200, 200)

    Raw Pearson r (e.g. for single-subject visualisation):

    >>> fc_r = compute_fc_matrix(time_series, fisher_z=False)
    >>> print(fc_r.min(), fc_r.max())

    Convert back from z to r for display:

    >>> fc_r = np.tanh(fc_z)

    See also
    --------
    extract_time_series : Produce the ``time_series`` argument.
    save_fc_matrix      : Save the returned matrix.
    plot_fc_matrix      : Visualise the returned matrix.
    summarise_fc        : Print summary statistics.
    """
    fc_matrix = np.corrcoef(time_series.T)
    fc_matrix = np.clip(fc_matrix, -1 + 1e-7, 1 - 1e-7)
    np.fill_diagonal(fc_matrix, 0)

    if fisher_z:
        fc_matrix = np.arctanh(fc_matrix)
        print(f'  [FC] Fisher z-matrix: {fc_matrix.shape}  '
              f'(min={fc_matrix.min():.3f}, max={fc_matrix.max():.3f})')
    else:
        print(f'  [FC] Pearson r-matrix: {fc_matrix.shape}  '
              f'(min={fc_matrix.min():.3f}, max={fc_matrix.max():.3f})')

    if _logger:
        mask = np.ones_like(fc_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        off_diag = fc_matrix[mask]
        _logger.log_step(
            'compute_fc_matrix',
            matrix_shape  = str(fc_matrix.shape),
            fisher_z      = fisher_z,
            value_min     = round(float(off_diag.min()), 4),
            value_max     = round(float(off_diag.max()), 4),
            value_mean    = round(float(off_diag.mean()), 4),
            value_std     = round(float(off_diag.std()), 4),
            pct_positive  = round(float(100 * (off_diag > 0).mean()), 1),
            pct_negative  = round(float(100 * (off_diag < 0).mean()), 1),
        )

    return fc_matrix


# =============================================================================
# Output
# =============================================================================

@_register
def save_fc_matrix(fc_matrix, labels, output_dir, filename='fc_matrix.csv'):
    """
    Save an FC matrix as a labelled CSV file.

    Rows and columns are named by ROI label, producing a square,
    symmetric CSV that any analysis tool (R, MATLAB, Python, Excel) can
    read back directly.

    Parameters
    ----------
    fc_matrix : np.ndarray, shape (n_rois, n_rois)
        FC matrix returned by ``compute_fc_matrix()``.

    labels : list of str
        ROI names, length must equal ``n_rois``.
        Obtain from ``get_parcellation()``.

    output_dir : str or Path
        Directory in which to save the file.  Created automatically if
        it does not exist.

    filename : str
        Output filename, including ``.csv`` extension.
        Default: ``'fc_matrix.csv'``.
        Tip: encode subject / session / atlas in the name, e.g.
        ``'sub-01_ses-01_atlas-schaefer200_fc-fisherz.csv'``.

    Returns
    -------
    out_path : str
        Absolute path to the saved file.

    Notes
    -----
    - The CSV includes row and column headers (ROI names), so reading
      it back with ``pd.read_csv(..., index_col=0)`` returns a
      labelled DataFrame.
    - Values are stored at full float64 precision.

    Examples
    --------
    >>> out = save_fc_matrix(fc_z, labels, './fc_output',
    ...                       filename='sub-01_fc-fisherz.csv')
    >>> print(out)
    ./fc_output/sub-01_fc-fisherz.csv

    Read back in Python:

    >>> df = pd.read_csv(out, index_col=0)
    >>> df.shape   # (200, 200)

    Read back in R:

    >>> # fc <- read.csv('sub-01_fc-fisherz.csv', row.names=1)

    See also
    --------
    compute_fc_matrix : Produce the ``fc_matrix`` argument.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = fc_matrix.shape[0]
    labels = list(labels)
    if len(labels) != n:
        if len(labels) > n:
            # Background label is always FIRST — take the last n entries
            # (filtering by content misses non-empty strings like 'Background')
            dropped = labels[:len(labels) - n]
            labels  = labels[len(labels) - n:]
            print(f'  [save] NOTE: {len(labels) + len(dropped)} labels for a {n}×{n} matrix — '
                  f'dropped first {len(dropped)} background label(s): {dropped}')
        else:
            # Fewer labels than ROIs — pad with generic names and warn
            print(f'  [save] WARNING: only {len(labels)} labels for a {n}×{n} matrix — '
                  f'padding missing entries with ROI_NNN placeholders.')
            labels += [f'ROI_{i:03d}' for i in range(len(labels), n)]

    df = pd.DataFrame(fc_matrix, index=labels, columns=labels)
    out_path = str(output_dir / filename)
    df.to_csv(out_path)
    print(f'  [save] Saved → {out_path}')

    if _logger:
        _logger.log_step(
            'save_fc_matrix',
            csv_path   = out_path,
            n_rois     = fc_matrix.shape[0],
            n_cells    = int(fc_matrix.shape[0] ** 2),
        )
        # Auto-save log with matching filename stem (.log instead of .csv)
        log_stem = Path(filename).stem
        log_path = output_dir / f'{log_stem}.log'
        _logger.save(log_path)

    return out_path


# =============================================================================
# Visualisation
# =============================================================================

@_register
def plot_fc_matrix(fc_matrix, labels=None,
                   title='Functional Connectivity Matrix',
                   cmap='RdBu_r', vmax=None, figsize=(10, 8),
                   output_path=None):
    """
    Plot a heatmap of a parcellation × parcellation FC matrix.

    Parameters
    ----------
    fc_matrix : np.ndarray, shape (n_rois, n_rois)
        FC matrix (Fisher z or Pearson r).

    labels : list of str or None
        ROI labels.  If provided *and* ``n_rois ≤ 50``, labels are
        drawn on both axes.  For larger matrices labels are hidden to
        keep the figure legible.

    title : str
        Figure title.

    cmap : str
        Matplotlib colormap name.  ``'RdBu_r'`` (blue = negative,
        red = positive) is conventional for FC matrices.

    vmax : float or None
        Colour scale maximum (symmetric around 0).  If ``None``,
        defaults to the 95th percentile of ``|fc_matrix|``.

    figsize : tuple of (width, height)
        Figure size in inches.

    output_path : str or None
        If provided, save the figure to this path (PNG or PDF).
        ``None`` = do not save.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Examples
    --------
    Default plot (large atlas, no labels):

    >>> fig = plot_fc_matrix(fc_z, title='sub-01 resting-state FC')
    >>> plt.show()

    Small atlas with labels:

    >>> fig = plot_fc_matrix(fc_z, labels=roi_labels,
    ...                       title='Destrieux FC', figsize=(12, 10))

    Save to disk:

    >>> fig = plot_fc_matrix(fc_z, output_path='./fc_output/fc_plot.png')

    See also
    --------
    compute_fc_matrix : Produce the ``fc_matrix`` argument.
    """
    if vmax is None:
        vmax = float(np.percentile(np.abs(fc_matrix), 95))

    n = fc_matrix.shape[0]
    show_labels = (labels is not None) and (n <= 50)
    tick_labels = labels if show_labels else False

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        fc_matrix,
        ax          = ax,
        cmap        = cmap,
        vmin        = -vmax,
        vmax        =  vmax,
        square      = True,
        xticklabels = tick_labels,
        yticklabels = tick_labels,
        cbar_kws    = {'shrink': 0.7, 'label': 'Fisher z'},
    )
    ax.set_title(title, fontsize=13, pad=12)
    if show_labels:
        ax.tick_params(axis='x', labelsize=7, rotation=90)
        ax.tick_params(axis='y', labelsize=7, rotation=0)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'  [plot] Saved → {output_path}')

    return fig


@_register
def plot_time_series(time_series, t_r, n_rois=8, labels=None,
                     title='Parcel time series', figsize=(12, 3.5)):
    """
    Plot a sample of parcel-averaged BOLD time series.

    Parameters
    ----------
    time_series : np.ndarray, shape (T, n_rois_total)
        Time series returned by ``extract_time_series()``.

    t_r : float
        Repetition time in seconds (used to label the x-axis).

    n_rois : int
        Number of parcels to plot (first ``n_rois`` columns).
        Default: 8.

    labels : list of str or None
        ROI labels.  If provided, used in the legend.

    title : str
        Figure title.

    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Examples
    --------
    >>> fig = plot_time_series(time_series, TR)
    >>> plt.show()

    Show 12 parcels with labels:

    >>> fig = plot_time_series(time_series, TR,
    ...                        n_rois=12, labels=roi_labels)

    See also
    --------
    extract_time_series : Produce the ``time_series`` argument.
    """
    n_plot  = min(n_rois, time_series.shape[1])
    t_axis  = np.arange(time_series.shape[0]) * t_r

    fig, ax = plt.subplots(figsize=figsize)
    offset  = 0
    for i in range(n_plot):
        lbl = labels[i] if labels else f'ROI {i+1}'
        ax.plot(t_axis, time_series[:, i] + offset, lw=0.75, label=lbl)
        offset += 4

    ax.set(xlabel='Time (s)', yticks=[], title=title)
    if labels and n_plot <= 12:
        ax.legend(fontsize=7, loc='upper right', ncol=2)
    plt.tight_layout()
    return fig


@_register
def plot_fd_trace(confounds_path, scrub_threshold=None,
                  t_r=None, figsize=(11, 2.5)):
    """
    Plot the framewise displacement (FD) trace for a single run.

    Parameters
    ----------
    confounds_path : str or Path
        Path to the fMRIPrep confounds TSV for the run.

    scrub_threshold : float or None
        If set, draw a horizontal dashed line at this FD value and mark
        volumes that exceed it with red dots.

    t_r : float or None
        Repetition time in seconds.  If provided, x-axis is in seconds;
        otherwise x-axis is in volumes.

    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    mean_fd : float
        Mean framewise displacement across all volumes.
    pct_scrubbed : float
        Percentage of volumes that would be scrubbed at ``scrub_threshold``
        (0.0 if ``scrub_threshold`` is None).

    Notes
    -----
    - FD is computed by fMRIPrep as the sum of absolute displacements of
      the 6 rigid-body motion parameters (converted to mm).
    - Power et al. (2012) recommend a threshold of 0.5 mm for
      resting-state and 0.9 mm for task fMRI.

    Examples
    --------
    >>> fig, mean_fd, pct = plot_fd_trace(
    ...     CONFOUNDS_PATH, scrub_threshold=0.5, t_r=TR
    ... )
    >>> print(f'Mean FD: {mean_fd:.3f} mm | {pct:.1f}% scrubbed')

    See also
    --------
    load_confounds : Scrubbing is applied during confound loading.
    """
    df = pd.read_csv(confounds_path, sep='\t')
    if 'framewise_displacement' not in df.columns:
        raise ValueError(
            'framewise_displacement column not found in confounds file.')

    fd = df['framewise_displacement'].fillna(0).values
    x  = np.arange(len(fd)) * t_r if t_r else np.arange(len(fd))
    xlabel = 'Time (s)' if t_r else 'Volume'

    scrubbed_idx = []
    pct_scrubbed = 0.0
    if scrub_threshold is not None:
        scrubbed_idx = np.where(fd > scrub_threshold)[0]
        pct_scrubbed = 100 * len(scrubbed_idx) / len(fd)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, fd, lw=0.8, color='steelblue', label='FD')
    if scrub_threshold is not None:
        ax.axhline(scrub_threshold, color='tomato', lw=1.2, ls='--',
                   label=f'Threshold ({scrub_threshold} mm)')
        if len(scrubbed_idx):
            ax.scatter(x[scrubbed_idx], fd[scrubbed_idx],
                       color='tomato', s=14, zorder=5)
    ax.set(xlabel=xlabel, ylabel='FD (mm)', title='Framewise Displacement')
    ax.legend(fontsize=9)
    plt.tight_layout()

    mean_fd = float(fd.mean())
    print(f'  [FD] Mean: {mean_fd:.3f} mm | '
          f'Max: {fd.max():.3f} mm | '
          f'Scrubbed: {len(scrubbed_idx)}/{len(fd)} '
          f'({pct_scrubbed:.1f}%)')

    if _logger:
        _logger.log_step(
            'plot_fd_trace',
            confounds_path   = str(confounds_path),
            n_volumes        = len(fd),
            mean_fd_mm       = round(mean_fd, 4),
            max_fd_mm        = round(float(fd.max()), 4),
            scrub_threshold  = scrub_threshold,
            volumes_flagged  = len(scrubbed_idx),
            pct_scrubbed     = round(pct_scrubbed, 1),
        )

    return fig, mean_fd, pct_scrubbed


# =============================================================================
# Utilities
# =============================================================================

@_register
def summarise_fc(fc_matrix, label='FC matrix'):
    """
    Print a summary of key statistics for an FC matrix.

    Parameters
    ----------
    fc_matrix : np.ndarray, shape (n_rois, n_rois)
        FC matrix (Fisher z or Pearson r).  The diagonal is ignored.

    label : str
        A descriptive label printed in the header.

    Returns
    -------
    stats : dict
        Dictionary with keys: ``mean``, ``std``, ``min``, ``max``,
        ``pct_positive``, ``pct_negative``.

    Examples
    --------
    >>> stats = summarise_fc(fc_z, label='sub-01 resting-state')
    >>> print(stats['mean'])

    See also
    --------
    compute_fc_matrix : Produce the ``fc_matrix`` argument.
    """
    mask = np.ones_like(fc_matrix, dtype=bool)
    np.fill_diagonal(mask, False)           # exclude diagonal
    vals = fc_matrix[mask]

    stats = dict(
        mean        = float(vals.mean()),
        std         = float(vals.std()),
        min         = float(vals.min()),
        max         = float(vals.max()),
        pct_positive= float(100 * (vals > 0).mean()),
        pct_negative= float(100 * (vals < 0).mean()),
    )

    n = fc_matrix.shape[0]
    print(f'\n  ── {label} ({n}×{n}) ─────────────────────────────')
    print(f'  Mean ± SD  : {stats["mean"]:+.4f} ± {stats["std"]:.4f}')
    print(f'  Range      : [{stats["min"]:+.4f},  {stats["max"]:+.4f}]')
    print(f'  % positive : {stats["pct_positive"]:.1f}%')
    print(f'  % negative : {stats["pct_negative"]:.1f}%')
    print()
    return stats


# =============================================================================
# Alignment QC
# =============================================================================

@_register
def check_alignment(bold_path, atlas_img, t_r=None):
    """
    Check that a BOLD image and a parcellation atlas share the same template space.

    Misaligned inputs produce silently wrong time series: the masker will
    resample one image to the other's grid, but if the images are in
    *different coordinate spaces* (e.g. BOLD in native T1w space, atlas in
    MNI152) the resampling will be anatomically meaningless.

    This function runs five checks and returns a structured report:

    1. **Voxel size** — do both images have the same voxel dimensions?
    2. **Affine origin** — are the scanner-coordinate origins compatible?
    3. **Field of view** — does the BOLD volume fully cover the atlas extent?
    4. **Orientation** — do the axis-direction cosines agree?
    5. **Space label** — do the NIfTI ``sform_code`` / ``qform_code``
       headers indicate the same coordinate system?

    Parameters
    ----------
    bold_path : str or Path
        Path to the preprocessed BOLD NIfTI.

    atlas_img : nibabel.Nifti1Image
        Parcellation label image from ``get_parcellation()``.

    t_r : float or None
        Repetition time in seconds.  Only used for display; does not
        affect any check.

    Returns
    -------
    report : dict
        Keys: ``'voxel_size'``, ``'affine_origin'``, ``'field_of_view'``,
        ``'orientation'``, ``'space_label'``, each mapping to a sub-dict::

            {
                'pass': bool,
                'bold': <value>,
                'atlas': <value>,
                'note': str,
            }

        Also includes ``'overall_pass': bool`` (True only when ALL checks
        pass) and ``'needs_resampling': bool``.

    Notes
    -----
    - A failed **space label** check is a hard error — the images cannot
      be meaningfully combined regardless of resampling.
    - Failed **voxel size** or **affine origin** checks mean resampling
      will occur.  ``NiftiLabelsMasker`` with
      ``resampling_target='labels'`` handles this automatically, but it
      adds compute time and mild interpolation error.
    - Failed **field of view** means some atlas parcels will fall outside
      the BOLD volume and will produce all-NaN time series.  Inspect
      which parcels are affected before proceeding.
    - fMRIPrep outputs (``space-MNI152NLin2009cAsym``) are always safe to
      combine with nilearn's default MNI152 atlases.

    Examples
    --------
    Basic usage:

    >>> report = check_alignment(BOLD_PATH, atlas_img, t_r=TR)

    Check a specific result:

    >>> if not report['field_of_view']['pass']:
    ...     print('WARNING: atlas extends outside the BOLD field of view!')

    Fail fast if space labels disagree:

    >>> assert report['space_label']['pass'], report['space_label']['note']

    See also
    --------
    get_parcellation : Produce the ``atlas_img`` argument.
    plot_alignment   : Visual overlay check.
    """
    bold_img = nib.load(str(bold_path))

    # ── 1. Voxel sizes ────────────────────────────────────────────────────────
    bold_vox  = np.round(np.sqrt((bold_img.affine[:3, :3] ** 2).sum(axis=0)), 2)
    atlas_vox = np.round(np.sqrt((atlas_img.affine[:3, :3] ** 2).sum(axis=0)), 2)
    vox_pass  = np.allclose(bold_vox, atlas_vox, atol=0.1)

    # ── 2. Affine origin (translation column) ────────────────────────────────
    bold_orig  = bold_img.affine[:3, 3]
    atlas_orig = atlas_img.affine[:3, 3]
    orig_pass  = np.allclose(bold_orig, atlas_orig, atol=2.0)  # 2 mm tolerance

    # ── 3. Field of view — does BOLD cover the atlas? ─────────────────────────
    # Map atlas corner voxels to world coords, then to BOLD voxel coords
    atlas_shape = np.array(atlas_img.shape[:3])
    corners = np.array([[i, j, k]
                        for i in [0, atlas_shape[0]-1]
                        for j in [0, atlas_shape[1]-1]
                        for k in [0, atlas_shape[2]-1]], dtype=float)
    corners_world = nib.affines.apply_affine(atlas_img.affine, corners)
    bold_inv       = np.linalg.inv(bold_img.affine)
    corners_bold   = nib.affines.apply_affine(bold_inv, corners_world)
    bold_shape     = np.array(bold_img.shape[:3])
    in_fov  = np.all((corners_bold >= 0) & (corners_bold < bold_shape - 1),
                     axis=1)
    fov_pass = bool(in_fov.all())
    n_outside = int((~in_fov).sum())

    # ── 4. Orientation (axis direction cosines) ───────────────────────────────
    def _cosines(affine):
        col_norms = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        return affine[:3, :3] / col_norms
    bold_cos  = _cosines(bold_img.affine)
    atlas_cos = _cosines(atlas_img.affine)
    orient_pass = np.allclose(bold_cos, atlas_cos, atol=0.01)

    # ── 5. Space label (sform / qform codes) ─────────────────────────────────
    def _space_code(img):
        hdr = img.header
        sc = int(getattr(hdr, 'get_sform', lambda **k: {})(coded=True)[1]
                 if hasattr(hdr, 'get_sform') else 0)
        qc = int(getattr(hdr, 'get_qform', lambda **k: {})(coded=True)[1]
                 if hasattr(hdr, 'get_qform') else 0)
        return max(sc, qc)   # 1 = scanner, 4 = MNI152

    bold_sc  = _space_code(bold_img)
    atlas_sc = _space_code(atlas_img)
    _CODE_NAMES = {0: 'unknown', 1: 'scanner/native', 2: 'aligned',
                   3: 'Talairach', 4: 'MNI152'}
    space_pass = (bold_sc == atlas_sc) or (bold_sc in (0, 1, 4) and atlas_sc in (0, 1, 4))

    # ── Build report ──────────────────────────────────────────────────────────
    report = {
        'voxel_size': {
            'pass':  vox_pass,
            'bold':  bold_vox.tolist(),
            'atlas': atlas_vox.tolist(),
            'note':  'OK' if vox_pass else
                     f'Mismatch ({bold_vox} vs {atlas_vox} mm). '
                     'NiftiLabelsMasker will resample automatically.',
        },
        'affine_origin': {
            'pass':  orig_pass,
            'bold':  np.round(bold_orig, 1).tolist(),
            'atlas': np.round(atlas_orig, 1).tolist(),
            'note':  'OK' if orig_pass else
                     'Origins differ by > 2 mm. Check that both images '
                     'are in the same template space.',
        },
        'field_of_view': {
            'pass':  fov_pass,
            'bold':  bold_shape.tolist(),
            'atlas': atlas_shape.tolist(),
            'note':  'OK' if fov_pass else
                     f'{n_outside}/8 atlas corners fall outside the BOLD '
                     'volume. Some parcels may produce NaN time series.',
        },
        'orientation': {
            'pass':  orient_pass,
            'bold':  bold_cos.round(3).tolist(),
            'atlas': atlas_cos.round(3).tolist(),
            'note':  'OK' if orient_pass else
                     'Axis orientations differ. Images may be in different '
                     'coordinate frames (e.g. RAS vs LAS).',
        },
        'space_label': {
            'pass':  space_pass,
            'bold':  _CODE_NAMES.get(bold_sc, bold_sc),
            'atlas': _CODE_NAMES.get(atlas_sc, atlas_sc),
            'note':  'OK' if space_pass else
                     f'Space codes differ: BOLD={_CODE_NAMES.get(bold_sc, bold_sc)}, '
                     f'Atlas={_CODE_NAMES.get(atlas_sc, atlas_sc)}. '
                     'This is likely a hard mismatch — verify template space.',
        },
    }
    report['overall_pass']     = all(v['pass'] for v in report.values()
                                     if isinstance(v, dict))
    report['needs_resampling'] = not vox_pass or not orig_pass

    # ── Pretty-print report ───────────────────────────────────────────────────
    STATUS = {True: '✓ PASS', False: '✗ FAIL'}
    W = 68
    print('=' * W)
    print('  BOLD ↔ Atlas Alignment Report')
    print('=' * W)
    print(f'  BOLD  : {bold_path}')
    print(f'  Atlas : {atlas_img.shape[:3]}  vox={atlas_vox}')
    if t_r:
        print(f'  TR    : {t_r} s  |  Volumes: {bold_img.shape[-1]}')
    print('-' * W)
    checks = ['voxel_size', 'affine_origin', 'field_of_view',
              'orientation', 'space_label']
    labels = ['Voxel size      ', 'Affine origin   ', 'Field of view   ',
              'Orientation     ', 'Space label     ']
    for key, lbl in zip(checks, labels):
        r = report[key]
        print(f'  {STATUS[r["pass"]]}  {lbl}  {r["note"]}')
    print('-' * W)
    overall = report['overall_pass']
    print(f'  Overall: {"ALL CHECKS PASSED" if overall else "⚠  ISSUES FOUND — see notes above"}')
    if report['needs_resampling']:
        print('  Note  : NiftiLabelsMasker will resample automatically '
              '(resampling_target="labels").')
    print('=' * W)

    if _logger:
        _logger.log_step(
            'check_alignment',
            bold_path        = str(bold_path),
            overall_pass     = report['overall_pass'],
            needs_resampling = report['needs_resampling'],
            voxel_size       = report['voxel_size']['note'],
            affine_origin    = report['affine_origin']['note'],
            field_of_view    = report['field_of_view']['note'],
            orientation      = report['orientation']['note'],
            space_label      = report['space_label']['note'],
        )

    return report


@_register
def plot_alignment(bold_path, atlas_img, n_cuts=6,
                   display_mode='z', figsize=(14, 4), vol_idx=0):
    """
    Visually verify that the BOLD image and atlas are in the same space.

    Loads a single BOLD volume as the background image and overlays the
    parcellation borders on top.  Misaligned images will show atlas parcels
    that do not correspond to the brain regions visible in the background.

    Parameters
    ----------
    bold_path : str or Path
        Path to the preprocessed BOLD NIfTI.

    atlas_img : nibabel.Nifti1Image
        Parcellation label image from ``get_parcellation()``.

    n_cuts : int
        Number of slice positions to display.  Default: 6.

    display_mode : {'z', 'x', 'y', 'ortho', 'tiled'}
        Slice direction.  ``'z'`` (axial) is most informative for
        checking cortical coverage.  Use ``'ortho'`` for a quick
        3-plane view.

    figsize : tuple
        Figure size passed to matplotlib.  Ignored when
        ``display_mode='ortho'`` (nilearn controls the size).

    vol_idx : int
        Index of the single volume to use as the background.  Default: 0
        (first volume).  Any volume works for alignment checking — the
        spatial coordinates are what matter, not the signal intensity.

    Returns
    -------
    display : nilearn display object
        Returned by ``nilearn.plotting.plot_roi``; call
        ``display.savefig('path.png')`` to save.

    Notes
    -----
    - Only a **single volume** is loaded from disk (via nibabel proxy
      indexing), keeping memory usage minimal regardless of run length.
      This replaces the previous ``mean_img()`` approach which loaded the
      entire 4-D BOLD into RAM.
    - A single volume is sufficient for alignment QC — the voxel grid and
      affine are the same for every volume in the run.
    - If the images are in the same space, atlas parcel borders should
      follow gyral anatomy in the BOLD background.
    - If parcels appear shifted, rotated, or at the wrong scale, the
      images are misaligned and extraction will produce wrong results.

    Examples
    --------
    Axial slice overlay (default, first volume):

    >>> display = plot_alignment(BOLD_PATH, atlas_img)

    Orthogonal 3-plane view:

    >>> display = plot_alignment(BOLD_PATH, atlas_img,
    ...                          display_mode='ortho')

    Coronal slices, more cuts:

    >>> display = plot_alignment(BOLD_PATH, atlas_img,
    ...                          display_mode='y', n_cuts=8)

    Save the overlay:

    >>> display.savefig('./fc_output/alignment_check.png', dpi=150)

    See also
    --------
    check_alignment : Numeric alignment checks.
    get_parcellation : Produce the ``atlas_img`` argument.
    """
    from nilearn import image as nli

    # Load a single volume via proxy indexing — only one frame is read
    # from disk, so memory use is ~1/T of loading the full 4-D BOLD.
    print(f'  [alignment] Loading volume {vol_idx} (single frame) …')
    ref_vol = nli.index_img(str(bold_path), vol_idx)

    print(f'  [alignment] Plotting {display_mode} overlay ({n_cuts} cuts) …')
    display = plotting.plot_roi(
        atlas_img,
        bg_img       = ref_vol,
        display_mode = display_mode,
        cut_coords   = n_cuts if display_mode != 'ortho' else None,
        colorbar     = False,
        alpha        = 0.45,
        title        = 'Alignment check: atlas parcels on BOLD\n'
                       '(parcels should follow gyral anatomy)',
        cmap         = 'tab20',
    )
    plt.gcf().set_size_inches(figsize)
    return display


# =============================================================================
# gPPI Analysis
# =============================================================================

@_register
def load_events(events_path, conditions=None):
    """
    Load a BIDS events TSV file for task-based connectivity analysis.

    Reads the timing file produced by fMRIPrep or supplied with your BIDS
    dataset, optionally filtering to a subset of trial types for focused
    analysis.

    Parameters
    ----------
    events_path : str or Path
        Path to the BIDS ``_events.tsv`` file for the run.

    conditions : list of str or None
        If provided, only rows whose ``trial_type`` matches one of these
        strings are kept.  ``None`` keeps all conditions.

    Returns
    -------
    events_df : pd.DataFrame
        DataFrame with at minimum columns ``'onset'``, ``'duration'``,
        ``'trial_type'``.  Ready to pass to ``build_hrf_regressors()``.

    conditions_found : list of str
        Sorted list of unique ``trial_type`` values in the returned DataFrame.

    Notes
    -----
    - BIDS requires ``'onset'`` and ``'duration'`` columns (in seconds).
    - ``'trial_type'`` is the standard column for condition labels.
    - If ``'trial_type'`` is absent the entire run is treated as one
      condition called ``'task'``.
    - Rows with NaN onsets or durations are silently dropped.

    Examples
    --------
    Load all events:

    >>> events_df, conds = load_events(EVENTS_PATH)
    >>> print(conds)   # ['CONDITION_A', 'CONDITION_B']

    Keep only two conditions:

    >>> events_df, conds = load_events(
    ...     EVENTS_PATH, conditions=['CONDITION_A', 'CONDITION_B']
    ... )

    See also
    --------
    build_hrf_regressors : Convolve the returned events with an HRF.
    compute_gppi_matrix  : Full gPPI pipeline.
    """
    df = pd.read_csv(events_path, sep='\t')

    for col in ('onset', 'duration'):
        if col not in df.columns:
            raise ValueError(
                f"Events file missing required BIDS column '{col}': {events_path}"
            )

    if 'trial_type' not in df.columns:
        print('  [events] No trial_type column — treating entire run as "task".')
        df['trial_type'] = 'task'

    n_before = len(df)
    df = df.dropna(subset=['onset', 'duration']).copy()
    if len(df) < n_before:
        print(f'  [events] Dropped {n_before - len(df)} row(s) with NaN onset/duration.')

    if conditions is not None:
        df = df[df['trial_type'].isin(conditions)].copy()

    conditions_found = sorted(df['trial_type'].unique())
    print(f'  [events] {len(df)} event(s) | '
          f'{len(conditions_found)} condition(s): {conditions_found}')

    if _logger:
        _logger.log_step(
            'load_events',
            events_path       = str(events_path),
            n_events          = len(df),
            conditions_found  = conditions_found,
            conditions_filter = conditions,
        )

    return df, conditions_found


@_register
def build_hrf_regressors(events_df, t_r, n_scans,
                          conditions=None, hrf_model='spm'):
    """
    Convolve task event timing with a haemodynamic response function (HRF).

    For each condition, creates a continuous time-series regressor by
    convolving a boxcar function (1 during each event, 0 otherwise) with
    the canonical HRF.  These regressors represent the *expected* BOLD
    response to each condition — the "psychological" component of a gPPI
    model.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events DataFrame from ``load_events()``, with columns
        ``'onset'``, ``'duration'``, ``'trial_type'``.

    t_r : float
        Repetition time in seconds.

    n_scans : int
        Number of volumes in the BOLD run.

    conditions : list of str or None
        Conditions to model.  ``None`` uses all unique ``trial_type``
        values in ``events_df``.

    hrf_model : str
        HRF shape.  ``'spm'`` (double-gamma, default) is standard.
        Other nilearn options: ``'glover'``, ``'spm + derivative'``.

    Returns
    -------
    hrf_regs : dict
        Mapping ``condition_name → np.ndarray`` of shape ``(n_scans,)``.
        Values are normalised so the peak amplitude is approximately 1.

    Notes
    -----
    - The double-gamma SPM HRF peaks ~5–6 s after stimulus onset and has
      a post-stimulus undershoot returning to baseline ~20–25 s later.
    - In a gPPI model these regressors are the *psychological variable*
      (the task context that modulates connectivity).  They are NOT
      activation regressors; their role is to capture the expected
      task-evoked signal so the PPI interaction term reflects connectivity
      above and beyond the task's main effect.
    - Regressors are evaluated at TR-centred frame times:
      ``frame_times = t_r / 2 + np.arange(n_scans) * t_r``.

    Examples
    --------
    >>> hrf_regs = build_hrf_regressors(events_df, TR, n_scans=240)
    >>> print(list(hrf_regs.keys()))         # ['CONDITION_A', 'CONDITION_B']
    >>> print(hrf_regs['CONDITION_A'].shape) # (240,)

    Plot regressors:

    >>> t = np.arange(n_scans) * TR
    >>> for name, reg in hrf_regs.items():
    ...     plt.plot(t, reg, label=name)
    >>> plt.legend(); plt.xlabel('Time (s)'); plt.show()

    See also
    --------
    load_events         : Produce the events_df argument.
    build_gppi_design   : Consume these regressors to build a design matrix.
    compute_gppi_matrix : Full gPPI pipeline (calls this internally).
    """
    from nilearn.glm.first_level import compute_regressor

    # TR-centred frame times (middle of each volume acquisition window)
    frame_times = t_r / 2 + np.arange(n_scans) * t_r

    if conditions is None:
        conditions = sorted(events_df['trial_type'].unique())

    hrf_regs = {}
    for cond in conditions:
        cond_df = events_df[events_df['trial_type'] == cond]
        if len(cond_df) == 0:
            print(f'  [HRF] WARNING: no events for condition "{cond}" — skipping.')
            continue
        exp_condition = (
            cond_df['onset'].values.astype(float),
            cond_df['duration'].values.astype(float),
            np.ones(len(cond_df)),     # unit amplitudes
        )
        reg, _ = compute_regressor(
            exp_condition = exp_condition,
            hrf_model     = hrf_model,
            frame_times   = frame_times,
        )
        hrf_regs[cond] = reg.ravel()

    print(f'  [HRF] {len(hrf_regs)} regressor(s) built '
          f'(model: {hrf_model}  |  n_scans: {n_scans}  |  TR: {t_r} s)')

    if _logger:
        _logger.log_step(
            'build_hrf_regressors',
            n_conditions = len(hrf_regs),
            conditions   = list(hrf_regs.keys()),
            hrf_model    = hrf_model,
            n_scans      = n_scans,
            t_r          = t_r,
        )

    return hrf_regs


@_register
def build_gppi_design(seed_ts, hrf_regressors, confounds=None):
    """
    Assemble the gPPI design matrix for a single seed ROI.

    The design matrix contains four blocks of columns:

    1. **Physiological** — the seed ROI time series (``'seed'``).
    2. **Psychological** — one HRF-convolved task regressor per condition
       (``'psych_<condition>'``).
    3. **PPI interaction** — element-wise product seed × psychological for
       each condition (``'ppi_<condition>'``).
    4. **Nuisance** — confound regressors from ``load_confounds()`` plus
       an intercept.

    The PPI betas from the resulting OLS fit capture context-dependent
    connectivity: the change in coupling between the seed and each target
    region *specifically during the task condition*, beyond what can be
    explained by either the seed signal or the task activation alone.

    Parameters
    ----------
    seed_ts : np.ndarray, shape (T,)
        Mean BOLD time series of the seed ROI.  Use ``standardize=False``
        in ``extract_time_series()`` so beta magnitudes are in BOLD signal
        units rather than Z-scores.

    hrf_regressors : dict
        Output of ``build_hrf_regressors()``.
        Mapping ``condition_name → array`` of shape ``(T,)``.

    confounds : np.ndarray of shape (T, n_regressors) or None
        Confound matrix from ``load_confounds()``.  Added as nuisance
        regressors.  ``None`` skips confound regression (not recommended).

    Returns
    -------
    design_matrix : np.ndarray, shape (T, n_columns)
        Full design matrix ready for OLS.

    col_names : list of str
        Name of each column in ``design_matrix``.

    ppi_cols : list of int
        Integer column indices of the PPI terms.  Used by
        ``compute_gppi_matrix()`` to extract PPI betas.

    Notes
    -----
    - The seed time series is not re-standardised here.  Using raw
      amplitude keeps PPI betas interpretable as connectivity strength
      per unit seed signal.
    - The Friston (2012) gPPI formulation ideally deconvolves the seed
      and psychological regressors to neural space before multiplication,
      then reconvolves.  The BOLD-space approximation used here (direct
      multiplication) is standard for ROI-to-ROI gPPI and appropriate
      for block-design or slow event-related paradigms.
    - The intercept is always the last column.

    Examples
    --------
    >>> D, names, ppi_idx = build_gppi_design(
    ...     seed_ts        = time_series[:, 0],
    ...     hrf_regressors = hrf_regs,
    ...     confounds      = confounds_array,
    ... )
    >>> print(D.shape)      # (T, 1 + n_conds*2 + n_conf + 1)
    >>> print(names[:5])    # ['seed', 'psych_A', 'psych_B', 'ppi_A', 'ppi_B']
    >>> print(ppi_idx)      # [3, 4] for two conditions

    Visualise the design matrix:

    >>> plt.imshow(D, aspect='auto', cmap='RdBu_r')
    >>> plt.xticks(range(len(names)), names, rotation=90); plt.colorbar()

    See also
    --------
    build_hrf_regressors : Build the hrf_regressors argument.
    compute_gppi_matrix  : Calls this function for every seed.
    """
    T    = len(seed_ts)
    cols = {}

    # ── 1. Physiological (seed time series) ──────────────────────────────────
    cols['seed'] = seed_ts

    # ── 2. Psychological (HRF-convolved task regressors) ─────────────────────
    for cond, reg in hrf_regressors.items():
        cols[f'psych_{cond}'] = reg[:T]

    # ── 3. PPI interaction (seed × psychological, one per condition) ──────────
    for cond, reg in hrf_regressors.items():
        cols[f'ppi_{cond}'] = seed_ts * reg[:T]

    # ── 4. Nuisance confounds ─────────────────────────────────────────────────
    if confounds is not None:
        for i in range(confounds.shape[1]):
            cols[f'conf_{i:03d}'] = confounds[:, i]

    # ── Intercept (always last) ───────────────────────────────────────────────
    cols['intercept'] = np.ones(T)

    col_names     = list(cols.keys())
    design_matrix = np.column_stack(list(cols.values()))
    ppi_cols      = [i for i, n in enumerate(col_names) if n.startswith('ppi_')]

    return design_matrix, col_names, ppi_cols


@_register
def compute_gppi_matrix(time_series, events_df, t_r, confounds=None,
                         conditions=None, hrf_model='spm', fisher_z=False):
    """
    Compute a full ROI × ROI generalised PPI connectivity matrix.

    For each seed ROI, fits an OLS general linear model with physiological
    (seed), psychological (HRF-convolved task), and PPI interaction
    regressors against every target ROI simultaneously.  The gPPI
    connectivity estimate is the sum of PPI beta coefficients across all
    modelled conditions.

    Parameters
    ----------
    time_series : np.ndarray, shape (T, n_rois)
        Parcel time series from ``extract_time_series()``.

        Important: use ``standardize=False`` and ``low_pass=None``.
        Task responses contain power above 0.10 Hz, so low-pass filtering
        discards signal of interest; standardisation changes the beta scale.

    events_df : pd.DataFrame
        Events DataFrame from ``load_events()``.

    t_r : float
        Repetition time in seconds.

    confounds : np.ndarray of shape (T, n_regressors) or None
        Confound matrix from ``load_confounds()``.

    conditions : list of str or None
        Conditions to model as PPI terms.  ``None`` models all
        ``trial_type`` values in ``events_df``.

    hrf_model : str
        HRF model for psychological regressors.  Default: ``'spm'``.

    fisher_z : bool
        If ``True``, apply ``arctanh()`` to the PPI betas.  Unlike FC
        (Pearson r), gPPI betas are OLS regression coefficients and do
        not require Fisher-z for group-level tests.  Default: ``False``.

    Returns
    -------
    gppi_matrix : np.ndarray, shape (n_rois, n_rois)
        gPPI connectivity matrix.  ``gppi_matrix[i, j]`` is the summed
        PPI beta when ROI ``i`` is the seed and ROI ``j`` is the target.
        Diagonal is set to 0.

    Notes
    -----
    - Unlike FC (symmetric by definition), gPPI matrices can be
      asymmetric because the seed modulates the regression.  In practice
      the asymmetry is small; symmetrising with ``(M + M.T) / 2`` before
      group analysis is common.
    - ``np.linalg.lstsq(D, Y)`` solves all ``n_rois`` targets in a single
      call per seed, making the inner loop efficient.  Expect ~1–3 min for
      Schaefer-200 at 2 mm resolution.

    Examples
    --------
    All conditions:

    >>> gppi_mat = compute_gppi_matrix(ts, events_df, TR, confounds_array)

    Two specific conditions:

    >>> gppi_mat = compute_gppi_matrix(
    ...     ts, events_df, TR, confounds_array,
    ...     conditions=['CONDITION_A', 'CONDITION_B'],
    ... )

    Symmetrise for group analysis:

    >>> gppi_sym = (gppi_mat + gppi_mat.T) / 2

    See also
    --------
    extract_time_series    : Use standardize=False, low_pass=None for gPPI.
    load_events            : Produce the events_df argument.
    build_hrf_regressors   : Called internally.
    build_gppi_design      : Called internally for each seed.
    save_fc_matrix         : Save the result as a labelled CSV.
    plot_fc_matrix         : Plot the connectivity heatmap.
    plot_gppi_seed_profile : Inspect one seed's connection profile.
    """
    n_scans, n_rois = time_series.shape

    # Build HRF regressors once — same for every seed
    hrf_regs     = build_hrf_regressors(events_df, t_r, n_scans,
                                         conditions, hrf_model)
    n_conditions = len(hrf_regs)

    gppi_matrix = np.zeros((n_rois, n_rois))

    print(f'  [gPPI] Computing {n_rois}×{n_rois} matrix '
          f'({n_conditions} condition(s)) …')

    for seed_idx in range(n_rois):
        seed_ts = time_series[:, seed_idx]

        D, col_names, ppi_cols = build_gppi_design(seed_ts, hrf_regs, confounds)

        # OLS: D @ B ≈ Y  →  B shape (n_regressors, n_rois)
        B, _, _, _ = np.linalg.lstsq(D, time_series, rcond=None)

        # Sum PPI betas across conditions for each target
        gppi_matrix[seed_idx, :] = B[ppi_cols, :].sum(axis=0)

        if (seed_idx + 1) % 25 == 0 or seed_idx == n_rois - 1:
            print(f'  [gPPI] {seed_idx + 1}/{n_rois} seeds complete …')

    np.fill_diagonal(gppi_matrix, 0)

    if fisher_z:
        gppi_matrix = np.arctanh(np.clip(gppi_matrix, -1 + 1e-7, 1 - 1e-7))
        print('  [gPPI] Fisher z applied.')

    print(f'  [gPPI] Matrix complete: {gppi_matrix.shape}  '
          f'(min={gppi_matrix.min():.4f}, max={gppi_matrix.max():.4f})')

    if _logger:
        mask = np.ones_like(gppi_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        od = gppi_matrix[mask]
        _logger.log_step(
            'compute_gppi_matrix',
            matrix_shape = str(gppi_matrix.shape),
            n_conditions = n_conditions,
            conditions   = list(hrf_regs.keys()),
            hrf_model    = hrf_model,
            fisher_z     = fisher_z,
            value_min    = round(float(od.min()), 4),
            value_max    = round(float(od.max()), 4),
            value_mean   = round(float(od.mean()), 4),
        )

    return gppi_matrix


@_register
def plot_gppi_seed_profile(gppi_matrix, labels, seed_label,
                            n_top=20, figsize=(12, 5)):
    """
    Bar chart of the strongest gPPI connections for one seed ROI.

    Displays the top-N connections (by absolute beta value) from a given
    seed, making it easy to identify which regions are most strongly
    coupled with the seed *specifically during the task context*.

    Parameters
    ----------
    gppi_matrix : np.ndarray, shape (n_rois, n_rois)
        gPPI connectivity matrix from ``compute_gppi_matrix()``.

    labels : list of str
        ROI labels of length ``n_rois``.

    seed_label : str
        Name of the seed ROI.  Must exactly match one of the strings in
        ``labels``.

    n_top : int
        Number of connections to display (ranked by |beta|).  Default: 20.

    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Examples
    --------
    >>> fig = plot_gppi_seed_profile(
    ...     gppi_matrix, labels,
    ...     seed_label = '7Networks_LH_Default_PFC_1',
    ...     n_top      = 20,
    ... )
    >>> plt.show()

    Save:

    >>> fig.savefig('./gppi_output/seed_profile.png', dpi=150,
    ...             bbox_inches='tight')

    See also
    --------
    compute_gppi_matrix : Produce the gppi_matrix argument.
    plot_fc_matrix      : Full matrix heatmap.
    """
    if seed_label not in labels:
        close = [l for l in labels if seed_label.split('_')[-1] in l]
        raise ValueError(
            f'"{seed_label}" not found in labels.\n'
            f'  Closest matches: {close[:5]}\n'
            f'  Run `print(labels[:10])` to inspect available labels.'
        )

    seed_idx       = labels.index(seed_label)
    conn           = gppi_matrix[seed_idx, :].copy()
    conn[seed_idx] = 0   # zero self-connection

    # Rank by |beta|, then sort by value for display
    top_idx    = np.argsort(np.abs(conn))[::-1][:n_top]
    top_idx    = top_idx[np.argsort(conn[top_idx])]
    top_vals   = conn[top_idx]
    top_labels = [labels[i] for i in top_idx]
    colors     = ['tomato' if v < 0 else 'steelblue' for v in top_vals]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(top_idx)), top_vals, color=colors)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(top_labels, fontsize=8)
    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.set_xlabel('gPPI β  (connectivity strength during task)', fontsize=10)
    ax.set_title(f'Top {n_top} gPPI connections\nSeed: {seed_label}',
                 fontsize=11)
    plt.tight_layout()
    return fig
