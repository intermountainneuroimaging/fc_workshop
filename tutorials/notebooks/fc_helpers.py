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

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False


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
        ('Graph Theory',   ['threshold_proportional', 'matrix_to_graph',
                            'compute_graph_metrics', 'run_threshold_sweep',
                            'plot_threshold_sweep', 'plot_node_metric_by_network',
                            'plot_degree_distribution']),
        ('Group FC',       ['load_group_matrices', 'compute_group_mean_fc',
                            'one_sample_ttest_fc', 'two_sample_ttest_fc',
                            'network_based_stats', 'plot_significant_fc']),
        ('Group Graphs',   ['load_group_node_metrics', 'ttest_node_metric',
                            'permutation_test_global_metric',
                            'plot_node_tstat_by_network']),
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
        - An FSL mcflirt 6-DOF motion file (e.g. ``*.par``, ``*.txt``,
          or any extension).

        The format is detected automatically by inspecting the first
        line of the file: if it contains exactly 6 whitespace-delimited
        numeric tokens and no header, it is treated as FSL 6-DOF;
        otherwise it is parsed as a tab-separated fMRIPrep TSV.

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

    def _is_fsl_par(path):
        """Return True if *path* looks like a headerless 6-column numeric file."""
        try:
            with open(path, 'r') as fh:
                first_line = fh.readline().strip()
            if not first_line:
                return False
            tokens = first_line.split()
            if len(tokens) != 6:
                return False
            float(tokens[0])   # raises ValueError if not numeric
            float(tokens[-1])
            return True
        except (OSError, ValueError):
            return False

    if _is_fsl_par(confounds_path):
        # ── FSL mcflirt 6-DOF format ──────────────────────────────────────
        # Headerless, whitespace-delimited, exactly 6 numeric columns.
        # Standard order: rot_x(rad), rot_y(rad), rot_z(rad),
        #                 trans_x(mm), trans_y(mm), trans_z(mm)
        df = pd.read_csv(confounds_path, sep=r'\s+', header=None,
                         names=_PAR_COLS, engine='python')
        fmt = 'fsl_par'
        print(f'  [confounds] Detected FSL 6-DOF motion format '
              f'({len(df)} volumes, headerless)')

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
                   cmap='RdBu_r', vmin=None, vmax=None, figsize=(10, 8),
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

    vmin : float or None
        Colour scale minimum.  If ``None``, defaults to ``-vmax``.

    vmax : float or None
        Colour scale maximum.  If ``None``,
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
    if vmin is None:
        vmin = -vmax

    n = fc_matrix.shape[0]
    show_labels = (labels is not None) and (n <= 50)
    tick_labels = labels if show_labels else False

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        fc_matrix,
        ax          = ax,
        cmap        = cmap,
        vmin        = vmin,
        vmax        = vmax,
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

    Accepts either an fMRIPrep confounds TSV (which already contains a
    ``framewise_displacement`` column) or an FSL mcflirt 6-DOF motion
    file (headerless, 6 whitespace-delimited numeric columns).  The
    format is detected automatically by inspecting the first line of
    the file — the same logic used by :func:`load_confounds`.

    For FSL files, FD is computed on the fly using the Power et al.
    (2012) formula: the sum of absolute finite differences in the three
    translation parameters plus 50 mm × the absolute finite differences
    in the three rotation parameters (converting radians to mm assuming a
    50 mm head radius).

    Parameters
    ----------
    confounds_path : str or Path
        Path to an fMRIPrep confounds TSV **or** an FSL 6-DOF motion
        file (any extension, e.g. ``*.par``, ``*.txt``).

    scrub_threshold : float or None
        If set, draw a horizontal dashed line at this FD value and mark
        volumes that exceed it with red dots.

    t_r : float or None
        Repetition time in seconds.  If provided, the x-axis is in
        seconds; otherwise it is in volumes.

    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    mean_fd : float
        Mean framewise displacement across all volumes.
    pct_scrubbed : float
        Percentage of volumes that would be scrubbed at
        ``scrub_threshold`` (0.0 if ``scrub_threshold`` is None).

    Notes
    -----
    - FSL column order: rot_x (rad), rot_y (rad), rot_z (rad),
      trans_x (mm), trans_y (mm), trans_z (mm).
    - fMRIPrep FD is pre-computed; FSL FD is derived here using the
      Power et al. (2012) formulation with r = 50 mm.
    - Power et al. (2012) recommend a threshold of 0.5 mm for
      resting-state and 0.9 mm for task fMRI.

    Examples
    --------
    fMRIPrep TSV:

    >>> fig, mean_fd, pct = plot_fd_trace(
    ...     CONFOUNDS_PATH, scrub_threshold=0.5, t_r=TR
    ... )

    FSL motion file (any extension):

    >>> fig, mean_fd, pct = plot_fd_trace(
    ...     'sub-01_task-rest_mc.par', scrub_threshold=0.5, t_r=TR
    ... )

    >>> print(f'Mean FD: {mean_fd:.3f} mm | {pct:.1f}% scrubbed')

    See also
    --------
    load_confounds : Scrubbing is applied during confound loading.
    """
    confounds_path = Path(confounds_path)
    _PAR_COLS = ['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']

    def _is_fsl_par(path):
        try:
            with open(path, 'r') as fh:
                first_line = fh.readline().strip()
            if not first_line:
                return False
            tokens = first_line.split()
            if len(tokens) != 6:
                return False
            float(tokens[0])
            float(tokens[-1])
            return True
        except (OSError, ValueError):
            return False

    if _is_fsl_par(confounds_path):
        motion = pd.read_csv(confounds_path, sep=r'\s+', header=None,
                             names=_PAR_COLS, engine='python')
        _r    = 50.0
        _diff = motion[_PAR_COLS].diff().fillna(0).abs()
        _diff[['rot_x', 'rot_y', 'rot_z']] *= _r
        fd  = _diff.sum(axis=1).values
        fmt = 'FSL 6-DOF (FD computed via Power 2012)'
    else:
        df = pd.read_csv(confounds_path, sep='\t')
        if 'framewise_displacement' not in df.columns:
            raise ValueError(
                'framewise_displacement column not found. '
                'Pass an fMRIPrep TSV or an FSL 6-DOF motion file.')
        fd  = df['framewise_displacement'].fillna(0).values
        fmt = 'fMRIPrep TSV'

    print(f'  [FD] Format: {fmt}')

    x      = np.arange(len(fd)) * t_r if t_r else np.arange(len(fd))
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
            format           = fmt,
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
def plot_hrf_regressors(hrf_regressors, t_r=2.0, title=None, figsize=None):
    """
    Plot HRF-convolved task regressors on a shared time axis.

    Each condition is drawn as a separate line so you can visually inspect
    whether the predicted BOLD peaks align with when neural activity is
    expected for each trial type.  A dashed zero-line is included for
    reference.

    Parameters
    ----------
    hrf_regressors : dict
        Mapping ``condition_name → np.ndarray`` of shape ``(n_scans,)``,
        as returned by ``build_hrf_regressors()``.

    t_r : float
        Repetition time in seconds.  Used to scale the x-axis to seconds.
        Default: 2.0.

    title : str or None
        Figure title.  If ``None``, a generic title is used.

    figsize : tuple or None
        Figure size in inches ``(width, height)``.  If ``None``, scales
        with the number of conditions: ``(14, 2 + n_cond * 0.8)``.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Examples
    --------
    >>> hrf_regressors = build_hrf_regressors(events_df, TR, n_scans=240)
    >>> fig = plot_hrf_regressors(hrf_regressors, t_r=TR,
    ...                           title='Subject 01 — HRF regressors')
    >>> plt.show()

    Save the figure:

    >>> fig.savefig('./gppi_output/hrf_regressors.png', dpi=150,
    ...             bbox_inches='tight')

    See also
    --------
    build_hrf_regressors : Produce the hrf_regressors argument.
    build_gppi_design    : Consume regressors to build the design matrix.
    """
    if not hrf_regressors:
        raise ValueError('hrf_regressors dict is empty — nothing to plot.')

    n_cond = len(hrf_regressors)
    if figsize is None:
        figsize = (14, 2 + n_cond * 0.9)

    # Reference n_scans from the first regressor
    n_scans = next(iter(hrf_regressors.values())).shape[0]
    time_s  = np.arange(n_scans) * t_r          # TR-aligned time axis

    # Colour cycle — distinct colours even for many conditions
    cmap   = plt.cm.get_cmap('tab10', max(n_cond, 1))
    colors = [cmap(i) for i in range(n_cond)]

    fig, axes = plt.subplots(
        n_cond, 1,
        figsize     = figsize,
        sharex      = True,
        squeeze     = False,
    )

    for ax, (cond, reg), color in zip(axes[:, 0],
                                      hrf_regressors.items(),
                                      colors):
        ax.fill_between(time_s, reg, alpha=0.18, color=color)
        ax.plot(time_s, reg, lw=1.6, color=color, label=cond)
        ax.axhline(0, color='#888888', lw=0.7, ls='--')
        ax.set_ylabel(cond, fontsize=8.5, rotation=0,
                      labelpad=90, va='center')
        ax.set_ylim(min(reg.min() * 1.15, -0.05),
                    max(reg.max() * 1.15,  0.05))
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f'{v:.1f}'))
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='y', labelsize=7.5)

    axes[-1, 0].set_xlabel('Time (s)', fontsize=10)

    _title = title if title else f'HRF-convolved task regressors  ({n_cond} condition(s))'
    axes[0, 0].set_title(_title, fontsize=11, pad=8)

    plt.tight_layout()
    return fig


@_register
def plot_gppi_design_matrix(design_matrix, col_names, title=None, figsize=None):
    """
    Visualise a gPPI design matrix as a colour-coded heatmap.

    Columns are grouped into four blocks and colour-coded in the column
    header strip so you can immediately see the structure of the model:

    * **Physiological** (seed time series) — blue
    * **Psychological** (HRF-convolved task regressors) — green
    * **PPI interaction** (seed × psychological) — orange
    * **Nuisance + intercept** (confounds) — grey

    Parameters
    ----------
    design_matrix : np.ndarray, shape (T, n_columns)
        Design matrix from ``build_gppi_design()``.

    col_names : list of str
        Column names of length ``n_columns``, as returned by
        ``build_gppi_design()``.

    title : str or None
        Figure title.  If ``None``, a generic title is used.

    figsize : tuple or None
        Figure size in inches ``(width, height)``.  If ``None``, width
        scales with number of columns and height is fixed at 6.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Examples
    --------
    >>> D, names, ppi_idx = build_gppi_design(seed_ts, hrf_regressors, confounds)
    >>> fig = plot_gppi_design_matrix(D, names,
    ...           title='Subject 01 — gPPI design matrix | seed ROI')
    >>> plt.show()

    Save:

    >>> fig.savefig('./gppi_output/design_matrix.png', dpi=150,
    ...             bbox_inches='tight')

    See also
    --------
    build_gppi_design   : Produce design_matrix and col_names.
    plot_hrf_regressors : Visualise individual HRF-convolved regressors.
    """
    n_cols = len(col_names)
    if figsize is None:
        figsize = (max(8, n_cols * 0.55 + 2), 6)

    # ── Classify each column into one of four blocks ──────────────────────────
    BLOCK_COLORS = {
        'seed':   '#2D5FA8',   # blue   — physiological
        'psych':  '#2D9B6F',   # green  — psychological
        'ppi':    '#D97706',   # orange — PPI interaction
        'nuisance': '#64748B', # grey   — confounds / intercept
    }

    def _block(name):
        if name == 'seed':                  return 'seed'
        if name.startswith('psych_'):       return 'psych'
        if name.startswith('ppi_'):         return 'ppi'
        return 'nuisance'

    blocks  = [_block(n) for n in col_names]
    hcolors = [BLOCK_COLORS[b] for b in blocks]

    # ── Z-score each column for display (so amplitudes are comparable) ────────
    dm_display = design_matrix.copy().astype(float)
    col_std    = dm_display.std(axis=0)
    col_std[col_std == 0] = 1.0
    dm_display = (dm_display - dm_display.mean(axis=0)) / col_std

    # ── Figure layout: colour header strip + heatmap ──────────────────────────
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(
        2, 1,
        height_ratios = [0.06, 1],
        hspace        = 0.02,
    )
    ax_hdr = fig.add_subplot(gs[0])
    ax_dm  = fig.add_subplot(gs[1])

    # Colour header — one patch per column
    for j, color in enumerate(hcolors):
        ax_hdr.add_patch(
            plt.Rectangle((j, 0), 1, 1,
                           color=color, transform=ax_hdr.transData)
        )
    ax_hdr.set_xlim(0, n_cols)
    ax_hdr.set_ylim(0, 1)
    ax_hdr.axis('off')

    # Legend for the four blocks
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=BLOCK_COLORS['seed'],     label='Physiological (seed)'),
        Patch(facecolor=BLOCK_COLORS['psych'],    label='Psychological (HRF)'),
        Patch(facecolor=BLOCK_COLORS['ppi'],      label='PPI interaction'),
        Patch(facecolor=BLOCK_COLORS['nuisance'], label='Nuisance / intercept'),
    ]
    ax_hdr.legend(
        handles    = legend_elements,
        loc        = 'upper left',
        ncol       = 4,
        fontsize   = 7.5,
        frameon    = False,
        bbox_to_anchor = (0, 2.6),
    )

    # Heatmap
    im = ax_dm.imshow(
        dm_display,
        aspect     = 'auto',
        cmap       = 'RdBu_r',
        vmin       = -3, vmax = 3,
        interpolation = 'nearest',
    )
    ax_dm.set_xticks(range(n_cols))
    ax_dm.set_xticklabels(col_names, rotation=60, ha='right', fontsize=7.5)
    ax_dm.set_ylabel('Time (volumes)', fontsize=9)
    ax_dm.set_xlabel('Regressor', fontsize=9)

    # Vertical dividers between blocks
    prev_block = blocks[0]
    for j, b in enumerate(blocks[1:], start=1):
        if b != prev_block:
            ax_dm.axvline(j - 0.5, color='white', lw=1.2, ls='--')
            prev_block = b

    plt.colorbar(im, ax=ax_dm, shrink=0.6, pad=0.01,
                 label='Z-scored amplitude')

    _title = title if title else f'gPPI design matrix  ({n_cols} columns)'
    fig.suptitle(_title, fontsize=10, y=1.01)

    plt.tight_layout()
    return fig


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
    Compute a full ROI × ROI × condition generalised PPI connectivity matrix.

    For each seed ROI, fits an OLS general linear model with physiological
    (seed), psychological (HRF-convolved task), and PPI interaction
    regressors against every target ROI simultaneously.  Each condition's
    PPI beta is stored as a separate slice along the third dimension, so
    condition-specific connectivity maps are preserved for group analysis.

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
    gppi_matrix : np.ndarray, shape (n_rois, n_rois, n_conditions)
        gPPI connectivity matrix.  ``gppi_matrix[i, j, k]`` is the PPI
        beta for seed ROI ``i``, target ROI ``j``, and condition ``k``
        (ordered as in ``conditions``).  Diagonals are set to 0.

    conditions_out : list of str
        Condition names corresponding to the third dimension, in the same
        order as ``gppi_matrix[:, :, k]``.

    Notes
    -----
    - Unlike FC (symmetric by definition), gPPI matrices can be
      asymmetric because the seed modulates the regression.  In practice
      the asymmetry is small; symmetrising with ``(M + M.T) / 2`` before
      group analysis is common.
    - ``np.linalg.lstsq(D, Y)`` solves all ``n_rois`` targets in a single
      call per seed, making the inner loop efficient.  Expect ~1–3 min for
      Schaefer-200 at 2 mm resolution.
    - To collapse across conditions (e.g. for a task-general connectivity
      map), sum or average the third dimension:
      ``gppi_mean = gppi_matrix.mean(axis=2)``.

    Examples
    --------
    All conditions:

    >>> gppi_mat, conds = compute_gppi_matrix(ts, events_df, TR, confounds_array)
    >>> print(gppi_mat.shape)    # (n_rois, n_rois, n_conditions)
    >>> print(conds)             # ['CONDITION_A', 'CONDITION_B']

    Access a specific condition by index:

    >>> enc_matrix = gppi_mat[:, :, 0]   # first condition

    Access by condition name:

    >>> idx = conds.index('ENCODING')
    >>> enc_matrix = gppi_mat[:, :, idx]

    Summarise each condition:

    >>> for k, cond in enumerate(conds):
    ...     summarise_gppi(gppi_mat[:, :, k], label=cond)

    Symmetrise one condition slice for group analysis:

    >>> enc_sym = (enc_matrix + enc_matrix.T) / 2

    See also
    --------
    extract_time_series    : Use standardize=False, low_pass=None for gPPI.
    load_events            : Produce the events_df argument.
    build_hrf_regressors   : Called internally.
    build_gppi_design      : Called internally for each seed.
    summarise_gppi         : Print per-condition connectivity statistics.
    save_fc_matrix         : Save one condition slice as a labelled CSV.
    plot_fc_matrix         : Plot the connectivity heatmap.
    plot_gppi_seed_profile : Inspect one seed's connection profile.
    """
    n_scans, n_rois = time_series.shape

    # Build HRF regressors once — same for every seed
    hrf_regs      = build_hrf_regressors(events_df, t_r, n_scans,
                                          conditions, hrf_model)
    conditions_out = list(hrf_regs.keys())
    n_conditions   = len(conditions_out)

    # 3-D output: (n_rois, n_rois, n_conditions)
    gppi_matrix = np.zeros((n_rois, n_rois, n_conditions))

    print(f'  [gPPI] Computing {n_rois}×{n_rois}×{n_conditions} matrix '
          f'({n_conditions} condition(s)) …')

    for seed_idx in range(n_rois):
        seed_ts = time_series[:, seed_idx]

        D, col_names, ppi_cols = build_gppi_design(seed_ts, hrf_regs, confounds)

        # OLS: D @ B ≈ Y  →  B shape (n_regressors, n_rois)
        B, _, _, _ = np.linalg.lstsq(D, time_series, rcond=None)

        # Store each condition's beta as a separate slice (no summing)
        for k, col in enumerate(ppi_cols):
            gppi_matrix[seed_idx, :, k] = B[col, :]

        if (seed_idx + 1) % 25 == 0 or seed_idx == n_rois - 1:
            print(f'  [gPPI] {seed_idx + 1}/{n_rois} seeds complete …')

    # Zero the diagonal of every condition slice
    for k in range(n_conditions):
        np.fill_diagonal(gppi_matrix[:, :, k], 0)

    if fisher_z:
        gppi_matrix = np.arctanh(np.clip(gppi_matrix, -1 + 1e-7, 1 - 1e-7))
        print('  [gPPI] Fisher z applied.')

    print(f'  [gPPI] Matrix complete: {gppi_matrix.shape}  '
          f'(min={gppi_matrix.min():.4f}, max={gppi_matrix.max():.4f})')

    if _logger:
        od = gppi_matrix[~np.eye(n_rois, dtype=bool), :]
        _logger.log_step(
            'compute_gppi_matrix',
            matrix_shape  = str(gppi_matrix.shape),
            n_conditions  = n_conditions,
            conditions    = conditions_out,
            hrf_model     = hrf_model,
            fisher_z      = fisher_z,
            value_min     = round(float(od.min()), 4),
            value_max     = round(float(od.max()), 4),
            value_mean    = round(float(od.mean()), 4),
        )

    return gppi_matrix, conditions_out


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


@_register
def summarise_gppi(gppi_matrix, label=None, n_top=5, threshold=0.0):
    """
    Print a concise text summary of a gPPI connectivity matrix.

    Reports the matrix dimensions, the distribution of beta weights, and
    the strongest positive and negative connections.  Useful for a quick
    sanity check after ``compute_gppi_matrix()``.

    Parameters
    ----------
    gppi_matrix : np.ndarray, shape (n_rois, n_rois)
        A single-condition slice of the 3-D gPPI output, e.g.
        ``gppi_matrix[:, :, idx]``.

    label : str or None
        Descriptive label printed in the header (e.g. subject / task /
        condition string).  If ``None``, only the matrix shape is shown.

    n_top : int
        Number of strongest positive and negative edges to list.
        Default: 5.

    threshold : float
        Edges with |beta| ≤ threshold are excluded from the top-N lists.
        Default: 0.0 (all non-zero edges shown).

    Returns
    -------
    stats : dict
        Keys:
        ``n_rois``, ``n_edges``, ``mean_beta``, ``std_beta``,
        ``max_beta``, ``min_beta``, ``pct_positive``, ``pct_negative``.

    Examples
    --------
    >>> stats = summarise_gppi(gppi_matrix[:, :, 0],
    ...                        label='sub-01 | MemTask | ENCODING')
    >>> print(stats['mean_beta'])

    See also
    --------
    compute_gppi_matrix  : Produce the gPPI matrix.
    plot_fc_matrix       : Visualise the full matrix as a heatmap.
    plot_gppi_seed_profile : Bar chart of the strongest connections for one seed.
    """
    n_rois = gppi_matrix.shape[0]

    # Extract upper triangle (unique off-diagonal edges)
    triu_idx   = np.triu_indices(n_rois, k=1)
    edge_vals  = gppi_matrix[triu_idx]
    n_edges    = len(edge_vals)

    # Exclude near-zero edges from top-N if threshold set
    mask      = np.abs(edge_vals) > threshold
    masked    = edge_vals[mask]

    mean_beta = float(np.mean(edge_vals))
    std_beta  = float(np.std(edge_vals))
    max_beta  = float(np.max(edge_vals))
    min_beta  = float(np.min(edge_vals))
    pct_pos   = float(np.mean(edge_vals > 0) * 100)
    pct_neg   = float(np.mean(edge_vals < 0) * 100)

    # ── Print header ──────────────────────────────────────────────────────────
    sep = '─' * 60
    hdr = f'  gPPI summary | {label}' if label else f'  gPPI summary'
    print(f'\n{sep}')
    print(hdr)
    print(sep)
    print(f'  ROIs          : {n_rois}')
    print(f'  Unique edges  : {n_edges}')
    print(f'  Mean β        : {mean_beta:+.4f}')
    print(f'  Std  β        : {std_beta:.4f}')
    print(f'  Range β       : [{min_beta:+.4f}, {max_beta:+.4f}]')
    print(f'  Positive edges: {pct_pos:.1f}%   |   Negative edges: {pct_neg:.1f}%')

    # ── Top positive ──────────────────────────────────────────────────────────
    pos_idx  = np.argsort(edge_vals)[::-1]
    pos_idx  = [i for i in pos_idx if edge_vals[i] > threshold][:n_top]
    if pos_idx:
        print(f'\n  Top {n_top} positive edges (β):')
        for rank, i in enumerate(pos_idx, 1):
            r, c = triu_idx[0][i], triu_idx[1][i]
            print(f'    {rank}. ROI[{r:3d}] ↔ ROI[{c:3d}]   β = {edge_vals[i]:+.4f}')

    # ── Top negative ──────────────────────────────────────────────────────────
    neg_idx  = np.argsort(edge_vals)
    neg_idx  = [i for i in neg_idx if edge_vals[i] < -threshold][:n_top]
    if neg_idx:
        print(f'\n  Top {n_top} negative edges (β):')
        for rank, i in enumerate(neg_idx, 1):
            r, c = triu_idx[0][i], triu_idx[1][i]
            print(f'    {rank}. ROI[{r:3d}] ↔ ROI[{c:3d}]   β = {edge_vals[i]:+.4f}')

    print(sep + '\n')

    stats = dict(
        n_rois       = n_rois,
        n_edges      = n_edges,
        mean_beta    = mean_beta,
        std_beta     = std_beta,
        max_beta     = max_beta,
        min_beta     = min_beta,
        pct_positive = pct_pos,
        pct_negative = pct_neg,
    )
    return stats


# =============================================================================
# Graph Theory
# =============================================================================

def _require_networkx():
    if not _HAS_NETWORKX:
        raise ImportError(
            'networkx is not installed.  Run:  pip install networkx')


def _network_from_label(label):
    """Extract Yeo network name from a Schaefer parcel label.

    Schaefer label format: ``7Networks_LH_Default_PFC_1``
    Returns the network component (e.g. ``'Default'``) or ``'Unknown'``.
    """
    parts = label.split('_')
    if len(parts) >= 3 and 'Networks' in parts[0]:
        return parts[2]
    return 'Unknown'


@_register
def threshold_proportional(fc_matrix, proportion=0.10, positive_only=True):
    """
    Threshold an FC matrix by keeping the strongest proportion of edges.

    Edges are ranked by absolute weight.  Only the top ``proportion`` of
    unique off-diagonal edges survive; all others are set to zero.

    This proportional approach ensures that matrices with different raw
    weight distributions are compared at the same *network density*, which
    is important for valid cross-subject or cross-condition comparisons.

    Parameters
    ----------
    fc_matrix : np.ndarray, shape (N, N)
        Square symmetric connectivity matrix (Pearson r or Fisher z).

    proportion : float
        Fraction of edges to retain, between 0 and 1.  A value of 0.10
        keeps the strongest 10 %% of possible connections.  Typical values
        for resting-state FC are 0.05 – 0.25.

    positive_only : bool
        If True (default), negative-weight edges are removed before
        ranking, so only positive connections survive.  Negative FC
        values are often noisy or artefactual, particularly after global
        signal regression.

    Returns
    -------
    adj : np.ndarray, shape (N, N)
        Thresholded adjacency matrix (same shape as input).  Self-loops
        are zero.  The result is symmetric.

    Notes
    -----
    - The diagonal is always zeroed before and after thresholding.
    - If ``positive_only=True`` the effective density will be lower than
      ``proportion`` when a substantial fraction of edges are negative.
    - Use ``run_threshold_sweep`` to test sensitivity across multiple
      sparsity levels before choosing a single threshold.

    Examples
    --------
    >>> adj = threshold_proportional(fc_matrix, proportion=0.10)
    >>> print(f'Density: {(adj > 0).sum() / (adj.shape[0]**2):.3f}')

    See also
    --------
    run_threshold_sweep    : Sweep across multiple thresholds at once.
    matrix_to_graph        : Convert the thresholded matrix to a NetworkX graph.
    compute_graph_metrics  : Compute node and graph-level metrics.
    """
    mat = np.array(fc_matrix, dtype=float).copy()
    np.fill_diagonal(mat, 0)
    if positive_only:
        mat[mat < 0] = 0

    # Upper-triangle values only (avoid double-counting)
    iu   = np.triu_indices_from(mat, k=1)
    vals = mat[iu]
    if positive_only:
        vals = vals[vals > 0]
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return np.zeros_like(mat)

    n_keep = max(1, int(np.floor(proportion * len(vals))))
    cutoff = np.sort(vals)[-n_keep]

    adj = np.where(mat >= cutoff, mat, 0.0)
    adj = np.maximum(adj, adj.T)   # ensure symmetry
    np.fill_diagonal(adj, 0)

    n_nodes  = adj.shape[0]
    n_edges  = int((adj > 0).sum() // 2)
    density  = n_edges / (n_nodes * (n_nodes - 1) / 2)
    print(f'  [threshold] proportion={proportion:.2f} | '
          f'edges={n_edges} | density={density:.4f} | '
          f'positive_only={positive_only}')
    return adj


@_register
def matrix_to_graph(adj, labels):
    """
    Convert a thresholded adjacency matrix to a weighted NetworkX graph.

    Each node receives two attributes: ``label`` (the ROI name from
    ``labels``) and ``network`` (the Yeo network parsed from Schaefer
    parcel labels, or ``'Unknown'`` for other atlases).

    Parameters
    ----------
    adj : np.ndarray, shape (N, N)
        Thresholded adjacency matrix produced by ``threshold_proportional``.
        Edge weights are stored on the graph.

    labels : list of str
        Length-N list of parcel label strings.

    Returns
    -------
    G : networkx.Graph
        Undirected weighted graph with N nodes.  Node integers match row /
        column indices of ``adj``.

    Examples
    --------
    >>> adj = threshold_proportional(fc_matrix, proportion=0.10)
    >>> G   = matrix_to_graph(adj, roi_labels)
    >>> print(f'Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}')

    See also
    --------
    threshold_proportional : Produce the ``adj`` argument.
    compute_graph_metrics  : Compute metrics from the graph or adjacency matrix.
    """
    _require_networkx()
    G = nx.from_numpy_array(adj)
    attrs = {}
    for i, lab in enumerate(labels):
        attrs[i] = {'label': lab, 'network': _network_from_label(lab)}
    nx.set_node_attributes(G, attrs)
    return G


@_register
def compute_graph_metrics(adj, labels):
    """
    Compute node-level and graph-level metrics from a thresholded adjacency matrix.

    **Node-level metrics** (one row per parcel):

    - ``degree``      — number of connections
    - ``strength``    — sum of edge weights (weighted degree)
    - ``clustering``  — weighted clustering coefficient (Onnela et al.)
    - ``betweenness`` — normalised betweenness centrality (unweighted paths)

    **Graph-level metrics** (one row, returned as a dict):

    - ``n_nodes``, ``n_edges``, ``density``
    - ``mean_degree``, ``mean_strength``, ``mean_clustering``
    - ``transitivity`` — ratio of triangles to connected triples
    - ``avg_path_length`` — average shortest path in the largest component
      (``np.nan`` if the largest component has only 1 node)
    - ``n_components`` — number of connected components

    Parameters
    ----------
    adj : np.ndarray, shape (N, N)
        Thresholded adjacency matrix (output of ``threshold_proportional``).

    labels : list of str
        Length-N list of parcel label strings.

    Returns
    -------
    node_df : pd.DataFrame, shape (N, 6+)
        One row per parcel.  Columns: ``label``, ``network``, ``degree``,
        ``strength``, ``clustering``, ``betweenness``.

    graph_dict : dict
        Graph-level summary metrics.

    Notes
    -----
    - Betweenness centrality uses unweighted (hop-count) paths, consistent
      with the convention that shorter paths in graph theory mean fewer hops,
      not lower edge weights.
    - Average path length is computed on the **largest connected component**
      when the graph is not fully connected.  This avoids infinite paths
      across disconnected components.

    Examples
    --------
    >>> adj      = threshold_proportional(fc_matrix, proportion=0.10)
    >>> node_df, graph_dict = compute_graph_metrics(adj, roi_labels)
    >>> print(node_df[['label', 'degree', 'strength', 'clustering']].head())
    >>> print(graph_dict)

    See also
    --------
    threshold_proportional : Produce the ``adj`` argument.
    run_threshold_sweep    : Compute metrics across multiple thresholds.
    plot_node_metric_by_network : Visualise node metrics by Yeo network.
    """
    _require_networkx()
    G = matrix_to_graph(adj, labels)

    degree      = dict(G.degree())
    strength    = dict(G.degree(weight='weight'))
    clustering  = nx.clustering(G, weight='weight')
    betweenness = nx.betweenness_centrality(G, weight=None, normalized=True)

    node_df = pd.DataFrame({
        'label'      : labels,
        'network'    : [_network_from_label(l) for l in labels],
        'degree'     : [degree.get(i, 0)      for i in range(len(labels))],
        'strength'   : [strength.get(i, 0.0)  for i in range(len(labels))],
        'clustering' : [clustering.get(i, 0.0) for i in range(len(labels))],
        'betweenness': [betweenness.get(i, 0.0) for i in range(len(labels))],
    })

    # Graph-level metrics
    if nx.is_connected(G):
        avg_path = nx.average_shortest_path_length(G, weight=None)
    else:
        lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        avg_path = (nx.average_shortest_path_length(lcc, weight=None)
                    if lcc.number_of_nodes() > 1 else np.nan)

    graph_dict = {
        'n_nodes'       : G.number_of_nodes(),
        'n_edges'       : G.number_of_edges(),
        'density'       : nx.density(G),
        'mean_degree'   : float(np.mean(list(degree.values()))),
        'mean_strength' : float(np.mean(list(strength.values()))),
        'mean_clustering': float(np.mean(list(clustering.values()))),
        'transitivity'  : nx.transitivity(G),
        'avg_path_length': avg_path,
        'n_components'  : nx.number_connected_components(G),
    }

    print(f'  [graph] nodes={graph_dict["n_nodes"]} | '
          f'edges={graph_dict["n_edges"]} | '
          f'density={graph_dict["density"]:.4f} | '
          f'mean_clustering={graph_dict["mean_clustering"]:.4f} | '
          f'components={graph_dict["n_components"]}')

    if _logger:
        _logger.log_step('compute_graph_metrics', **graph_dict)

    return node_df, graph_dict


@_register
def run_threshold_sweep(fc_matrix, labels,
                        thresholds=(0.05, 0.10, 0.15, 0.20, 0.25),
                        positive_only=True):
    """
    Compute graph metrics across a range of proportional thresholds.

    Graph-theory conclusions should be stable across a reasonable range of
    sparsity levels.  This function runs ``threshold_proportional`` and
    ``compute_graph_metrics`` at each threshold and concatenates the results
    into tidy DataFrames for inspection and plotting.

    Parameters
    ----------
    fc_matrix : np.ndarray, shape (N, N)
        Full FC matrix (unthresholded Pearson r or Fisher z).

    labels : list of str
        Length-N parcel label strings.

    thresholds : tuple of float
        Proportional thresholds to evaluate.  Default is
        ``(0.05, 0.10, 0.15, 0.20, 0.25)``.

    positive_only : bool
        Passed to ``threshold_proportional``.  Default True.

    Returns
    -------
    node_sweep : pd.DataFrame
        Node-level metrics at every threshold.  Columns include
        ``threshold``, ``label``, ``network``, ``degree``, ``strength``,
        ``clustering``, ``betweenness``.

    graph_sweep : pd.DataFrame
        Graph-level metrics at every threshold.  One row per threshold.

    Examples
    --------
    >>> node_sweep, graph_sweep = run_threshold_sweep(fc_matrix, roi_labels)
    >>> print(graph_sweep[['threshold', 'density',
    ...                     'mean_clustering', 'avg_path_length']])

    See also
    --------
    threshold_proportional : Single-threshold version.
    plot_threshold_sweep   : Visualise the sweep results.
    """
    node_rows  = []
    graph_rows = []
    for thr in thresholds:
        adj = threshold_proportional(fc_matrix, proportion=thr,
                                     positive_only=positive_only)
        ndf, gdict = compute_graph_metrics(adj, labels)
        ndf  = ndf.copy();  ndf['threshold']  = thr
        gdict['threshold'] = thr
        node_rows.append(ndf)
        graph_rows.append(gdict)
    node_sweep  = pd.concat(node_rows, ignore_index=True)
    graph_sweep = pd.DataFrame(graph_rows)
    return node_sweep, graph_sweep


@_register
def plot_threshold_sweep(graph_sweep,
                         metrics=('density', 'mean_degree',
                                  'mean_strength', 'mean_clustering',
                                  'avg_path_length'),
                         figsize=(8, 11)):
    """
    Plot graph-level metrics across a proportional threshold sweep.

    Each panel shows how one graph metric changes as the network is made
    sparser (higher threshold) or denser (lower threshold).  Stable metrics
    across the sweep indicate that conclusions are robust to the choice of
    threshold.

    Parameters
    ----------
    graph_sweep : pd.DataFrame
        Output of ``run_threshold_sweep``.  Must contain a ``threshold``
        column and one column per metric in ``metrics``.

    metrics : tuple of str
        Graph-level metric column names to plot.  Any column present in
        ``graph_sweep`` is valid.

    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Examples
    --------
    >>> node_sweep, graph_sweep = run_threshold_sweep(fc_matrix, roi_labels)
    >>> fig = plot_threshold_sweep(graph_sweep)
    >>> plt.show()

    See also
    --------
    run_threshold_sweep : Produce the ``graph_sweep`` argument.
    """
    cols = [m for m in metrics if m in graph_sweep.columns]
    fig, axes = plt.subplots(len(cols), 1, figsize=figsize, sharex=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, metric in zip(axes, cols):
        ax.plot(graph_sweep['threshold'], graph_sweep[metric],
                marker='o', lw=1.5, color='steelblue')
        ax.set_ylabel(metric.replace('_', ' '), fontsize=9)
        ax.grid(axis='y', lw=0.4, alpha=0.5)
    axes[-1].set_xlabel('Proportional threshold', fontsize=10)
    axes[0].set_title('Graph metrics across threshold sweep', fontsize=11)
    plt.tight_layout()
    return fig


@_register
def plot_node_metric_by_network(node_df, metric='strength',
                                threshold=None, figsize=(9, 4),
                                sort_by_value=True):
    """
    Bar chart of a node metric averaged by Yeo network.

    Aggregates the chosen node-level metric across parcels within each
    Yeo/Schaefer network and displays the mean as a horizontal bar.  Error
    bars show ± 1 standard deviation across parcels in that network.

    Parameters
    ----------
    node_df : pd.DataFrame
        Output of ``compute_graph_metrics`` or a single-threshold slice of
        ``run_threshold_sweep``.  Must contain ``network`` and ``metric``
        columns.

    metric : str
        Column name of the node metric to plot.  Common choices:
        ``'strength'``, ``'degree'``, ``'clustering'``, ``'betweenness'``.

    threshold : float or None
        If ``node_df`` is from ``run_threshold_sweep`` (which contains
        multiple thresholds), pass the threshold value to filter to.
        Leave as ``None`` if ``node_df`` already contains a single
        threshold.

    figsize : tuple
        Figure size in inches.

    sort_by_value : bool
        If True (default), sort bars from highest to lowest mean value.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Examples
    --------
    Single threshold:

    >>> adj      = threshold_proportional(fc_matrix, proportion=0.10)
    >>> node_df, _ = compute_graph_metrics(adj, roi_labels)
    >>> fig = plot_node_metric_by_network(node_df, metric='strength')

    From a sweep:

    >>> node_sweep, _ = run_threshold_sweep(fc_matrix, roi_labels)
    >>> fig = plot_node_metric_by_network(
    ...     node_sweep, metric='clustering', threshold=0.10)

    See also
    --------
    compute_graph_metrics : Produce node_df for a single threshold.
    run_threshold_sweep   : Produce node_df across thresholds.
    plot_degree_distribution : Per-node degree histogram.
    """
    df = node_df.copy()
    if threshold is not None and 'threshold' in df.columns:
        df = df[df['threshold'] == threshold]

    summary = (df.groupby('network')[metric]
               .agg(['mean', 'std'])
               .reset_index())
    if sort_by_value:
        summary = summary.sort_values('mean', ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(summary['network'], summary['mean'],
                  yerr=summary['std'], capsize=4,
                  color='steelblue', edgecolor='white', lw=0.5)
    ax.set_ylabel(f'Mean {metric}', fontsize=10)
    ax.set_title(
        f'Mean node {metric} by Yeo network'
        + (f'  (threshold = {threshold})' if threshold is not None else ''),
        fontsize=11,
    )
    ax.tick_params(axis='x', rotation=35)
    ax.grid(axis='y', lw=0.4, alpha=0.5)
    plt.tight_layout()
    return fig


@_register
def plot_degree_distribution(adj, bins=20, figsize=(7, 4)):
    """
    Plot the degree distribution of a thresholded graph.

    The degree distribution describes how many connections each node has.
    Scale-free networks show a heavy tail (few highly connected hubs);
    random networks show a narrow Poisson-like distribution.  Comparing
    the empirical distribution to a random graph null is a basic sanity
    check for brain network analyses.

    Parameters
    ----------
    adj : np.ndarray, shape (N, N)
        Thresholded adjacency matrix (output of ``threshold_proportional``).

    bins : int
        Number of histogram bins.

    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    degrees : np.ndarray
        Array of node degrees (length N).

    Examples
    --------
    >>> adj = threshold_proportional(fc_matrix, proportion=0.10)
    >>> fig, degrees = plot_degree_distribution(adj)
    >>> print(f'Mean degree: {degrees.mean():.1f}  Max: {degrees.max()}')
    >>> plt.show()

    See also
    --------
    threshold_proportional     : Produce the ``adj`` argument.
    plot_node_metric_by_network : Summarise by network.
    """
    _require_networkx()
    G       = nx.from_numpy_array(adj)
    degrees = np.array([d for _, d in G.degree()])
    mean_k  = degrees.mean()

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(degrees, bins=bins, color='steelblue', edgecolor='white', lw=0.5)
    ax.axvline(mean_k, color='tomato', lw=1.5, ls='--',
               label=f'Mean degree = {mean_k:.1f}')
    ax.set_xlabel('Degree', fontsize=10)
    ax.set_ylabel('Number of nodes', fontsize=10)
    ax.set_title('Node degree distribution', fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig, degrees


# =============================================================================
# Group-Level FC Analysis
# =============================================================================

def _fdr_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction (internal helper).

    Parameters
    ----------
    pvals : array-like, shape (n,)
        Raw p-values (will be flattened).
    alpha : float
        FDR level.

    Returns
    -------
    reject : np.ndarray of bool, shape (n,)
        True where the null hypothesis is rejected.
    p_adj : np.ndarray of float, shape (n,)
        BH-adjusted p-values (same order as input).
    """
    pv   = np.asarray(pvals, dtype=float).ravel()
    n    = len(pv)
    idx  = np.argsort(pv)               # ascending rank order
    sp   = pv[idx]                       # sorted p-values
    # Step-up adjusted p: p_adj(i) = min_{j>=i}( p(j) * n / j )
    raw  = sp * n / np.arange(1, n + 1)
    padj = np.minimum.accumulate(raw[::-1])[::-1]
    padj = np.clip(padj, 0.0, 1.0)
    rej  = padj <= alpha
    # Restore original order
    padj_out        = np.empty(n)
    rej_out         = np.empty(n, dtype=bool)
    padj_out[idx]   = padj
    rej_out[idx]    = rej
    return rej_out, padj_out


@_register
def load_group_matrices(matrix_paths, ref_labels=None):
    """
    Load multiple FC matrix CSVs into a single 3-D array.

    Each CSV is expected to be a square, labelled matrix as written by
    ``save_fc_matrix``: parcel names as both the row index and the column
    header.  Labels are verified to be identical across subjects; a
    ``ValueError`` is raised if they differ.

    Parameters
    ----------
    matrix_paths : list of str or Path
        Paths to individual-subject FC matrix CSVs, one per subject.

    ref_labels : list of str or None
        Expected parcel labels.  If None, the labels from the first file
        are used as the reference and all subsequent files are checked
        against them.

    Returns
    -------
    matrices : np.ndarray, shape (n_subjects, N, N)
        Stacked FC matrices (Fisher z or Pearson r, depending on what was
        saved).
    labels : list of str
        Length-N parcel label strings (from the reference file).

    Examples
    --------
    >>> paths = sorted(Path('matrices/').glob('sub-*_fc.csv'))
    >>> matrices, labels = load_group_matrices(paths)
    >>> print(matrices.shape)   # (n_subjects, N, N)

    See also
    --------
    compute_group_mean_fc : Compute mean and SEM across subjects.
    one_sample_ttest_fc   : Test whether each edge differs from zero.
    """
    paths = list(matrix_paths)
    if len(paths) == 0:
        raise ValueError('matrix_paths is empty.')

    mats   = []
    labels = ref_labels
    for i, p in enumerate(paths):
        df = pd.read_csv(p, index_col=0)
        if labels is None:
            labels = df.columns.tolist()
        elif list(df.columns) != list(labels):
            raise ValueError(
                f'Label mismatch at file {i} ({Path(p).name}).\n'
                f'  Expected first label: {labels[0]}\n'
                f'  Got first label     : {df.columns[0]}')
        mats.append(df.values.astype(float))

    matrices = np.stack(mats, axis=0)
    print(f'  [load_group_matrices] {len(paths)} subjects | '
          f'matrix shape: {matrices.shape[1]}×{matrices.shape[2]}')
    return matrices, labels


@_register
def compute_group_mean_fc(matrices, labels):
    """
    Compute the group mean and standard error of the FC matrix.

    Parameters
    ----------
    matrices : np.ndarray, shape (n_subjects, N, N)
        Stacked subject FC matrices from ``load_group_matrices``.

    labels : list of str
        Length-N parcel label strings.

    Returns
    -------
    mean_mat : np.ndarray, shape (N, N)
        Element-wise mean across subjects.
    sem_mat : np.ndarray, shape (N, N)
        Element-wise standard error of the mean (std / sqrt(n)).

    Examples
    --------
    >>> mean_mat, sem_mat = compute_group_mean_fc(matrices, labels)
    >>> print(f'Mean global FC: {np.nanmean(mean_mat):.3f}')

    See also
    --------
    one_sample_ttest_fc : Statistical test for each edge.
    plot_significant_fc : Visualise significant edges.
    """
    n        = matrices.shape[0]
    mean_mat = matrices.mean(axis=0)
    sem_mat  = matrices.std(axis=0, ddof=1) / np.sqrt(n)

    global_mean = float(np.nanmean(mean_mat[np.triu_indices(len(labels), k=1)]))
    print(f'  [group_mean_fc] n={n} | global mean FC={global_mean:.4f} | '
          f'mean SEM={float(sem_mat.mean()):.4f}')

    if _logger:
        _logger.log_step('compute_group_mean_fc',
                         n_subjects=n,
                         global_mean_fc=round(global_mean, 4))
    return mean_mat, sem_mat


@_register
def one_sample_ttest_fc(matrices, alpha=0.05):
    """
    Test whether each FC edge differs significantly from zero at the group level.

    Runs a one-sample t-test (H₀: μ = 0) at every upper-triangle edge,
    then applies Benjamini-Hochberg FDR correction across all edges to
    control the false discovery rate.

    Parameters
    ----------
    matrices : np.ndarray, shape (n_subjects, N, N)
        Stacked subject FC matrices from ``load_group_matrices``.

    alpha : float
        FDR significance level.  Default 0.05.

    Returns
    -------
    t_mat : np.ndarray, shape (N, N)
        Symmetric matrix of t-statistics.
    p_fdr_mat : np.ndarray, shape (N, N)
        Symmetric matrix of BH-adjusted p-values.
    sig_mask : np.ndarray of bool, shape (N, N)
        True where the edge survives FDR correction.

    Notes
    -----
    - The number of simultaneous tests is N*(N-1)/2 (one per unique edge).
    - FDR controls the *expected proportion* of false discoveries among all
      rejected hypotheses, making it less conservative than Bonferroni while
      still controlling for multiple comparisons.
    - For an additional cluster-level correction see ``network_based_stats``.

    Examples
    --------
    >>> t_mat, p_fdr, sig = one_sample_ttest_fc(matrices, alpha=0.05)
    >>> print(f'Significant edges: {sig.sum() // 2}')

    See also
    --------
    two_sample_ttest_fc : Between-group edge comparison.
    network_based_stats : Cluster-level (NBS) correction.
    plot_significant_fc : Visualise the significant-edge mask.
    """
    from scipy.stats import t as _tdist
    n_sub, N, _ = matrices.shape
    iu = np.triu_indices(N, k=1)

    mu   = matrices.mean(axis=0)
    std  = matrices.std(axis=0, ddof=1)
    sem  = std / np.sqrt(n_sub)
    t_mat = np.where(sem > 0, mu / sem, 0.0)

    p_mat = 2.0 * _tdist.sf(np.abs(t_mat), df=n_sub - 1)

    # FDR on upper triangle only (avoid double-counting)
    rej, p_fdr_up = _fdr_bh(p_mat[iu], alpha=alpha)

    p_fdr_mat = np.ones((N, N))
    sig_mask  = np.zeros((N, N), dtype=bool)
    p_fdr_mat[iu]              = p_fdr_up
    p_fdr_mat[(iu[1], iu[0])] = p_fdr_up
    sig_mask[iu]               = rej
    sig_mask[(iu[1], iu[0])]  = rej

    n_sig   = int(rej.sum())
    n_total = len(rej)
    print(f'  [one_sample_ttest_fc] {n_sig}/{n_total} edges significant '
          f'(FDR q<{alpha}; {100*n_sig/n_total:.1f}%)')

    if _logger:
        _logger.log_step('one_sample_ttest_fc',
                         n_subjects=n_sub, alpha=alpha,
                         n_edges_tested=n_total,
                         n_significant=n_sig)
    return t_mat, p_fdr_mat, sig_mask


@_register
def two_sample_ttest_fc(matrices_a, matrices_b, alpha=0.05):
    """
    Compare FC matrices between two groups at each edge (Welch's t-test + FDR).

    For every upper-triangle edge, fits an independent-samples Welch's
    t-test (unequal variance) and then applies Benjamini-Hochberg FDR
    correction across all edges.  Positive t-values indicate group A > B.

    Parameters
    ----------
    matrices_a : np.ndarray, shape (n_a, N, N)
        FC matrices for group A (e.g. patients or condition 1).

    matrices_b : np.ndarray, shape (n_b, N, N)
        FC matrices for group B (e.g. controls or condition 2).

    alpha : float
        FDR significance level.  Default 0.05.

    Returns
    -------
    t_mat : np.ndarray, shape (N, N)
        Symmetric t-statistic matrix (A − B).
    p_fdr_mat : np.ndarray, shape (N, N)
        BH-adjusted p-values.
    sig_mask : np.ndarray of bool, shape (N, N)
        True where the edge survives FDR correction.

    Examples
    --------
    >>> t_mat, p_fdr, sig = two_sample_ttest_fc(
    ...     matrices_patients, matrices_controls, alpha=0.05)
    >>> print(f'Stronger in patients: {(sig & (t_mat > 0)).sum() // 2} edges')
    >>> print(f'Stronger in controls: {(sig & (t_mat < 0)).sum() // 2} edges')

    See also
    --------
    one_sample_ttest_fc : Within-group test against zero.
    network_based_stats : Cluster-level correction for two-group designs.
    """
    from scipy.stats import t as _tdist
    a  = np.asarray(matrices_a, dtype=float)
    b  = np.asarray(matrices_b, dtype=float)
    na, N, _ = a.shape
    nb       = b.shape[0]
    iu       = np.triu_indices(N, k=1)

    mu_a = a.mean(axis=0);  var_a = a.var(axis=0, ddof=1)
    mu_b = b.mean(axis=0);  var_b = b.var(axis=0, ddof=1)

    se    = np.sqrt(var_a / na + var_b / nb)
    t_mat = np.where(se > 0, (mu_a - mu_b) / se, 0.0)

    # Welch-Satterthwaite degrees of freedom
    df_num = (var_a / na + var_b / nb) ** 2
    df_den = ((var_a / na) ** 2 / (na - 1) +
              (var_b / nb) ** 2 / (nb - 1))
    df    = np.where(df_den > 0, df_num / df_den, 1.0)

    p_mat = 2.0 * _tdist.sf(np.abs(t_mat), df=df)

    rej, p_fdr_up = _fdr_bh(p_mat[iu], alpha=alpha)

    p_fdr_mat = np.ones((N, N))
    sig_mask  = np.zeros((N, N), dtype=bool)
    p_fdr_mat[iu]              = p_fdr_up
    p_fdr_mat[(iu[1], iu[0])] = p_fdr_up
    sig_mask[iu]               = rej
    sig_mask[(iu[1], iu[0])]  = rej

    n_sig   = int(rej.sum())
    n_total = len(rej)
    pos     = int((rej & (t_mat[iu] > 0)).sum())
    neg     = int((rej & (t_mat[iu] < 0)).sum())
    print(f'  [two_sample_ttest_fc] n_A={na}  n_B={nb} | '
          f'{n_sig}/{n_total} significant edges (FDR q<{alpha}) | '
          f'A>B: {pos}  B>A: {neg}')

    if _logger:
        _logger.log_step('two_sample_ttest_fc',
                         n_a=na, n_b=nb, alpha=alpha,
                         n_edges_tested=n_total,
                         n_significant=n_sig,
                         edges_a_gt_b=pos, edges_b_gt_a=neg)
    return t_mat, p_fdr_mat, sig_mask


@_register
def network_based_stats(matrices_a, matrices_b, threshold=3.0,
                         alpha=0.05, n_perm=1000, seed=42):
    """
    Network-Based Statistics (NBS) for two-group connectome comparisons.

    NBS is a cluster-level correction that controls the family-wise error
    rate (FWER) over connected components of the thresholded t-statistic
    matrix.  It is the connectome analog of cluster-extent correction in
    mass-univariate neuroimaging (Zalesky et al., *NeuroImage*, 2010).

    **Algorithm**

    1. Compute Welch's t-statistic at each edge (A − B).
    2. Apply a primary threshold (``|t| ≥ threshold``) to form a
       binary suprathreshold graph.
    3. Identify connected components of that graph; record their sizes.
    4. Permute group labels ``n_perm`` times; repeat steps 2–3 and
       record the maximum component size in each permutation.
    5. A component is significant if its size exceeds the
       (1 − alpha) percentile of the permutation null distribution.

    Parameters
    ----------
    matrices_a : np.ndarray, shape (n_a, N, N)
        FC matrices for group A.

    matrices_b : np.ndarray, shape (n_b, N, N)
        FC matrices for group B.

    threshold : float
        Primary t-statistic threshold.  A common choice is 3.0 (≈ p < .001
        uncorrected).  Higher values → smaller, more specific components.

    alpha : float
        FWER level for the permutation test.  Default 0.05.

    n_perm : int
        Number of permutations.  At least 1000 recommended for reliable
        p-values; 5000 for publication.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    sig_mask : np.ndarray of bool, shape (N, N)
        True for edges belonging to a significant NBS component.

    component_sizes : list of int
        Number of *nodes* in each observed suprathreshold component
        (sorted descending).

    null_max : np.ndarray, shape (n_perm,)
        Maximum component size from each permutation (the null distribution).

    Notes
    -----
    - NBS is designed for two-group comparisons.  For one-sample tests
      (group mean vs zero), use ``one_sample_ttest_fc`` with FDR.
    - The primary threshold is a sensitivity/specificity trade-off:
      lower → more edges enter components but correction is harder;
      higher → fewer edges, more specific components.

    Examples
    --------
    >>> sig, sizes, null = network_based_stats(
    ...     matrices_patients, matrices_controls,
    ...     threshold=3.0, alpha=0.05, n_perm=1000)
    >>> print(f'Significant edges: {sig.sum() // 2}')

    See also
    --------
    two_sample_ttest_fc : FDR correction for the same two-group test.
    plot_significant_fc : Visualise the significant-edge mask.
    """
    _require_networkx()
    rng  = np.random.default_rng(seed)
    a    = np.asarray(matrices_a, dtype=float)
    b    = np.asarray(matrices_b, dtype=float)
    na, N, _ = a.shape
    nb       = b.shape[0]
    iu       = np.triu_indices(N, k=1)
    all_mats = np.concatenate([a, b], axis=0)

    def _welch_t(mats):
        aa, bb = mats[:na], mats[na:]
        ma, mb = aa.mean(0), bb.mean(0)
        va, vb = aa.var(0, ddof=1), bb.var(0, ddof=1)
        se = np.sqrt(va / na + vb / nb)
        return np.where(se > 0, (ma - mb) / se, 0.0)

    def _max_component(t_full):
        supr = np.abs(t_full[iu]) >= threshold
        si, sj = iu[0][supr], iu[1][supr]
        G = nx.Graph()
        G.add_nodes_from(range(N))
        G.add_edges_from(zip(si.tolist(), sj.tolist()))
        comps = list(nx.connected_components(G))
        return comps, [len(c) for c in comps]

    # Observed
    t_obs              = _welch_t(all_mats)
    comps_obs, sizes_obs = _max_component(t_obs)

    # Permutation null
    null_max = np.zeros(n_perm, dtype=int)
    for p in range(n_perm):
        perm_mats = all_mats[rng.permutation(na + nb)]
        _, s_perm = _max_component(_welch_t(perm_mats))
        null_max[p] = max(s_perm) if s_perm else 0
        if (p + 1) % 200 == 0:
            print(f'  [NBS] permutation {p+1}/{n_perm} …', end='\r')
    print()

    crit = float(np.percentile(null_max, 100 * (1 - alpha)))

    # Mark edges in significant components
    sig_mask = np.zeros((N, N), dtype=bool)
    sig_components = [(c, s) for c, s in zip(comps_obs, sizes_obs) if s > crit]
    for comp, _ in sig_components:
        comp = list(comp)
        for u in comp:
            for v in comp:
                if u != v and all_mats[:na, u, v].mean() != 0:
                    sig_mask[u, v] = True

    n_sig_comp  = len(sig_components)
    n_sig_edges = int(sig_mask[iu].sum())
    print(f'  [NBS] primary threshold: |t|≥{threshold} | '
          f'critical component size: {crit:.0f} nodes')
    print(f'  [NBS] {n_sig_comp} significant component(s), '
          f'{n_sig_edges} suprathreshold edges')

    if _logger:
        _logger.log_step('network_based_stats',
                         threshold=threshold, alpha=alpha,
                         n_perm=n_perm,
                         n_sig_components=n_sig_comp,
                         n_sig_edges=n_sig_edges,
                         critical_size=round(crit, 1))

    sizes_sorted = sorted(sizes_obs, reverse=True)
    return sig_mask, sizes_sorted, null_max


@_register
def plot_significant_fc(mean_matrix, sig_mask, labels,
                         title='Group FC', vmin=-1, vmax=1,
                         cmap='RdBu_r', figsize=(13, 5)):
    """
    Plot the group mean FC matrix alongside a significance-masked version.

    Two panels are shown side by side: the full (unmasked) group mean on
    the left, and the same matrix with non-significant edges set to zero on
    the right.  This makes it easy to see which edges survive correction.

    Parameters
    ----------
    mean_matrix : np.ndarray, shape (N, N)
        Group mean FC matrix from ``compute_group_mean_fc``.

    sig_mask : np.ndarray of bool, shape (N, N)
        Significance mask from ``one_sample_ttest_fc``,
        ``two_sample_ttest_fc``, or ``network_based_stats``.

    labels : list of str
        Parcel label strings (used only for the axis tick count).

    title : str
        Figure suptitle.

    vmin, vmax : float
        Colour axis limits.  Default −1 to +1.

    cmap : str
        Matplotlib colormap.  Default ``'RdBu_r'``.

    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Examples
    --------
    >>> mean_mat, _ = compute_group_mean_fc(matrices, labels)
    >>> _, _, sig   = one_sample_ttest_fc(matrices)
    >>> fig = plot_significant_fc(mean_mat, sig, labels,
    ...                            title='Group resting-state FC')
    >>> plt.show()

    See also
    --------
    one_sample_ttest_fc : Produce sig_mask for one-sample test.
    two_sample_ttest_fc : Produce sig_mask for two-group test.
    network_based_stats : NBS sig_mask.
    """
    n_sig = int(sig_mask[np.triu_indices(len(labels), k=1)].sum())

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                              gridspec_kw={'wspace': 0.12})

    for ax, mat, subtitle in [
        (axes[0], mean_matrix,
         'Group mean FC (all edges)'),
        (axes[1], np.where(sig_mask, mean_matrix, 0.0),
         f'FDR-significant edges only\n({n_sig} edges)'),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(subtitle, fontsize=10)
        ax.set_xlabel('Node', fontsize=9)
        ax.set_ylabel('Node', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig


# =============================================================================
# Group-Level Graph Theory
# =============================================================================

@_register
def load_group_node_metrics(csv_paths, subject_ids=None):
    """
    Load node-metric CSVs from multiple subjects into a single DataFrame.

    Each CSV is the ``node_df`` saved by the graph theory tutorial (columns:
    ``label``, ``network``, ``degree``, ``strength``, ``clustering``,
    ``betweenness``).  A ``subject`` column is prepended so that the
    combined DataFrame can be grouped by subject or by parcel.

    Parameters
    ----------
    csv_paths : list of str or Path
        Paths to per-subject node metric CSV files (one per subject).

    subject_ids : list of str or None
        Subject identifiers to use as the ``subject`` column values.
        If None, subjects are labelled ``'sub-01'``, ``'sub-02'``, etc.

    Returns
    -------
    group_node_df : pd.DataFrame
        Concatenated DataFrame with shape (n_subjects × N_ROIS, 7+).

    Examples
    --------
    >>> paths = sorted(Path('graph_metrics/').glob('*_graph_metrics.csv'))
    >>> group_df = load_group_node_metrics(paths)
    >>> print(group_df.groupby('network')['strength'].mean())

    See also
    --------
    ttest_node_metric            : Per-parcel t-test on node metrics.
    plot_node_tstat_by_network   : Visualise t-statistics by network.
    """
    paths = list(csv_paths)
    ids   = (subject_ids if subject_ids is not None
             else [f'sub-{i+1:02d}' for i in range(len(paths))])
    dfs = []
    for sid, p in zip(ids, paths):
        df = pd.read_csv(p)
        df.insert(0, 'subject', sid)
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    n_sub = out['subject'].nunique()
    n_row = len(out)
    print(f'  [load_group_node_metrics] {n_sub} subjects | {n_row} rows | '
          f'columns: {list(out.columns)}')
    return out


@_register
def ttest_node_metric(group_node_df, metric='strength',
                       popmean=0.0, alpha=0.05):
    """
    One-sample t-test per parcel for a node-level graph metric.

    For each parcel, tests whether the group mean of ``metric`` differs
    significantly from ``popmean`` (default 0).  BH FDR correction is
    applied across all parcels.

    This answers: *which specific nodes show significantly non-zero
    strength / clustering / betweenness at the group level?*

    Parameters
    ----------
    group_node_df : pd.DataFrame
        Combined node metric DataFrame from ``load_group_node_metrics``.
        Must contain columns ``'label'``, ``'network'``, and ``metric``.

    metric : str
        Node-level metric to test.  One of ``'degree'``, ``'strength'``,
        ``'clustering'``, ``'betweenness'``.

    popmean : float
        Null hypothesis mean.  Default 0.

    alpha : float
        FDR significance level.  Default 0.05.

    Returns
    -------
    results_df : pd.DataFrame
        One row per parcel.  Columns: ``label``, ``network``, ``n``,
        ``mean``, ``std``, ``t``, ``p_raw``, ``p_fdr``, ``significant``.

    Examples
    --------
    >>> results = ttest_node_metric(group_df, metric='strength')
    >>> print(results[results['significant']].sort_values('t', ascending=False).head(10))

    See also
    --------
    load_group_node_metrics    : Load the input DataFrame.
    plot_node_tstat_by_network : Visualise results by network.
    """
    from scipy.stats import ttest_1samp as _tt1

    rows = []
    for (label, net), grp in group_node_df.groupby(
            ['label', 'network'], sort=False):
        vals = grp[metric].dropna().values
        if len(vals) < 2:
            continue
        t, p = _tt1(vals, popmean=popmean)
        rows.append({'label': label, 'network': net,
                     'n': len(vals), 'mean': vals.mean(),
                     'std': vals.std(ddof=1), 't': t, 'p_raw': p})

    res = pd.DataFrame(rows)
    rej, p_fdr = _fdr_bh(res['p_raw'].values, alpha=alpha)
    res['p_fdr']       = p_fdr
    res['significant'] = rej

    n_sig = int(rej.sum())
    print(f'  [ttest_node_metric] metric={metric} | '
          f'{n_sig}/{len(res)} parcels significant (FDR q<{alpha})')

    if _logger:
        _logger.log_step('ttest_node_metric',
                         metric=metric, popmean=popmean, alpha=alpha,
                         n_parcels=len(res), n_significant=n_sig)
    return res


@_register
def permutation_test_global_metric(matrices, labels, metric_name,
                                    threshold=0.10, n_perm=500, seed=42):
    """
    Test whether a group-mean graph metric exceeds a random-graph null.

    For each permutation, generates an Erdős–Rényi random graph matched
    to the observed network density, computes the graph metric, and builds
    a null distribution.  The observed group mean is then compared to this
    distribution.

    This is the standard approach for assessing **small-world** properties:
    brain networks are expected to show higher clustering coefficient and
    comparable path length relative to matched random graphs.

    Parameters
    ----------
    matrices : np.ndarray, shape (n_subjects, N, N)
        Stacked FC matrices from ``load_group_matrices``.

    labels : list of str
        Length-N parcel label strings.

    metric_name : str
        Graph-level metric to test.  Must be a key returned by
        ``compute_graph_metrics`` graph_dict:
        ``'density'``, ``'mean_clustering'``, ``'mean_strength'``,
        ``'transitivity'``, ``'avg_path_length'``, or ``'mean_degree'``.

    threshold : float
        Proportional threshold used when constructing each subject's graph.

    n_perm : int
        Number of random-graph permutations.  Default 500.
        Use ≥1000 for publication-quality inference.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    obs_mean : float
        Observed group-mean value of ``metric_name``.
    obs_per_subject : list of float
        Per-subject metric values (useful for confidence intervals).
    null_dist : np.ndarray, shape (n_perm,)
        Metric values from random graphs (the null distribution).
    p_perm : float
        Permutation p-value: proportion of null values ≥ obs_mean.

    Notes
    -----
    - ``avg_path_length`` uses only the largest connected component and
      can be slow for dense graphs.  Consider using ``mean_clustering``
      for faster permutation runs.
    - For the small-world coefficient σ = (C/C_rand) / (L/L_rand), run
      this function separately for clustering and path length and divide.

    Examples
    --------
    >>> obs, per_sub, null, p = permutation_test_global_metric(
    ...     matrices, labels, 'mean_clustering', threshold=0.10, n_perm=500)
    >>> print(f'Observed: {obs:.4f}  |  p (vs random) = {p:.4f}')

    See also
    --------
    compute_graph_metrics : Compute the metric on a single subject.
    run_threshold_sweep   : Metric stability across thresholds.
    """
    import contextlib, io
    _require_networkx()
    rng  = np.random.default_rng(seed)
    mats = np.asarray(matrices, dtype=float)

    # Observed: one value per subject
    print(f'  [permutation] Computing observed {metric_name} …')
    obs_per_subject = []
    for mat in mats:
        adj = threshold_proportional(mat, proportion=threshold,
                                     positive_only=True)
        with contextlib.redirect_stdout(io.StringIO()):
            _, gd = compute_graph_metrics(adj, labels)
        obs_per_subject.append(gd.get(metric_name, np.nan))
    obs_mean = float(np.nanmean(obs_per_subject))

    # Matched random-graph density from the first subject
    adj0   = threshold_proportional(mats[0], proportion=threshold,
                                    positive_only=True)
    N      = adj0.shape[0]
    p_rand = float((adj0 > 0).sum()) / (N * (N - 1))   # density

    # Null distribution: Erdős–Rényi random graphs
    print(f'  [permutation] Running {n_perm} random-graph permutations '
          f'(N={N}, density≈{p_rand:.4f}) …')
    null_dist = np.zeros(n_perm)
    for i in range(n_perm):
        G_r   = nx.erdos_renyi_graph(N, p_rand,
                                     seed=int(rng.integers(1_000_000)))
        adj_r = nx.to_numpy_array(G_r, dtype=float)
        with contextlib.redirect_stdout(io.StringIO()):
            _, gd = compute_graph_metrics(adj_r, labels)
        null_dist[i] = gd.get(metric_name, np.nan)
        if (i + 1) % 100 == 0:
            print(f'  [permutation]   {i+1}/{n_perm}', end='\r')
    print()

    # One-tailed p: fraction of null ≥ observed
    p_perm     = float(np.nanmean(null_dist >= obs_mean))
    null_mean  = float(np.nanmean(null_dist))
    null_std   = float(np.nanstd(null_dist))
    print(f'  [permutation] Observed {metric_name:20s}: {obs_mean:.4f}')
    print(f'  [permutation] Random null (mean±SD)   : '
          f'{null_mean:.4f} ± {null_std:.4f}')
    print(f'  [permutation] p (observed ≥ random)   : {p_perm:.4f}')

    if _logger:
        _logger.log_step('permutation_test_global_metric',
                         metric_name=metric_name, threshold=threshold,
                         n_perm=n_perm, obs_mean=round(obs_mean, 4),
                         null_mean=round(null_mean, 4),
                         p_perm=round(p_perm, 4))

    return obs_mean, obs_per_subject, null_dist, p_perm


@_register
def plot_node_tstat_by_network(ttest_df, metric='strength',
                                alpha=0.05, figsize=(11, 5)):
    """
    Two-panel figure: mean t-statistic and percentage of significant parcels
    per Yeo network, after ``ttest_node_metric``.

    The left panel shows the mean t-statistic across parcels in each
    network (positive = above the null mean; red bars = below null).
    The right panel shows the percentage of parcels in each network whose
    FDR-adjusted p-value survives the significance threshold.

    Parameters
    ----------
    ttest_df : pd.DataFrame
        Output of ``ttest_node_metric``.  Must contain columns
        ``'network'``, ``'t'``, ``'significant'``, ``'label'``.

    metric : str
        Metric label for the figure title (purely cosmetic).

    alpha : float
        Alpha level label for the right-panel title (purely cosmetic).

    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Examples
    --------
    >>> results = ttest_node_metric(group_df, metric='strength')
    >>> fig = plot_node_tstat_by_network(results, metric='strength')
    >>> plt.show()

    See also
    --------
    ttest_node_metric : Produce the ttest_df argument.
    """
    net = (ttest_df
           .groupby('network')
           .agg(mean_t=('t', 'mean'),
                n_sig =('significant', 'sum'),
                n_tot =('label', 'count'))
           .reset_index())
    net['pct_sig'] = 100.0 * net['n_sig'] / net['n_tot']
    net = net.sort_values('mean_t', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: mean t by network
    colors = ['tomato' if t < 0 else 'steelblue' for t in net['mean_t']]
    axes[0].bar(net['network'], net['mean_t'], color=colors, edgecolor='white')
    axes[0].axhline(0, color='black', lw=0.8, ls='--')
    axes[0].set_title(f'Mean t-statistic by network\n(metric: {metric})',
                      fontsize=10)
    axes[0].set_ylabel('Mean t', fontsize=9)
    axes[0].tick_params(axis='x', rotation=40)
    axes[0].grid(axis='y', lw=0.4, alpha=0.5)

    # Right: % significant parcels
    axes[1].bar(net['network'], net['pct_sig'],
                color='steelblue', edgecolor='white')
    axes[1].set_title(f'% significant parcels by network\n'
                      f'(FDR q < {alpha})', fontsize=10)
    axes[1].set_ylabel('% significant parcels', fontsize=9)
    axes[1].tick_params(axis='x', rotation=40)
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis='y', lw=0.4, alpha=0.5)

    plt.tight_layout()
    return fig


# =============================================================================
# GLM-based group FC statistics
# =============================================================================

@_register
def glm_fc(matrices, design_matrix, contrast, alpha=0.05):
    """
    Mass-univariate OLS GLM across all FC edges with BH-FDR correction.

    Fits Y = X @ beta + epsilon independently at every upper-triangle edge,
    then tests the scalar contrast c'beta against zero using a t-statistic.
    This unified framework handles one-sample tests, group comparisons,
    continuous covariates, and any factorial design expressible as a design
    matrix.

    Parameters
    ----------
    matrices : np.ndarray, shape (n_subjects, N, N)
        Stacked subject FC matrices from ``load_group_matrices``.

    design_matrix : np.ndarray or pd.DataFrame, shape (n_subjects, n_regressors)
        The GLM design matrix X.  Each row is one subject; each column is one
        regressor.

        *One-sample t-test* (test whether mean FC != 0):
            X = np.ones((n, 1))
            contrast = [1]

        *Two-sample t-test* (group A=0, group B=1):
            X = np.column_stack([np.ones(n), group_vec])
            contrast = [0, 1]

        *Continuous covariate* (e.g. age, symptom score):
            X = np.column_stack([np.ones(n), covariate_z])
            contrast = [0, 1]

    contrast : array-like, shape (n_regressors,)
        Contrast vector c.  The tested quantity is c @ beta.

    alpha : float
        FDR significance level (Benjamini-Hochberg).  Default 0.05.

    Returns
    -------
    t_mat : np.ndarray, shape (N, N)
        Symmetric matrix of contrast t-statistics.

    p_fdr_mat : np.ndarray, shape (N, N)
        Symmetric matrix of BH-adjusted p-values.

    sig_mask : np.ndarray of bool, shape (N, N)
        True where the edge survives FDR correction.

    Examples
    --------
    One-sample:

    >>> X = np.ones((n_subjects, 1))
    >>> t, p, sig = glm_fc(matrices, X, contrast=[1])

    Two-sample:

    >>> X = np.column_stack([np.ones(n), group_labels])
    >>> t, p, sig = glm_fc(matrices, X, contrast=[0, 1])

    See also
    --------
    load_group_matrices : Load and stack individual-subject FC CSVs.
    plot_significant_fc : Heatmap view of significant edges.
    plot_chord_diagram  : Chord/ring diagram of significant edges.
    """
    from scipy.stats import t as _tdist

    X   = np.array(design_matrix, dtype=float)
    c   = np.array(contrast, dtype=float)
    n_sub, N, _ = matrices.shape
    n_reg = X.shape[1]
    df    = n_sub - n_reg

    if df <= 0:
        raise ValueError(
            f'Degrees of freedom = {df} (n_subjects={n_sub}, '
            f'n_regressors={n_reg}). Add more subjects or reduce regressors.'
        )
    if X.shape[0] != n_sub:
        raise ValueError(
            f'design_matrix has {X.shape[0]} rows but matrices has '
            f'{n_sub} subjects.'
        )
    if len(c) != n_reg:
        raise ValueError(
            f'contrast length {len(c)} != n_regressors {n_reg}.'
        )

    iu      = np.triu_indices(N, k=1)
    n_edges = len(iu[0])

    # Stack upper triangle: (n_sub, n_edges)
    Y = matrices[:, iu[0], iu[1]]

    # OLS solution
    XtX_inv = np.linalg.inv(X.T @ X)       # (n_reg, n_reg)
    beta    = XtX_inv @ (X.T @ Y)           # (n_reg, n_edges)

    # Residual variance
    resid  = Y - X @ beta
    sigma2 = np.sum(resid ** 2, axis=0) / df   # (n_edges,)

    # Contrast t-statistic
    c_beta = c @ beta                       # (n_edges,)
    c_var  = float(c @ XtX_inv @ c)        # scalar
    se     = np.sqrt(np.maximum(sigma2 * c_var, 0.0))
    t_vec  = np.where(se > 0, c_beta / se, 0.0)
    p_vec  = 2.0 * _tdist.sf(np.abs(t_vec), df=df)

    # BH-FDR on upper triangle
    rej, p_fdr_vec = _fdr_bh(p_vec, alpha=alpha)

    # Reconstruct symmetric (N, N) matrices
    t_mat     = np.zeros((N, N))
    p_fdr_mat = np.ones((N, N))
    sig_mask  = np.zeros((N, N), dtype=bool)

    t_mat[iu]              = t_vec
    t_mat[(iu[1], iu[0])] = t_vec
    p_fdr_mat[iu]              = p_fdr_vec
    p_fdr_mat[(iu[1], iu[0])] = p_fdr_vec
    sig_mask[iu]               = rej
    sig_mask[(iu[1], iu[0])]   = rej

    n_sig = int(rej.sum())
    print(f'  [glm_fc] contrast={list(c)}  df={df}  '
          f'{n_sig}/{n_edges} edges significant '
          f'(FDR q<{alpha}, {100*n_sig/n_edges:.1f}%)')

    if _logger:
        _logger.log_step('glm_fc', n_subjects=n_sub, n_regressors=n_reg,
                         contrast=list(c), alpha=alpha,
                         n_edges_tested=n_edges, n_significant=n_sig)

    return t_mat, p_fdr_mat, sig_mask


@_register
def plot_chord_diagram(sig_mask, labels, mean_matrix=None,
                       title='Significant FC — Chord Diagram',
                       positive_color='#D62728', negative_color='#1F77B4',
                       figsize=(9, 9), output_path=None):
    """
    Draw a circular chord diagram of significant FC edges grouped by network.

    Nodes are arranged on a ring, sorted and colour-coded by their resting-
    state network (parsed from Schaefer-style labels).  Significant edges are
    drawn as straight chords inside the ring, coloured by sign when a mean
    matrix is supplied.

    Parameters
    ----------
    sig_mask : np.ndarray of bool, shape (N, N)
        Significance mask from ``glm_fc``, ``one_sample_ttest_fc``, etc.

    labels : list of str, length N
        Parcel label strings.  Schaefer labels (``7Networks_LH_Default_PFC_1``)
        are parsed to extract the network name (``Default``).

    mean_matrix : np.ndarray, shape (N, N) or None
        Group mean FC matrix.  If supplied, chord colour encodes sign and
        alpha/width scale with |mean FC|.  If None, chords are grey.

    title : str
        Figure title.

    positive_color, negative_color : str
        Colours for positive and negative significant edges.

    figsize : tuple
        Figure size in inches.

    output_path : str or None
        If provided, save the figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Examples
    --------
    >>> fig = plot_chord_diagram(sig_mask, labels, mean_matrix=mean_fc,
    ...                          title='Group mean FC — significant edges')
    >>> plt.show()

    See also
    --------
    glm_fc             : Produce sig_mask.
    plot_significant_fc : Heatmap view of the same mask.
    """
    import matplotlib.patches as mpatches

    N  = len(labels)
    iu = np.triu_indices(N, k=1)

    # ── Network assignment ────────────────────────────────────────────────────
    def _net(lbl):
        parts = lbl.split('_')
        if len(parts) >= 3 and 'Networks' in parts[0]:
            return parts[2]
        return parts[0]

    node_nets   = [_net(l) for l in labels]
    unique_nets = list(dict.fromkeys(node_nets))   # order-preserving

    # Sort nodes so network blocks are contiguous on the ring
    sort_order = sorted(range(N), key=lambda i: node_nets[i])
    inv_order  = [0] * N
    for pos, orig in enumerate(sort_order):
        inv_order[orig] = pos

    # ── Colour palette ────────────────────────────────────────────────────────
    cmap_nets = plt.cm.get_cmap('Set2', max(len(unique_nets), 3))
    net_color = {n: cmap_nets(i) for i, n in enumerate(unique_nets)}

    # ── Angular geometry ──────────────────────────────────────────────────────
    gap_frac   = 0.012                        # gap between network blocks (frac of 2pi)
    total_gap  = gap_frac * 2 * np.pi * len(unique_nets)
    arc_total  = 2 * np.pi - total_gap
    arc_node   = arc_total / N

    # Starting angle for each network block
    net_start = {}
    cursor    = 0.0
    for net in unique_nets:
        net_start[net] = cursor
        n_in_net        = sum(1 for nn in node_nets if nn == net)
        cursor         += arc_node * n_in_net + gap_frac * 2 * np.pi

    # Mid-angle for each node
    node_angle  = np.zeros(N)
    net_counts  = {n: 0 for n in unique_nets}
    for pos, orig in enumerate(sort_order):
        net               = node_nets[orig]
        node_angle[pos]   = net_start[net] + net_counts[net] * arc_node + arc_node / 2
        net_counts[net]  += 1

    R_node  = 1.00   # node ring
    R_arc   = 1.07   # network arc band
    R_lbl   = 1.20   # label radius

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')

    n_sig_edges = int(sig_mask[iu].sum())

    # Chords (drawn first, behind nodes)
    for k in range(len(iu[0])):
        i_orig, j_orig = int(iu[0][k]), int(iu[1][k])
        if not sig_mask[i_orig, j_orig]:
            continue

        a_i = node_angle[inv_order[i_orig]]
        a_j = node_angle[inv_order[j_orig]]
        xi, yi = R_node * np.cos(a_i), R_node * np.sin(a_i)
        xj, yj = R_node * np.cos(a_j), R_node * np.sin(a_j)

        if mean_matrix is not None:
            val  = float(mean_matrix[i_orig, j_orig])
            col  = positive_color if val >= 0 else negative_color
            alph = np.clip(0.12 + 0.55 * abs(val), 0, 0.85)
            lw   = np.clip(0.25 + 1.50 * abs(val), 0.25, 2.5)
        else:
            col, alph, lw = '#888888', 0.25, 0.5

        ax.plot([xi, xj], [yi, yj], color=col, alpha=alph,
                linewidth=lw, zorder=1, solid_capstyle='round')

    # Network arcs and node dots
    for net in unique_nets:
        members_pos = [inv_order[i] for i in range(N) if node_nets[i] == net]
        if not members_pos:
            continue
        col = net_color[net]

        a_s = node_angle[min(members_pos)] - arc_node / 2
        a_e = node_angle[max(members_pos)] + arc_node / 2
        theta = np.linspace(a_s, a_e, max(len(members_pos) * 3, 60))
        ax.plot(R_arc * np.cos(theta), R_arc * np.sin(theta),
                color=col, linewidth=6, solid_capstyle='butt', zorder=2)

        for pos in members_pos:
            a = node_angle[pos]
            ax.scatter(R_node * np.cos(a), R_node * np.sin(a),
                       s=14, color=col, zorder=4, linewidths=0)

        # Label at midpoint of arc
        a_mid = (a_s + a_e) / 2
        lx, ly = R_lbl * np.cos(a_mid), R_lbl * np.sin(a_mid)
        rot    = np.degrees(a_mid) % 360
        if 90 < rot < 270:
            rot += 180
        ax.text(lx, ly, net, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color=col,
                rotation=rot, rotation_mode='anchor', zorder=5)

    # Edge colour legend
    if mean_matrix is not None:
        legend_handles = [
            mpatches.Patch(color=positive_color, label='Positive FC'),
            mpatches.Patch(color=negative_color, label='Negative FC'),
        ]
        ax.legend(handles=legend_handles, loc='lower right',
                  bbox_to_anchor=(1.3, 0.0), fontsize=9, frameon=False)

    ax.set_title(f'{title}\n({n_sig_edges // 2} significant edges)',
                 fontsize=12, pad=14)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'  [plot_chord_diagram] Saved -> {output_path}')

    return fig
