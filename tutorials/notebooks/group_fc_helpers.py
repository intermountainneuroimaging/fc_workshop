"""
group_fc_helpers.py
===================
Group-level functional connectivity analysis helpers.

Companion to fc_helpers.py — import both for the full workshop toolkit:
    from fc_helpers import plot_fc_matrix, compute_fc, ...
    from group_fc_helpers import *

Function index
--------------
Data preparation
    aggregate_subject_runs      Average FC matrices across multiple runs/sessions
    save_subject_fc             Save a subject-level FC matrix to CSV

Loading data
    load_fc_from_file_list      Stack matrices listed in a .txt file (1 path per line)
    load_design_matrix          Read a design matrix from CSV, optionally aligned by subject ID

Group statistics
    permutation_glm             Mass-univariate OLS GLM with permutation-based FWE and FDR
    group_fc_help               Print full docstring for any function in this module

Visualisation
    plot_corrected_maps         Four-panel figure: t-stat | uncorrected | FDR | FWE
    plot_null_distribution      Permutation null distribution of max|t|

Saving
    save_edge_results           Upper-triangle edge table with t, p_unc, p_fdr, p_fwe

Dependencies: numpy, pandas, scipy, matplotlib, seaborn, statsmodels
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy import stats as _scipy_stats


# ══════════════════════════════════════════════════════════════════════════════
# Help browser
# ══════════════════════════════════════════════════════════════════════════════

def group_fc_help(func_name=None):
    """
    Print documentation for a group_fc_helpers function.

    Usage
    -----
    group_fc_help()                          # list all functions
    group_fc_help('permutation_glm')         # full docstring for one function
    """
    _funcs = {
        'aggregate_subject_runs': aggregate_subject_runs,
        'save_subject_fc':        save_subject_fc,
        'load_fc_from_file_list': load_fc_from_file_list,
        'load_design_matrix':     load_design_matrix,
        'permutation_glm':        permutation_glm,
        'plot_corrected_maps':    plot_corrected_maps,
        'plot_null_distribution': plot_null_distribution,
        'save_edge_results':      save_edge_results,
    }
    if func_name is None:
        print("group_fc_helpers — available functions\n")
        for name in _funcs:
            first_line = (_funcs[name].__doc__ or '').strip().split('\n')[0]
            print(f"  {name:<30} {first_line}")
        print("\nCall group_fc_help('function_name') for full docs.")
    else:
        fn = _funcs.get(func_name)
        if fn is None:
            print(f"Unknown function '{func_name}'. Call group_fc_help() to list all.")
        else:
            print(fn.__doc__)


# ══════════════════════════════════════════════════════════════════════════════
# Data preparation
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_subject_runs(run_paths, method='mean'):
    """
    Average FC matrices across multiple runs or sessions for one subject.

    Call this once per subject when you have more than one run or session.
    The output is a single (N × N) subject-level FC matrix ready to stack
    into a group array.

    Parameters
    ----------
    run_paths : list of str or Path
        Ordered list of FC matrix CSV paths — one per run or session.
        Each CSV must have identical row/column labels (parcel names).
    method : {'mean'}
        Aggregation method.  'mean' computes the element-wise average across
        all runs.  Fisher-z averaging is **not** applied here; if your FC
        matrices were stored as raw Pearson r values and you want proper
        averaging, z-transform before calling this function and invert
        afterwards.

    Returns
    -------
    fc_agg : ndarray, shape (N, N)
        Aggregated FC matrix.
    labels : list of str
        Parcel labels (taken from the first matrix).

    Raises
    ------
    ValueError
        If parcel labels are inconsistent across runs.

    Examples
    --------
    >>> import glob
    >>> run_paths = sorted(glob.glob('derivatives/sub-01/func/sub-01_run-*_fc.csv'))
    >>> fc_mean, labels = aggregate_subject_runs(run_paths)
    """
    if len(run_paths) == 0:
        raise ValueError("run_paths is empty.")

    matrices = []
    ref_labels = None
    for p in run_paths:
        df = pd.read_csv(p, index_col=0)
        if ref_labels is None:
            ref_labels = list(df.columns)
        else:
            if list(df.columns) != ref_labels:
                raise ValueError(
                    f"Label mismatch in {p}.\n"
                    f"Expected: {ref_labels[:3]}...\n"
                    f"Got:      {list(df.columns)[:3]}..."
                )
        matrices.append(df.values)

    stack = np.array(matrices)  # (n_runs, N, N)
    if method == 'mean':
        fc_agg = stack.mean(axis=0)
    else:
        raise ValueError(f"Unsupported aggregation method: '{method}'. Use 'mean'.")

    return fc_agg, ref_labels


def save_subject_fc(fc_matrix, labels, output_path):
    """
    Save a subject-level FC matrix to a labelled CSV file.

    Parameters
    ----------
    fc_matrix : ndarray, shape (N, N)
    labels : list of str
        Parcel names — used as both row index and column headers.
    output_path : str or Path
        Full path including filename, e.g. 'derivatives/sub-01/sub-01_fc.csv'.
        Parent directories are created automatically.

    Examples
    --------
    >>> save_subject_fc(fc_mean, labels, 'derivatives/sub-01/sub-01_fc.csv')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fc_matrix, index=labels, columns=labels).to_csv(output_path)
    print(f"Saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Loading data
# ══════════════════════════════════════════════════════════════════════════════

def load_fc_from_file_list(txt_path):
    """
    Load and stack FC matrices listed in a plain-text file (one path per line).

    File format
    -----------
    Each non-empty, non-comment line is treated as an absolute or relative
    path to a subject-level FC matrix CSV.  Lines beginning with '#' are
    ignored.

    Example file (fc_inputs.txt)
    ----------------------------
        # Group A — controls
        /data/derivatives/sub-01/sub-01_fc.csv
        /data/derivatives/sub-02/sub-02_fc.csv
        # Group B — patients
        /data/derivatives/sub-03/sub-03_fc.csv

    Parameters
    ----------
    txt_path : str or Path
        Path to the plain-text file listing FC matrix CSVs.

    Returns
    -------
    matrices : ndarray, shape (n_subjects, N, N)
        Stacked FC matrices.
    labels : list of str
        Parcel labels (verified to be identical across all subjects).
    subject_ids : list of str
        Filename stems (e.g. 'sub-01_fc' from 'sub-01_fc.csv').

    Raises
    ------
    FileNotFoundError
        If txt_path or any listed CSV does not exist.
    ValueError
        If parcel labels differ between subjects.

    Examples
    --------
    >>> matrices, labels, sids = load_fc_from_file_list('fc_inputs.txt')
    >>> print(matrices.shape)   # (n_subjects, N, N)
    """
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Input list not found: {txt_path}")

    paths = []
    with open(txt_path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith('#'):
                paths.append(Path(line))

    if len(paths) == 0:
        raise ValueError(f"No valid paths found in {txt_path}")

    matrices, ref_labels = [], None
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"FC matrix not found: {p}")
        df = pd.read_csv(p, index_col=0)
        if ref_labels is None:
            ref_labels = list(df.columns)
        else:
            if list(df.columns) != ref_labels:
                raise ValueError(f"Label mismatch in {p}")
        matrices.append(df.values)

    subject_ids = [p.stem for p in paths]
    print(f"Loaded {len(matrices)} subjects | {len(ref_labels)} parcels each")
    return np.array(matrices), ref_labels, subject_ids


def load_design_matrix(csv_path, subject_ids=None):
    """
    Load a design matrix from a CSV file.

    File format
    -----------
    Rows = subjects.  Columns = regressors.  The first column may optionally
    be named 'subject_id' to allow row alignment by subject identifier.

    One-sample t-test (test whether group mean FC ≠ 0)
    ---------------------------------------------------
    subject_id,intercept
    sub-01,1
    sub-02,1
    sub-03,1

    Two-sample t-test (group A=0, group B=1)
    -----------------------------------------
    subject_id,intercept,group
    sub-01,1,0
    sub-02,1,0
    sub-03,1,1
    sub-04,1,1

    Covariate / continuous regression
    -----------------------------------
    subject_id,intercept,age_z
    sub-01,1,-1.23
    sub-02,1, 0.45
    sub-03,1, 0.78

    Multiple regressors (ANCOVA style)
    ------------------------------------
    subject_id,intercept,group,age_z,fd_mean_z
    sub-01,1,0,-1.23,0.12
    sub-02,1,0, 0.45,0.08
    sub-03,1,1, 0.78,0.25

    Note: Always z-score continuous covariates before saving the CSV so that
    the intercept represents the group mean at the average covariate value.

    Parameters
    ----------
    csv_path : str or Path
    subject_ids : list of str, optional
        If the CSV has a 'subject_id' column and subject_ids is provided,
        rows are reordered to match subject_ids.  This ensures alignment with
        the matrix stack returned by load_fc_from_file_list().

    Returns
    -------
    X : ndarray, shape (n_subjects, n_regressors)
        Float design matrix.
    column_names : list of str
        Regressor names.

    Examples
    --------
    >>> X, cols = load_design_matrix('design_matrix.csv', subject_ids=sids)
    >>> print(cols)   # ['intercept', 'group']
    """
    dm = pd.read_csv(csv_path)
    if 'subject_id' in dm.columns:
        dm = dm.set_index('subject_id')
        if subject_ids is not None:
            missing = set(subject_ids) - set(dm.index)
            if missing:
                raise ValueError(f"subject_ids not found in design matrix: {missing}")
            dm = dm.loc[subject_ids]
    column_names = list(dm.columns)
    X = dm.values.astype(float)
    print(f"Design matrix: {X.shape[0]} subjects × {X.shape[1]} regressors  {column_names}")
    return X, column_names


# ══════════════════════════════════════════════════════════════════════════════
# Permutation GLM
# ══════════════════════════════════════════════════════════════════════════════

def permutation_glm(matrices, design_matrix, contrast,
                    n_perm=5000, alpha=0.05,
                    perm_method='auto', random_state=42):
    """
    Mass-univariate OLS GLM with permutation-based FWE and BH-FDR correction.

    Statistical model
    -----------------
    Y  = X @ beta + epsilon          Y: (n_subjects × n_edges)
    beta_hat = (X'X)^-1 X'Y
    t = c'beta_hat / sqrt(sigma2 * c'(X'X)^-1 c)

    where sigma2 = RSS / (n - rank(X)) and c is the contrast vector.

    Permutation scheme
    ------------------
    'shuffle'
        Randomly permute rows of the design matrix X.
        Valid for: two-sample tests, regression against a continuous variable.
        Assumption: exchangeability of observations under H0 (i.e. rows are
        i.i.d. under the null).

    'sign_flip'
        Randomly flip the sign of each subject's data vector (+1 or -1).
        Valid for: one-sample tests where X = [intercept].
        Assumption: the data distribution is symmetric around zero under H0.

    'auto' (default)
        Uses 'sign_flip' if the design matrix is a single intercept column
        (all values equal to 1), otherwise uses 'shuffle'.

    Multiple comparison correction
    ------------------------------
    FWE (family-wise error rate) — max-t method
        At each permutation, record max|t| across all edges.
        FWE-corrected p-value for edge e = proportion of permutations where
        max|t| ≥ |t_obs(e)|.  Controls FWER strongly under exchangeability.

    FDR (false discovery rate) — Benjamini–Hochberg
        Applied to the parametric (t-distribution) p-values.
        Controls the expected proportion of false discoveries, not the
        probability of any false discovery.  More powerful than FWE.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  DESIGN LIMITATIONS — read before using                         ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  ✗  Repeated measures / paired designs                          ║
    ║     Permuting rows breaks the pairing structure.  Use a         ║
    ║     sign-flip approach on the pair-wise difference instead.     ║
    ║                                                                  ║
    ║  ✗  Nuisance covariates in the model                            ║
    ║     Simple row-shuffle does not properly control nuisance.      ║
    ║     Use Freedman–Lane permutation: regress out nuisance from Y, ║
    ║     permute residuals, add fitted nuisance back, refit model.   ║
    ║     (Not implemented here.)                                      ║
    ║                                                                  ║
    ║  ✗  Hierarchical / mixed-effects structures                     ║
    ║     Subjects from multiple sites, families, or scanners are     ║
    ║     not exchangeable.  Use a multi-level permutation scheme.    ║
    ║                                                                  ║
    ║  ✗  Very small n                                                 ║
    ║     With n < 8 per group, the permutation space is too small    ║
    ║     to achieve p < 0.05 reliably.                               ║
    ║                                                                  ║
    ║  ✓  FWE stationarity assumption                                  ║
    ║     Max-t FWE assumes the null distribution of t is the same    ║
    ║     at every edge.  This approximately holds for FC matrices     ║
    ║     within a parcel set but may be violated across very         ║
    ║     different edge types (e.g. short- vs long-range).           ║
    ║                                                                  ║
    ║  Minimum achievable p-value = 1 / (n_perm + 1).                ║
    ║  Use n_perm ≥ 5000 for publication (10 000 for FWE maps).       ║
    ╚══════════════════════════════════════════════════════════════════╝

    Parameters
    ----------
    matrices : ndarray, shape (n_subjects, N, N)
        Stack of subject FC matrices (symmetric, diagonal zero).
    design_matrix : array-like, shape (n_subjects, n_regressors)
        GLM design matrix X.  Rows must match subject order in matrices.
    contrast : array-like, shape (n_regressors,)
        Contrast vector c.  Example for two-sample group difference:
        if X = [intercept, group], use c = [0, 1].
    n_perm : int
        Number of permutations.  Use 500 for development, 5000+ for analysis.
    alpha : float
        Significance threshold for all three correction methods.
    perm_method : {'auto', 'shuffle', 'sign_flip'}
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    results : dict
        't_mat'       : ndarray (N, N)  observed t-statistics (symmetric)
        'p_unc'       : ndarray (N, N)  uncorrected two-tailed p-values
        'p_fdr'       : ndarray (N, N)  BH FDR-adjusted p-values
        'p_fwe'       : ndarray (N, N)  max-t FWE-corrected p-values
        'sig_unc'     : ndarray (N, N) bool  p_unc < alpha
        'sig_fdr'     : ndarray (N, N) bool  p_fdr < alpha
        'sig_fwe'     : ndarray (N, N) bool  p_fwe < alpha
        'null_max_t'  : ndarray (n_perm,)  null distribution of max|t|
        'df'          : int  residual degrees of freedom
        'perm_method' : str  method actually used

    Examples
    --------
    >>> # One-sample: test group mean FC ≠ 0
    >>> X = np.ones((n, 1))
    >>> c = [1]
    >>> results = permutation_glm(matrices, X, c, n_perm=5000)

    >>> # Two-sample: group B > group A
    >>> X = np.column_stack([np.ones(n), group_labels])  # group: 0=A, 1=B
    >>> c = [0, 1]
    >>> results = permutation_glm(matrices, X, c, n_perm=5000)

    >>> # Continuous regression (e.g. symptom score)
    >>> X = np.column_stack([np.ones(n), z_scores])
    >>> c = [0, 1]
    >>> results = permutation_glm(matrices, X, c, n_perm=5000)
    """
    from scipy.stats import t as _tdist

    try:
        from statsmodels.stats.multitest import multipletests as _multipletests
    except ImportError:
        raise ImportError("statsmodels is required: pip install statsmodels")

    rng   = np.random.default_rng(random_state)
    X     = np.array(design_matrix, dtype=float)
    c     = np.array(contrast, dtype=float)
    n, N, _ = matrices.shape
    rank  = np.linalg.matrix_rank(X)
    df    = n - rank

    if df <= 0:
        raise ValueError(
            f"Degrees of freedom = {df}. Model has more parameters ({rank}) "
            f"than subjects ({n}). Reduce model complexity."
        )

    # Auto-select permutation method
    if perm_method == 'auto':
        is_intercept_only = (X.shape[1] == 1 and np.allclose(X, 1.0))
        perm_method = 'sign_flip' if is_intercept_only else 'shuffle'
    print(f"Permutation method : {perm_method}")
    print(f"n_subjects         : {n}   df = {df}")
    print(f"n_edges            : {N*(N-1)//2}   n_perm = {n_perm}")

    iu = np.triu_indices(N, k=1)
    Y  = matrices[:, iu[0], iu[1]]        # (n_subjects, n_edges)

    # ── OLS fit (vectorised across all edges) ─────────────────────────────────
    XtX_inv  = np.linalg.inv(X.T @ X)
    c_var    = float(c @ XtX_inv @ c)     # scalar, same for all edges

    def _tstat(X_, Y_):
        beta   = XtX_inv @ (X_.T @ Y_)   # (n_reg, n_edges)
        resid  = Y_ - X_ @ beta
        sigma2 = (resid ** 2).sum(axis=0) / df
        c_beta = c @ beta                  # (n_edges,)
        se     = np.sqrt(np.maximum(sigma2 * c_var, 0.0))
        return np.where(se > 0, c_beta / se, 0.0)

    t_obs = _tstat(X, Y)

    # ── Permutation null ───────────────────────────────────────────────────────
    null_max_t = np.empty(n_perm)
    for i in range(n_perm):
        if perm_method == 'shuffle':
            X_perm = X[rng.permutation(n)]
            t_perm = _tstat(X_perm, Y)
        else:                              # sign_flip
            signs  = rng.choice([-1.0, 1.0], size=(n, 1))
            t_perm = _tstat(X, Y * signs)
        null_max_t[i] = np.abs(t_perm).max()

    # ── Uncorrected p-values (parametric, two-tailed) ─────────────────────────
    p_unc_vec = 2.0 * _tdist.sf(np.abs(t_obs), df=df)

    # ── FWE: proportion of null max|t| >= observed |t| ───────────────────────
    p_fwe_vec = np.array(
        [(null_max_t >= abs(ti)).sum() / n_perm for ti in t_obs]
    )
    p_fwe_vec = np.clip(p_fwe_vec, 1.0 / (n_perm + 1), 1.0)

    # ── FDR: Benjamini–Hochberg ───────────────────────────────────────────────
    _, p_fdr_vec, _, _ = _multipletests(p_unc_vec, alpha=alpha, method='fdr_bh')

    # ── Symmetrise to (N, N) ──────────────────────────────────────────────────
    def _sym(vec, fill_diag=1.0):
        m = np.zeros((N, N))
        m[iu] = vec
        m += m.T
        np.fill_diagonal(m, fill_diag)
        return m

    t_mat = _sym(t_obs,     fill_diag=0.0)
    p_unc = _sym(p_unc_vec, fill_diag=1.0)
    p_fdr = _sym(p_fdr_vec, fill_diag=1.0)
    p_fwe = _sym(p_fwe_vec, fill_diag=1.0)

    n_unc = int((p_unc_vec < alpha).sum())
    n_fdr = int((p_fdr_vec < alpha).sum())
    n_fwe = int((p_fwe_vec < alpha).sum())
    print(f"\nSignificant edges (α={alpha}):")
    print(f"  Uncorrected : {n_unc:>6,}")
    print(f"  FDR         : {n_fdr:>6,}")
    print(f"  FWE (max-t) : {n_fwe:>6,}")

    return {
        't_mat'      : t_mat,
        'p_unc'      : p_unc,
        'p_fdr'      : p_fdr,
        'p_fwe'      : p_fwe,
        'sig_unc'    : p_unc < alpha,
        'sig_fdr'    : p_fdr < alpha,
        'sig_fwe'    : p_fwe < alpha,
        'null_max_t' : null_max_t,
        'df'         : df,
        'perm_method': perm_method,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def plot_corrected_maps(results, labels=None, alpha=0.05,
                        vmax=None, figsize=(20, 5), output_path=None):
    """
    Four-panel figure: t-statistic | uncorrected | FDR corrected | FWE corrected.

    Significant edges are shown at their observed t-value; non-significant
    edges are set to zero so the thresholded maps are directly comparable.

    Parameters
    ----------
    results : dict
        Output of permutation_glm().
    labels : list of str, optional
        Parcel labels for axis tick marks.  Only displayed when N ≤ 20.
    alpha : float
        Threshold displayed in panel titles.
    vmax : float, optional
        Symmetric colour scale limit for the t-statistic.  Defaults to the
        98th percentile of |t| in the upper triangle.
    figsize : tuple
    output_path : str, optional
        If provided, the figure is saved at 150 dpi.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    t_mat = results['t_mat']
    N     = t_mat.shape[0]
    iu    = np.triu_indices(N, k=1)

    if vmax is None:
        vmax = float(np.percentile(np.abs(t_mat[iu]), 98))
    vmax  = max(vmax, 0.1)

    panels = [
        ('t-statistic\n(all edges)',                    t_mat),
        (f'Uncorrected\n(p < {alpha})',                 np.where(results['sig_unc'], t_mat, 0.0)),
        (f'FDR corrected\n(q < {alpha}, BH)',           np.where(results['sig_fdr'], t_mat, 0.0)),
        (f'FWE corrected\n(p < {alpha}, max-t perm)',   np.where(results['sig_fwe'], t_mat, 0.0)),
    ]

    tick_kw = dict(rotation=90, fontsize=6) if (labels and N <= 20) else {}
    show_ticks = labels is not None and N <= 20

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.patch.set_facecolor('white')

    for ax, (title, data) in zip(axes, panels):
        n_sig = int((np.abs(data[iu]) > 0).sum())
        label_str = '' if title.startswith('t-stat') else f'\n{n_sig:,} edges'
        im = ax.imshow(data, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(title + label_str, fontsize=10, fontweight='bold', pad=6)
        ax.set_xlabel('Node', fontsize=8)
        ax.set_ylabel('Node', fontsize=8)
        if show_ticks:
            ax.set_xticks(range(N))
            ax.set_xticklabels(labels, **tick_kw)
            ax.set_yticks(range(N))
            ax.set_yticklabels(labels, fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='t')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    return fig


def plot_null_distribution(results, alpha=0.05,
                           figsize=(7, 4), output_path=None):
    """
    Plot the permutation null distribution of max|t| with observed maximum.

    Useful for understanding FWE correction and checking that n_perm is
    sufficient (the null distribution should look smooth and bell-shaped).

    Parameters
    ----------
    results : dict
        Output of permutation_glm().
    alpha : float
    figsize : tuple
    output_path : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    null    = results['null_max_t']
    t_mat   = results['t_mat']
    N       = t_mat.shape[0]
    iu      = np.triu_indices(N, k=1)
    obs_max = float(np.abs(t_mat[iu]).max())
    crit    = float(np.percentile(null, 100 * (1 - alpha)))
    p_fwe_obs = float((null >= obs_max).sum() / len(null))

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(null, bins=50, color='#2D9B6F', edgecolor='white',
            alpha=0.85, label='Null max |t|')
    ax.axvline(crit, color='#E8AD2E', lw=1.8, ls='--',
               label=f'{100*(1-alpha):.0f}th pct null = {crit:.2f}')
    ax.axvline(obs_max, color='#C53030', lw=2.5,
               label=f'Observed max |t| = {obs_max:.2f}  (FWE p = {p_fwe_obs:.4f})')
    ax.set_xlabel('Max |t| across edges', fontsize=11)
    ax.set_ylabel('Permutations', fontsize=11)
    ax.set_title('Permutation Null Distribution — FWE max-t',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Saving results
# ══════════════════════════════════════════════════════════════════════════════

def save_edge_results(results, labels, output_path):
    """
    Save edge-level statistics to a CSV (upper triangle only).

    Columns
    -------
    roi_i, roi_j    Parcel names for each edge
    t               Observed t-statistic
    p_unc           Uncorrected two-tailed p-value
    p_fdr           BH FDR-adjusted p-value
    p_fwe           Max-t FWE-corrected p-value
    sig_unc         Boolean: p_unc < alpha
    sig_fdr         Boolean: p_fdr < alpha
    sig_fwe         Boolean: p_fwe < alpha

    Parameters
    ----------
    results : dict
        Output of permutation_glm().
    labels : list of str
        Parcel names.
    output_path : str or Path

    Returns
    -------
    df : pd.DataFrame

    Examples
    --------
    >>> df = save_edge_results(results, labels, 'group_edges.csv')
    >>> df[df.sig_fdr].sort_values('t', ascending=False).head(10)
    """
    N  = results['t_mat'].shape[0]
    iu = np.triu_indices(N, k=1)

    df = pd.DataFrame({
        'roi_i'   : [labels[i] for i in iu[0]],
        'roi_j'   : [labels[j] for j in iu[1]],
        't'       : results['t_mat'][iu],
        'p_unc'   : results['p_unc'][iu],
        'p_fdr'   : results['p_fdr'][iu],
        'p_fwe'   : results['p_fwe'][iu],
        'sig_unc' : results['sig_unc'][iu],
        'sig_fdr' : results['sig_fdr'][iu],
        'sig_fwe' : results['sig_fwe'][iu],
    })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} edges to: {output_path}")
    print(f"  Sig (unc) : {df.sig_unc.sum():>6,}")
    print(f"  Sig (FDR) : {df.sig_fdr.sum():>6,}")
    print(f"  Sig (FWE) : {df.sig_fwe.sum():>6,}")
    return df
