"""Optimize resting-state model parameters with Optuna differential evolution.

Each non-test run is identified by --id and stored in results/waves/id-{id}.
The Optuna journal is the complete trial record for that study. Re-running the
same ID appends trials, provided configuration-critical arguments are unchanged.
"""

from __future__ import annotations

import os
import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import optuna
from scipy.stats import zscore, ks_2samp
from scipy.signal import butter, filtfilt, hilbert
import matplotlib.pyplot as plt

from nsbutils.plotting import plot_surf

from neuromodes.eigen import EigenSolver
from neuromodes.stats import sigmoid_rescale, zscorew
from neuromodes.io import fetch_example_surf
from neuromodes.mesh import unmask_data


plt.rcParams["figure.dpi"] = 300

DEMO_DIR = Path(__file__).parent.parent.resolve()
DEFAULT_ALPHA = None
DEFAULT_R = 18.0
DEFAULT_GAMMA = 116.0
METRIC_CHOICES = ("edge_fc_corr", "node_fc_corr", "fcd_ks")
SESSION_CHOICES = (1, 2)
PARAM_ORDER = ("alpha", "r", "gamma")


@dataclass(frozen=True)
class GridSpec:
    """A bounded, regularly spaced parameter domain."""

    min: float
    max: float
    step: float

# TODO: verify this is function works as expected
def params_equal(p1: Dict[str, Any], p2: Dict[str, Any], tol: float = 1e-9) -> bool:
    if set(p1.keys()) != set(p2.keys()):
        return False
    for k in p1.keys():
        v1, v2 = p1[k], p2[k]
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            if abs(float(v1) - float(v2)) > tol:
                return False
        elif v1 != v2:
            return False
    return True

def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write JSON to path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.stem}.{Path.cwd().stat().st_ino}.tmp{path.suffix}")
    tmp_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def parse_grid3(values: Tuple[float, float, float], name: str) -> GridSpec:
    """Parse a (minimum, maximum, step) CLI triplet."""

    min_val, max_val, step = (float(value) for value in values)
    if max_val < min_val:
        min_val, max_val = max_val, min_val
    if step <= 0:
        raise ValueError(f"{name} step must be > 0")

    n_steps = (max_val - min_val) / step
    if not np.isclose(n_steps, round(n_steps), rtol=0.0, atol=1e-10):
        raise ValueError(
            f"{name}: (max - min) must be an integer multiple of step. "
            f"Received min={min_val}, max={max_val}, step={step}."
        )
    return GridSpec(min=min_val, max=max_val, step=step)


def next_run_id(results_dir: Path, *, prefix: str = "id-") -> int:
    """Return the next positive run ID in results_dir."""

    run_ids = []
    if results_dir.exists():
        for child in results_dir.iterdir():
            if child.is_dir() and child.name.startswith(prefix):
                suffix = child.name[len(prefix):]
                if suffix.isdigit() and int(suffix) > 0:
                    run_ids.append(int(suffix))
    return max(run_ids, default=0) + 1


def collect_config_mismatches(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    """Return human-readable differences between nested JSON-compatible values."""

    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        mismatches: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected:
                mismatches.append(f"{child_prefix}: unexpected current value")
            elif key not in actual:
                mismatches.append(f"{child_prefix}: missing current value")
            else:
                mismatches.extend(collect_config_mismatches(expected[key], actual[key], child_prefix))
        return mismatches

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{prefix}: expected list length {len(expected)}, got {len(actual)}"]
        mismatches = []
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            mismatches.extend(
                collect_config_mismatches(expected_item, actual_item, f"{prefix}[{index}]")
            )
        return mismatches

    return [] if expected == actual else [f"{prefix}: expected {expected!r}, got {actual!r}"]


def _build_param_specs(
    args: argparse.Namespace,
) -> Tuple[Dict[str, GridSpec], Dict[str, Any], Dict[str, Any]]:
    defaults = {
        "alpha": DEFAULT_ALPHA,
        "r": DEFAULT_R,
        "gamma": DEFAULT_GAMMA,
    }

    free_params: Dict[str, GridSpec] = {}
    for name in PARAM_ORDER:
        values = getattr(args, name)
        if values is not None:
            free_params[name] = parse_grid3(tuple(values), name)

    fixed_params = {
        name: defaults[name]
        for name in PARAM_ORDER
        if name not in free_params and defaults[name] is not None
    }
    return free_params, fixed_params, defaults


def _fetch_empirical_constants() -> Tuple[int, float, float, int]:
    """Return constants for the empirical BOLD data (nt, dt, dt_model, tsteady)."""
    return 1200, 0.72, 0.09, 550


def _setup_surface_and_masks(subj_id: str, visit: str) -> Tuple[str, np.ndarray]:
    surf = str(
        DEMO_DIR
        / "data"
        / subj_id
        / visit
        / f"{subj_id}.L.midthickness_MSMAll.4k_fs_LR.surf.gii"
    )
    _, medmask = fetch_example_surf(density="4k")

    return surf, medmask


def _load_empirical_data(subj_id: str, visit: str, sessions: list, medmask: np.ndarray) -> np.ndarray:
    bold_data = []
    for session in sessions:
        for acquisition in ("LR", "RL"):
            bold_path = (
                DEMO_DIR
                / "data"
                / subj_id
                / visit
                / f"rfMRI_REST{session}_{acquisition}_Atlas_MSMAll_hp2000_clean_rclean_tclean_4k.L.func.gii"
            )
            bold = np.asarray(nib.load(str(bold_path)).agg_data(), dtype=np.float32)[:, medmask].T
            # Perform GSR
            # bold_gsr = bold - np.mean(bold, axis=0, keepdims=True)
            # z-score
            bold_z = zscore(bold, axis=1).astype(np.float32)
            if np.isnan(bold_z).any():
                raise ValueError(f"NaN values found in empirical BOLD data: {bold_path}")
            if np.shape(bold_z) != (np.sum(medmask), 1200):
                raise ValueError(
                    f"Empirical BOLD data shape mismatch: {bold_path} has shape {bold_z.shape}, "
                    f"expected ({np.sum(medmask)}, 1200)"
                )
            bold_data.append(bold_z)
    return np.concatenate(bold_data, axis=1).astype(np.float32)


def _simulate_bold(
    *,
    surf: str,
    medmask: np.ndarray,
    hetero_map: Optional[np.ndarray],
    alpha: Optional[float],
    r: Optional[float],
    gamma: Optional[float],
    noise_seed: int,
    n_modes: int,
    n_runs: int,
    nt_emp: int,
    dt_emp: float,
    dt_model: float,
    tsteady: int,
) -> np.ndarray:
    solver = EigenSolver(geometry=surf, mask=medmask)

    if hetero_map is None:
        hetero = None
    else:
        if alpha is None:
            raise ValueError("alpha must be numeric when a heterogeneity map is supplied")
        hetero = sigmoid_rescale(
            zscorew(hetero_map, mass=solver.mass),
            lower=0.0,
            upper=2.0,
            steepness=float(alpha),
        )

    solver.solve(hetero=hetero, n_modes=int(n_modes), seed=365)
    downsample_factor = int(dt_emp / dt_model)
    nt_model = int(nt_emp * downsample_factor) + int(tsteady)
    bold = np.empty((int(np.sum(medmask)), nt_emp*n_runs), dtype=np.float32)

    for i in range(n_runs):
        sim_kwargs: Dict[str, Any] = {
            "dt": dt_model,
            "nt": nt_model,
            "seed": int(noise_seed) + i,
            "cache_input": True,
            "method": "fourier",
        }
        if r is not None:
            sim_kwargs["r"] = float(r)
        if gamma is not None:
            sim_kwargs["gamma"] = float(gamma)

        bold_i = solver.balloon_model(solver.sim_nft_waves(**sim_kwargs), dt=dt_model).astype(np.float32)
        bold_i = bold_i[:, tsteady:][:, ::downsample_factor]
        bold[:, i*nt_emp:(i+1)*nt_emp] = zscore(bold_i, axis=1).astype(np.float32)

    return bold


def calc_fc(bold: np.ndarray) -> np.ndarray:
    """Compute the functional-connectivity matrix from BOLD time series."""

    return np.corrcoef(zscore(bold, axis=1), dtype=np.float32)


def calc_edge_fc(fc: np.ndarray, eps: float = 1e-7, fisher_z: bool = True) -> np.ndarray:
    triu_i, triu_j = np.triu_indices(fc.shape[0], k=1)
    edge_fc = np.clip(fc[triu_i, triu_j], -1 + eps, 1 - eps)
    return np.arctanh(edge_fc) if fisher_z else edge_fc


def calc_node_fc(fc: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    fc_clipped = np.clip(fc, -1 + eps, 1 - eps)
    np.fill_diagonal(fc_clipped, np.nan)
    return np.nanmean(np.arctanh(fc_clipped), axis=1)


def filter_bold(bold, fnq, band_freq=(0.01, 0.1), k=2):
    """
    Apply Butterworth bandpass filter to BOLD signal.
    
    Parameters
    ----------
    bold : np.ndarray
        BOLD time series to filter.
    fnq : float
        Nyquist frequency in Hz.
    band_freq : tuple, default=(0.01, 0.1)
        Frequency band (low, high) in Hz.
    k : int, default=2
        Filter order.
    
    Returns
    -------
    bold_filtered : np.ndarray
        Bandpass filtered BOLD signal.
    """
    # Normalize frequency band to Nyquist frequency
    Wn = [band_freq[0] / fnq, band_freq[1] / fnq]
    b, a = butter(k, Wn, btype="bandpass")

    # Z-score and apply zero-phase filter
    bold_z = zscore(bold, axis=1).astype(np.float32)
    bold_filtered = filtfilt(b, a, bold_z, axis=1)

    return bold_filtered


def calc_fcd_efficient3(bold, fnq, band_freq=(0.04, 0.07), win_len=10, win_step=2,
                        time_chunk_size=100, edge_chunk_size=100_000, metric="phase",
                        verbose=False):
    """
    Calculate FCD using block-windowed synchrony with memory-efficient chunking.

    This implementation replaces the per-timepoint sliding average (n_avg) used
    in calc_fcd_efficient2 with discrete, overlapping windows of length win_len
    and step win_step. Synchrony is averaged within each window rather than
    smoothed at every timepoint, so the number of output windows is
    approximately (nt_trunc - win_len) // win_step + 1 instead of nt_trunc.
    This brings the method in line with the standard windowed-FCD construction
    (cf. get_fcd, get_fcd2) while retaining the phase/amplitude synchrony
    metric and chunked edge/time processing from calc_fcd_efficient2.

    Both time and edges are processed in chunks and row normalization is
    deferred until after accumulation, so the full (n_windows, n_edges)
    synchrony matrix is never held in memory at once.

    Parameters
    ----------
    bold : np.ndarray
        BOLD time series with shape (n_regions, n_timepoints).
    fnq : float
        Nyquist frequency in Hz.
    band_freq : tuple, default=(0.04, 0.07)
        Frequency band (low, high) in Hz for bandpass filtering.
    win_len : int, default=10
        Window length in samples over which synchrony is averaged.
    win_step : int, default=5
        Step size in samples between consecutive window starts. win_step
        < win_len gives overlapping windows; win_step == win_len gives
        non-overlapping windows.
    time_chunk_size : int, default=50
        Number of raw timepoints to process per chunk when computing
        instantaneous synchrony prior to windowing.
    edge_chunk_size : int, default=100_000
        Number of edges to process per chunk. Controls peak memory
        (n_windows * edge_chunk_size * 4 bytes for float32).
    metric : str, default="phase"
        Metric to use: "phase" or "amplitude".
    verbose : bool, default=False
        Print timing information.

    Returns
    -------
    fcd_upper : np.ndarray
        Upper triangular FCD matrix values.
    """
    t1 = time.time()
    if win_len < 1:
        raise ValueError("win_len must be greater than 0")
    if win_step < 1:
        raise ValueError("win_step must be greater than 0")
    if win_step > win_len:
        raise ValueError("win_step should not exceed win_len (this would skip timepoints)")

    bold = bold.astype(np.float32, copy=False)
    n_regions, nt = bold.shape

    bold_filtered = filter_bold(bold, fnq, band_freq=band_freq, k=2)
    analytic = hilbert(bold_filtered, axis=1)
    if metric == "amplitude":
        signal = np.abs(analytic)
    elif metric == "phase":
        signal = np.angle(analytic)
    else:
        raise ValueError("Invalid metric. Choose 'amplitude' or 'phase'.")

    signal_trunc = np.ascontiguousarray(signal[:, 9:nt - 9], dtype=np.float32)
    nt_trunc = signal_trunc.shape[1]

    if nt_trunc < win_len:
        raise ValueError(
            f"Truncated time series ({nt_trunc} samples) is shorter than win_len ({win_len})."
        )

    triu_i, triu_j = np.triu_indices(n_regions, k=1)
    n_edges = len(triu_i)

    # Window start indices define the windowed timeline. Each window
    # [w, w + win_len) is averaged into one synchrony vector.
    window_starts = np.arange(0, nt_trunc - win_len + 1, win_step)
    n_windows = len(window_starts)

    # Accumulators only, never the full (n_windows, n_edges) matrix
    fcd_mat = np.zeros((n_windows, n_windows), dtype=np.float32)
    norms_sq = np.zeros(n_windows, dtype=np.float32)

    for e_start in range(0, n_edges, edge_chunk_size):
        e_end = min(e_start + edge_chunk_size, n_edges)
        ei = triu_i[e_start:e_end]
        ej = triu_j[e_start:e_end]
        n_edges_chunk = e_end - e_start

        # Raw instantaneous synchrony for this edge chunk, all timepoints
        synchrony_full = np.empty((nt_trunc, n_edges_chunk), dtype=np.float32)

        for t_start in range(0, nt_trunc, time_chunk_size):
            t_end = min(t_start + time_chunk_size, nt_trunc)
            n_time_chunk = t_end - t_start

            for i in range(n_time_chunk):
                signal_t = signal_trunc[:, t_start + i]
                diff = signal_t[ei] - signal_t[ej]
                synchrony_full[t_start + i, :] = np.cos(diff, out=diff)

        # Average instantaneous synchrony within each block window
        phase_chunk = np.empty((n_windows, n_edges_chunk), dtype=np.float32)
        for w_idx, w_start in enumerate(window_starts):
            phase_chunk[w_idx, :] = synchrony_full[w_start:w_start + win_len, :].mean(axis=0)

        # Accumulate this edge chunk's contribution to the Gram matrix
        # and to each row's squared norm, deferring normalization until
        # all edge chunks have been processed.
        norms_sq += np.sum(phase_chunk ** 2, axis=1)
        fcd_mat += phase_chunk @ phase_chunk.T

    # (a / ||a||) . (b / ||b||) = (a . b) / (||a|| ||b||), applied after
    # accumulating the dot products across all edge chunks.
    norms = np.sqrt(norms_sq)
    norms[norms < 1e-6] = 1.0
    fcd_mat /= np.outer(norms, norms)

    triu_ind = np.triu_indices(fcd_mat.shape[0], k=1)

    t2 = time.time()
    if verbose:
        print(f"FCD calculation completed in {(t2 - t1) / 60:.4f} minutes. "
              f"n_windows={n_windows}")
    return fcd_mat[triu_ind]


def evaluate_model(
    model_outputs: Mapping[str, np.ndarray],
    emp_outputs: Mapping[str, np.ndarray],
    metrics: Sequence[str],
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    if "edge_fc_corr" in metrics:
        model_edge_fc = calc_edge_fc(model_outputs["fc"], fisher_z=True)
        emp_edge_fc = calc_edge_fc(emp_outputs["fc"], fisher_z=True)
        results["edge_fc_corr"] = float(np.corrcoef(model_edge_fc, emp_edge_fc)[0, 1])
    if "node_fc_corr" in metrics:
        model_node_fc = calc_node_fc(model_outputs["fc"])
        emp_node_fc = calc_node_fc(emp_outputs["fc"])
        results["node_fc_corr"] = float(np.corrcoef(model_node_fc, emp_node_fc)[0, 1])
    if "fcd_ks" in metrics:
        results['fcd_ks'] = 1 - ks_2samp(
            model_outputs['fcd'].flatten(), 
            emp_outputs['fcd'].flatten()
        )[0]
    return results


def build_search_space(free_params: Mapping[str, GridSpec]) -> Dict[str, optuna.distributions.FloatDistribution]:
    """Build the complete, explicit discrete search space for DESampler."""

    return {
        name: optuna.distributions.FloatDistribution(
            low=spec.min,
            high=spec.max,
            step=spec.step,
        )
        for name, spec in free_params.items()
    }


def _get_journal_storage(journal_path: Path) -> optuna.storages.JournalStorage:
    """Create a file-backed JournalStorage for one Optuna study."""

    journal_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from optuna.storages.journal import JournalFileBackend
    except ImportError:
        from optuna.storages import JournalFileStorage as JournalFileBackend
    return optuna.storages.JournalStorage(JournalFileBackend(str(journal_path)))


class ConvergenceCallback:
    """Stop after a fixed number of completed trials without meaningful improvement."""

    def __init__(
        self,
        patience_trials: int = 80,
        tol: float = 1e-5,
        min_trials: int = 100,
    ) -> None:
        if patience_trials < 1:
            raise ValueError("patience_trials must be >= 1")
        if tol < 0:
            raise ValueError("tol must be >= 0")
        if min_trials < 0:
            raise ValueError("min_trials must be >= 0")

        self.patience_trials = int(patience_trials)
        self.tol = float(tol)
        self.min_trials = int(min_trials)
        self.best_value: Optional[float] = None
        self.last_improvement_completed: Optional[int] = None

    def __call__(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> None:
        if (
            trial.state != optuna.trial.TrialState.COMPLETE
            or trial.value is None
            or not np.isfinite(trial.value)
        ):
            return

        completed_trials = [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None
            and np.isfinite(t.value)
        ]
        n_completed = len(completed_trials)
        current_best = float(study.best_value)

        if self.best_value is None:
            self.best_value = current_best
            self.last_improvement_completed = n_completed
            return

        if current_best < self.best_value - self.tol:
            self.best_value = current_best
            self.last_improvement_completed = n_completed
            return

        if (
            n_completed >= self.min_trials
            and self.last_improvement_completed is not None
            and n_completed - self.last_improvement_completed
            >= self.patience_trials
        ):
            print(
                "Stopping after no best-objective improvement of at least "
                f"{self.tol:g} for {self.patience_trials} completed trials."
            )
            study.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize resting-state model parameters with Optuna differential evolution."
    )
    parser.add_argument(
        "--id",
        type=int,
        default=None,
        help=(
            "Optional run ID for intentional continuation. If omitted, the next available "
            "positive ID is used. ID 0 is a scratch/test slot and is overwritten on rerun."
        ),
    )
    parser.add_argument("--subj_id", type=str, required=True)
    parser.add_argument("--visit", type=str, required=True)
    parser.add_argument(
        "--session", 
        type=int, 
        nargs="+", 
        choices=SESSION_CHOICES, 
        default=[1, 2],
        help=(
            "Session numbers to include in the analysis. E.g., '--session 1 2' will include "
            "both sessions 1 and 2 but '--session 1' will only include the first session. "
            "Default is both sessions."
        )
    )
    parser.add_argument("--n_runs", type=int, default=4)    # 4 runs to match no. of empirical timeseries
    parser.add_argument("--n_modes", type=int, default=500)
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        choices=METRIC_CHOICES,
        default=["edge_fc_corr", "node_fc_corr", "fcd_ks"],
    )
    parser.add_argument("--fcd_band_freq", type=float, nargs=2, default=(0.04, 0.07), metavar=("LOW", "HIGH"))
    parser.add_argument("--fcd_win_len", type=int, default=10, help="FCD window length in samples.")
    parser.add_argument("--fcd_win_step", type=int, default=2, help="FCD window step in samples.")
    parser.add_argument("--alpha", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--r", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--gamma", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--noise_seed", type=int, default=365)

    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel Optuna workers.")
    parser.add_argument(
        "--min_trials",
        type=int,
        default=150,
        help="Minimum number of completed trials before convergence callback can stop the study.",
    )
    parser.add_argument(
        "--max_trials",
        type=int,
        default=500,
        help="Maximum number of additional trials to run in this submission.",
    )
    parser.add_argument("--opt_seed", type=int, default=365, help="Optuna sampler random seed.")
    parser.add_argument(
        "--conv_patience",
        type=int,
        default=50,
        help="Patience in completed trials for convergence callback (stop if no improvement for this many trials).",
    )
    parser.add_argument(
        "--conv_tol",
        type=float,
        default=1e-4,
        help="Minimum absolute best-objective improvement that resets convergence patience.",
    )

    args = parser.parse_args()
    args.metrics = list(dict.fromkeys(args.metrics))
    return args


def main() -> None:
    t0 = time.time()
    args = parse_args()

    if args.id is not None and args.id < 0:
        raise ValueError("--id must be >= 0")
    if not args.metrics:
        raise ValueError("At least one metric must be supplied via --metrics")
    if args.n_runs < 1 or args.n_modes < 1 or args.max_trials < 1 or args.n_jobs < 1:
        raise ValueError("n_runs, n_modes, max_trials, and n_jobs must all be >= 1")

    free_params, fixed_params, defaults = _build_param_specs(args)
    if not free_params:
        raise ValueError("Provide at least one free parameter, e.g. --alpha MIN MAX STEP")

    results_dir = DEMO_DIR / "results" / "waves"
    if args.id == 0:
        run_id = 0
        run_dir = results_dir / "id-0"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
    elif args.id is None:
        run_id = next_run_id(results_dir)
        run_dir = results_dir / f"id-{run_id}"
        while True:
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                run_id = next_run_id(results_dir)
                run_dir = results_dir / f"id-{run_id}"
    else:
        run_id = int(args.id)
        run_dir = results_dir / f"id-{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

    subj_dir = run_dir / args.subj_id / args.visit
    if len(args.session) == 1:
        subj_dir = subj_dir / f"ses-{args.session[0]}"
    subj_dir.mkdir(parents=True, exist_ok=True)
    study_name = f"waves-reproducibility_id-{run_id}"
    journal_path = subj_dir / "optuna.journal.log"
    critical_config_path = subj_dir / "critical_config.json"
    full_config_path = subj_dir / "full_config.json"

    critical_config = {
        "study_name": study_name,
        "subj_id": args.subj_id,
        "visit": args.visit,
        "session": list(args.session),
        "metrics": list(args.metrics),
        "n_runs": int(args.n_runs),
        "n_modes": int(args.n_modes),
        "fcd_win_len": int(args.fcd_win_len),
        "fcd_win_step": int(args.fcd_win_step),
        "fcd_band_freq": list(args.fcd_band_freq),
        "noise_seed": int(args.noise_seed),
        "defaults": defaults,
        "fixed_params": fixed_params,
        "free_parameters": list(free_params),
        "opt_seed": int(args.opt_seed),
        "conv_patience": int(args.conv_patience),
        "conv_tol": float(args.conv_tol),
    }

    if critical_config_path.exists() and run_id != 0:
        stored_config = json.loads(critical_config_path.read_text(encoding="utf-8"))
        mismatches = collect_config_mismatches(stored_config, critical_config)
        if mismatches:
            raise ValueError(
                f"Run ID {run_id} has configuration mismatches against {critical_config_path}:\n - "
                + "\n - ".join(mismatches)
            )
    else:
        atomic_write_json(critical_config_path, critical_config)

    full_config = {
        **critical_config,
        "optimization_parameters": {name: asdict(spec) for name, spec in free_params.items()},
        "min_trials": int(args.min_trials),
        "max_trials": int(args.max_trials),
        "n_jobs": int(args.n_jobs),
        "journal_path": str(journal_path),
    }
    atomic_write_json(full_config_path, full_config)

    ext_input_cache_dir = DEMO_DIR / "_cache"
    ext_input_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CACHE_DIR"] = str(ext_input_cache_dir)

    surf, medmask = _setup_surface_and_masks(args.subj_id, args.visit)
    hetero_map = np.asarray(
        nib.load(
            str(
                DEMO_DIR
                / "data"
                / args.subj_id
                / args.visit
                / f"{args.subj_id}.L.MyelinMap_BC_MSMAll.4k_fs_LR.func.gii"
            )
        ).darrays[0].data
    )[medmask]

	# Plot heteromap and save
    hetero_fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plot_surf(surf, unmask_data(hetero_map, medmask), cmap="turbo", ax=ax, cbar=True)
    hetero_fig.savefig(subj_dir / f"myelinmap.png", dpi=200, bbox_inches="tight")

    # Load empirical BOLD data and calculate FC
    print("Loading empirical data and calculating outputs...")
    nt_emp, dt_emp, dt_model, tsteady = _fetch_empirical_constants()
    emp_bold = _load_empirical_data(args.subj_id, args.visit, args.session, medmask)

    # Plot and save empirical BOLD time series (carpet plot)
    fig_emp_bold, ax = plt.subplots(1, 1, figsize=(6, 4))
    emp_bold_min = -np.max(np.abs(emp_bold))
    emp_bold_max = np.max(np.abs(emp_bold))
    im = ax.imshow(emp_bold, aspect="auto", cmap="seismic", vmin=emp_bold_min, vmax=emp_bold_max)
    ax.set_xlabel("Time (TRs)")
    ax.set_ylabel("Vertices")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_emp_bold.savefig(subj_dir / f"empirical_bold.png", dpi=200, bbox_inches="tight")

    emp_outputs = {}
    if "edge_fc_corr" in args.metrics or "node_fc_corr" in args.metrics:
        emp_outputs["fc"] = calc_fc(emp_bold)
    if "fcd_ks" in args.metrics:
        emp_outputs["fcd"] = calc_fcd_efficient3(
            emp_bold, 
            fnq=1/(2*dt_emp), 
            band_freq=args.fcd_band_freq, 
            win_len=args.fcd_win_len, 
            win_step=args.fcd_win_step
        )

    # Plot and save empirical FC matrix
    fig_emp_fc, ax = plt.subplots(1, 1, figsize=(6, 4))
    im = ax.imshow(emp_outputs["fc"], cmap="seismic", vmin=-1, vmax=1)
    ax.set_xlabel("Vertices")
    ax.set_ylabel("Vertices")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_emp_fc.savefig(subj_dir / f"empirical_fc.png", dpi=200, bbox_inches="tight")

    sampler = optuna.samplers.TPESampler(args.opt_seed)
    storage = _get_journal_storage(journal_path)

    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            sampler=sampler,
            load_if_exists=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not create or load study {study_name!r}") from exc

    stored_attrs = study.user_attrs.get("critical_config")
    if stored_attrs is None:
        study.set_user_attr("critical_config", critical_config)
    else:
        mismatches = collect_config_mismatches(stored_attrs, critical_config)
        if mismatches:
            raise ValueError(
                f"Optuna study {study_name!r} has configuration mismatches:\n - "
                + "\n - ".join(mismatches)
            )

    def objective(trial: optuna.trial.Trial) -> float:
        params: Dict[str, Any] = {**defaults, **fixed_params}
        for name, spec in free_params.items():
            params[name] = trial.suggest_float(name, spec.min, spec.max, step=spec.step)

        # TODO: rethink this since it still counts as a trial and will be included in the convergence callback
		# Check for an existing completed trial with the same parameters
        for t in trial.study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE and params_equal(t.params, params):
                print(f"Skipping duplicate trial {t.number} with identical parameters.")
                return float(t.value)

        try:
            bold = _simulate_bold(
                surf=surf,
                medmask=medmask,
                hetero_map=hetero_map,
                alpha=params["alpha"],
                r=params["r"],
                gamma=params["gamma"],
                noise_seed=int(args.noise_seed),
                n_modes=int(args.n_modes),
                n_runs=int(args.n_runs),
                nt_emp=nt_emp,
                dt_emp=dt_emp,
                dt_model=dt_model,
                tsteady=tsteady,
            )
            metrics = evaluate_model(
                {
                    "fc": calc_fc(bold), 
                    "fcd": calc_fcd_efficient3(
                        bold, 
                        fnq=1/(2*dt_emp), 
                        band_freq=args.fcd_band_freq, 
                        win_len=args.fcd_win_len, 
                        win_step=args.fcd_win_step
                    )
                }, 
                emp_outputs, 
                args.metrics
			)
            score = float(sum(metrics[name] for name in args.metrics))
            if not np.isfinite(score):
                raise ValueError(f"Non-finite score: {score}")

            for name, value in metrics.items():
                trial.set_user_attr(name, value)
            trial.set_user_attr("score", score)
            return -score
        except Exception as exc:
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            raise

    callback = ConvergenceCallback(
        patience_trials=args.conv_patience + args.n_jobs - 1,  # account for parallel workers
        tol=float(args.conv_tol),
        min_trials=args.min_trials
    )

    print(
        f"Starting Optuna study {study_name!r} with {args.n_jobs} workers"
    )
    # Compute number of unique parameter combinations
    n_unique = 1
    for spec in free_params.values():
        n_steps = int(round((spec.max - spec.min) / spec.step)) + 1
        n_unique *= n_steps

	# Limit the number of trials to the number of unique parameter combinations
    if n_unique < args.max_trials:
        print(
			f"Warning: The search space has only {n_unique} unique parameter combinations, "
			f"but --max_trials={args.max_trials}. Maximum trials will be limited to {n_unique}."
		)
        max_trials = n_unique
    else:
        max_trials = args.max_trials
        
    study.optimize(
        objective,
        n_trials=max_trials,
        n_jobs=int(args.n_jobs),
        callbacks=[callback],
        catch=(RuntimeError, ValueError, FloatingPointError),
    )

    if not study.best_trials:
        raise RuntimeError("No completed Optuna trials are available.")

    # Plot and save Optuna visualizations
    fig_opt_history = optuna.visualization.plot_optimization_history(study)
    fig_opt_history.write_image(str(subj_dir / "optimization_history.png"))
    fig_param_contour = optuna.visualization.plot_contour(study, params=list(free_params))
    fig_param_contour.write_image(str(subj_dir / "parameter_contour.png"))
    fig_param_importance = optuna.visualization.plot_param_importances(study)
    fig_param_importance.write_image(str(subj_dir / "parameter_importance.png"))
    for metric in args.metrics:
        fig = optuna.visualization.plot_contour(
            study, params=list(free_params),
            target=lambda t: t.user_attrs[metric],
            target_name=metric,
        )
        fig.write_image(str(subj_dir / f"parameter_contour_{metric}.png"))

    # Print optuna summary and best trial information
    t1 = time.time()
    best_trial = study.best_trial
    print(f"Completed trials: {len(study.trials)}")
    print(f"Best trial: {best_trial.number}")
    print(f"Best objective: {best_trial.value:.8f}")
    print(f"Best score: {-best_trial.value:.8f}")
    print(f"Best parameters: {best_trial.params}")
    print(f"Results directory: {subj_dir}")
    print(f"Total optimization time with {args.n_jobs} CPUs: {(t1 - t0) / 3600:.3f} hrs")

    # Save optuna summary and best trial information
    summary_path = subj_dir / "optuna_summary.json"
    summary = {
        "study_name": study_name,
        "n_trials": len(study.trials),
        "best_trial_number": best_trial.number,
        "best_objective": float(best_trial.value),
        "best_score": float(-best_trial.value),
        "best_params": best_trial.params,
        "metrics": {metric: best_trial.user_attrs.get(metric) for metric in args.metrics},
        "total_time_hrs": np.round((t1 - t0) / 3600, 3),
    }
    atomic_write_json(summary_path, summary)

if __name__ == "__main__":
    main()
