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
import optunahub
from scipy.stats import zscore
import matplotlib.pyplot as plt

from nsbutils.plotting import plot_surf

from neuromodes.eigen import EigenSolver
from neuromodes.stats import sigmoid_rescale, zscorew
from neuromodes.io import fetch_example_surf


plt.rcParams["figure.dpi"] = 300

DEMO_DIR = Path(__file__).parent.resolve()
DEFAULT_ALPHA = None
DEFAULT_R = 18.0
DEFAULT_GAMMA = 116.0
METRIC_CHOICES = ("edge_fc_corr", "node_fc_corr")
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


def _load_empirical_data(subj_id: str, visit: str, medmask: np.ndarray) -> np.ndarray:
    bold_data = []
    for session in ("1", "2"):
        for acquisition in ("LR", "RL"):
            bold_path = (
                DEMO_DIR
                / "data"
                / subj_id
                / visit
                / f"rfMRI_REST{session}_{acquisition}_Atlas_MSMAll_hp2000_clean_rclean_tclean_4k.L.func.gii"
            )
            bold = np.asarray(nib.load(str(bold_path)).agg_data(), dtype=np.float32)[:, medmask].T
            bold_data.append(zscore(bold, axis=1).astype(np.float32))
            if np.isnan(bold_data).any():
                raise ValueError(f"NaN values found in empirical BOLD data: {bold_path}")
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
            zscorew(hetero_map[medmask], mass=solver.mass),
            lower=0.0,
            upper=2.0,
            center=1.0,
            steepness=float(alpha),
        )

    solver.solve(hetero=hetero, n_modes=int(n_modes), seed=365)
    downsample_factor = int(dt_emp / dt_model)
    nt_model = int(nt_emp * downsample_factor) + int(tsteady)
    bold = np.empty((int(np.sum(medmask)), nt_emp, n_runs), dtype=np.float32)

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
        bold[:, :, i] = zscore(bold_i, axis=1).astype(np.float32)

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


def _load_desampler() -> type:
    """Load Optuna Hub's differential-evolution sampler."""

    module = optunahub.load_module("samplers/differential_evolution")
    return module.DESampler


def _create_sampler(
    *,
    search_space: Mapping[str, optuna.distributions.BaseDistribution],
    population_size: int,
    f: float,
    cr: float,
    seed: int,
) -> optuna.samplers.BaseSampler:
    """Instantiate DESampler with its documented DE controls."""

    DESampler = _load_desampler()
    try:
        return DESampler(
            search_space=dict(search_space),
            population_size=population_size,
            F=f,
            CR=cr,
            seed=seed,
        )
    except TypeError as exc:
        raise TypeError(
            "The installed Optuna Hub DESampler API does not accept the expected "
            "search_space, population_size, F, CR, and seed arguments. Check the "
            "installed sampler version and update _create_sampler accordingly."
        ) from exc


def _get_journal_storage(journal_path: Path) -> optuna.storages.JournalStorage:
    """Create a file-backed JournalStorage for one Optuna study."""

    journal_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from optuna.storages.journal import JournalFileBackend
    except ImportError:
        from optuna.storages import JournalFileStorage as JournalFileBackend
    return optuna.storages.JournalStorage(JournalFileBackend(str(journal_path)))


class ConvergenceCallback:
    """Stop after a fixed number of trials without a meaningful best-value improvement."""

    def __init__(self, patience_trials: int, tol: float) -> None:
        if patience_trials < 1:
            raise ValueError("patience_trials must be >= 1")
        if tol < 0:
            raise ValueError("conv_tol must be >= 0")
        self.patience_trials = int(patience_trials)
        self.tol = float(tol)
        self.best_value = np.inf
        self.last_improvement_trial: Optional[int] = None

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            return

        value = float(trial.value)
        if value < self.best_value - self.tol:
            self.best_value = value
            self.last_improvement_trial = trial.number
            return

        if self.last_improvement_trial is None:
            self.last_improvement_trial = trial.number
            return

        if trial.number - self.last_improvement_trial >= self.patience_trials:
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
    parser.add_argument("--n_runs", type=int, default=4)    # 4 runs to match no. of empirical timeseries
    parser.add_argument("--n_modes", type=int, default=500)
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        choices=METRIC_CHOICES,
        default=["edge_fc_corr", "node_fc_corr"],
    )
    parser.add_argument("--alpha", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--r", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--gamma", type=float, nargs=3, default=None, metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--noise_seed", type=int, default=365)

    parser.add_argument(
        "--n_trials",
        type=int,
        default=500,
        help="Maximum number of additional trials to run in this submission.",
    )
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel Optuna workers.")
    parser.add_argument(
        "--popsize",
        type=int,
        default=16,
        help="Population-size multiplier. Actual DE population is popsize times n_dim.",
    )
    parser.add_argument("--F", dest="f", type=float, default=0.8, help="DE mutation scaling factor.")
    parser.add_argument("--CR", dest="cr", type=float, default=0.7, help="DE crossover probability.")
    parser.add_argument("--de_seed", type=int, default=365, help="DE sampler random seed.")
    parser.add_argument(
        "--conv_patience",
        type=int,
        default=10,
        help="Patience in population-equivalent generations.",
    )
    parser.add_argument(
        "--conv_tol",
        type=float,
        default=1e-5,
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
    if args.n_runs < 1 or args.n_modes < 1 or args.n_trials < 1 or args.n_jobs < 1:
        raise ValueError("n_runs, n_modes, n_trials, and n_jobs must all be >= 1")
    if args.popsize < 4:
        raise ValueError("--popsize must be >= 4")
    if not 0.0 <= args.f <= 2.0:
        raise ValueError("--F must be in [0, 2]")
    if not 0.0 <= args.cr <= 1.0:
        raise ValueError("--CR must be in [0, 1]")

    free_params, fixed_params, defaults = _build_param_specs(args)
    if not free_params:
        raise ValueError("Provide at least one free parameter, e.g. --alpha MIN MAX STEP")

    n_dim = len(free_params)
    population_size = int(args.popsize) * n_dim
    patience_trials = int(args.conv_patience) * population_size

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
    subj_dir.mkdir(parents=True, exist_ok=True)
    study_name = f"waves-reproducibility_id-{run_id}"
    journal_path = subj_dir / "optuna.journal.log"
    critical_config_path = subj_dir / "critical_config.json"
    full_config_path = subj_dir / "full_config.json"

    critical_config = {
        "study_name": study_name,
        "subj_id": args.subj_id,
        "visit": args.visit,
        "metrics": list(args.metrics),
        "n_runs": int(args.n_runs),
        "n_modes": int(args.n_modes),
        "noise_seed": int(args.noise_seed),
        "defaults": defaults,
        "fixed_params": fixed_params,
        "free_parameters": list(free_params),
        "popsize": int(args.popsize),
        "population_size": population_size,
        "F": float(args.f),
        "CR": float(args.cr),
        "de_seed": int(args.de_seed),
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
        "n_trials": int(args.n_trials),
        "n_jobs": int(args.n_jobs),
        "journal_path": str(journal_path),
    }
    atomic_write_json(full_config_path, full_config)

    ext_input_cache_dir = DEMO_DIR / "_cache"
    ext_input_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CACHE_DIR"] = str(ext_input_cache_dir)

    surf, medmask = _setup_surface_and_masks(args.subj_id, args.visit)
    hetero_map = nib.load(
        str(
            DEMO_DIR
            / "data"
            / args.subj_id
            / args.visit
            / f"{args.subj_id}.L.MyelinMap_BC_MSMAll.4k_fs_LR.func.gii"
        )
    ).darrays[0].data

	# Plot heteromap and save
    hetero_map[~medmask] = np.nan
    hetero_fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plot_surf(surf, hetero_map, cmap="turbo", ax=ax)
    hetero_fig.savefig(subj_dir / f"myelinmap.png", dpi=300, bbox_inches="tight")
    
    print("Loading empirical data and calculating FC...")
    emp_bold = _load_empirical_data(args.subj_id, args.visit, medmask)
    emp_outputs = {"fc": calc_fc(emp_bold)}
    nt_emp, dt_emp, dt_model, tsteady = _fetch_empirical_constants()

    search_space = build_search_space(free_params)
    sampler = _create_sampler(
        search_space=search_space,
        population_size=population_size,
        f=float(args.f),
        cr=float(args.cr),
        seed=int(args.de_seed),
    )
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
                {"fc": calc_fc(np.hstack([bold[:, :, i] for i in range(args.n_runs)]))}, 
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
        patience_trials=patience_trials,
        tol=float(args.conv_tol),
    )

    print(
        f"Starting Optuna DE study {study_name!r}: {args.n_trials} maximum additional trials, "
        f"{args.n_jobs} workers, population size {population_size}."
    )
    # Compute number of unique parameter combinations
    n_unique = 1
    for spec in free_params.values():
        n_steps = int(round((spec.max - spec.min) / spec.step)) + 1
        n_unique *= n_steps

	# Limit the number of trials to the number of unique parameter combinations
    if n_unique < args.n_trials:
        print(
			f"Warning: The search space has only {n_unique} unique parameter combinations, "
			f"but --n_trials={args.n_trials}. Maximum trials will be limited to {n_unique}."
		)
        max_trials = n_unique
    else:
        max_trials = args.n_trials
        
    study.optimize(
        objective,
        n_trials=max_trials,
        n_jobs=int(args.n_jobs),
        callbacks=[callback],
        catch=(RuntimeError, ValueError, FloatingPointError),
    )

    if not study.best_trials:
        raise RuntimeError("No completed Optuna trials are available.")

    best_trial = study.best_trial
    print(f"Completed trials: {len(study.trials)}")
    print(f"Best trial: {best_trial.number}")
    print(f"Best objective: {best_trial.value:.8f}")
    print(f"Best score: {-best_trial.value:.8f}")
    print(f"Best parameters: {best_trial.params}")
    print(f"Results directory: {subj_dir}")
    print(f"Total optimization time: {(time.time() - t0) / 3600:.3f} hrs")

    fig_opt_history = optuna.visualization.plot_optimization_history(study)
    fig_opt_history.write_image(str(subj_dir / "optimization_history.png"))
    fig_param_contour = optuna.visualization.plot_contour(study, params=list(free_params))
    fig_param_contour.write_image(str(subj_dir / "parameter_contour.png"))
    fig_param_importance = optuna.visualization.plot_param_importances(study)
    fig_param_importance.write_image(str(subj_dir / "parameter_importance.png"))

if __name__ == "__main__":
    main()
