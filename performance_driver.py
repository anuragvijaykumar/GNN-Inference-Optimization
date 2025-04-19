#!/usr/bin/env python3
"""
Runs <model, dataset, optimisation> N times and logs everything
into a single Weights & Biases run with 100 points.
"""

import argparse, csv, datetime as dt, json, subprocess, sys, uuid, shutil
from pathlib import Path
import shlex
import wandb

# ────────────────────────────────────────────────────────────────────────────
# Use the `python` on PATH (i.e. your conda env) rather than sys.executable
PYTHON_EXEC   = shutil.which("python") or shutil.which("python3") or sys.executable

ROOT          = Path(__file__).parent
DRIVER_SCRIPT = ROOT / "main.py"
LOG_ROOT      = ROOT / "logs"
METRIC_JSON   = ROOT / "last_run_metrics.json"
CSV_FIELDS    = ["repeat", "epoch", "inference_ms", "gpu_mem_MB"]

def sh(cmd: str):
    print(">>", cmd, flush=True)
    if subprocess.call(cmd, shell=True):
        raise RuntimeError(f"command failed: {cmd}")

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_type",   required=True)
    p.add_argument("--dataset",      required=True)
    p.add_argument("--optimization", required=True)
    p.add_argument("--dataset_dir",  required=True)
    p.add_argument("--repeats",      type=int, default=1)
    p.add_argument("--profile",      action="store_true")
    p.add_argument("--no-profile",   dest="profile", action="store_false")
    p.set_defaults(profile=False)
    return p.parse_args()

def ensure_header(path: Path, fieldnames):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fp:
            csv.DictWriter(fp, fieldnames).writeheader()

def append_row(path: Path, row: dict, fieldnames):
    with path.open("a", newline="") as fp:
        csv.DictWriter(fp, fieldnames).writerow(row)

def main() -> None:
    A = parse_cli()
    cfg = f"{A.model_type}_{A.dataset}_{A.optimization}"
    out_dir = LOG_ROOT / cfg
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / "results.csv"
    ensure_header(results_csv, CSV_FIELDS)

    driver   = shlex.quote(str(DRIVER_SCRIPT))
    data_dir = shlex.quote(A.dataset_dir)

    # one W&B run for _all_ repeats
    wb = wandb.init(
        project  = "GNN-Inference-Optimization",
        entity   = "ysu_cis",
        name     = f"{cfg}_{A.repeats}runs",
        config   = vars(A),
        reinit   = False
    )

    for r in range(1, A.repeats + 1):
        cmd = (
            f"{PYTHON_EXEC} {driver} "
            f"--model_type {A.model_type} "
            f"--dataset_name {A.dataset} "
            f"--dataset_dir {data_dir} "
            f"--optimization {A.optimization}"
        )
        if A.profile:
            raise NotImplementedError("Profiling is disabled in this driver")
        else:
            sh(cmd)

        with METRIC_JSON.open() as fp:
            m = json.load(fp)

        row = {
            "repeat":       r,
            "epoch":        m["epoch"],
            "inference_ms": m["inference_ms"],
            "gpu_mem_MB":   m["gpu_mem_MB"],
        }

        append_row(results_csv, row, CSV_FIELDS)
        wb.log(row)

    wb.finish()
    print("Completed →", results_csv)

if __name__ == "__main__":
    main()
