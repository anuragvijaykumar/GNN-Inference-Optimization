#!/usr/bin/env python3
"""
Single-run inference driver:
 - loads <model, dataset>
 - executes one forward pass
 - measures wall-clock + peak GPU memory
 - writes last_run_metrics.json
"""
from __future__ import annotations
import argparse, contextlib, json, time
from pathlib import Path
import torch, torch.nn.functional as F

from dataset import load_dataset
import models

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_type",   required=True, choices=["GCN", "GIN", "GAT"])
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--dataset_dir",  required=True)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--epochs",       type=int, default=1)
    p.add_argument("--hidden_dim",   type=int, default=64)
    p.add_argument("--optimization",
                   choices=["baseline", "compile", "amp", "compile_amp"],
                   default="baseline")
    p.add_argument("--sparsity_type", default="none",
                   help="none | irregular | ...  (ignored for baseline)")
    p.add_argument("--sparsity_rate", type=float, default=0.0,
                   help="0.0‑1.0 pruning ratio (ignored for baseline)")
    p.add_argument("--kernel_type", default="cusparse",
                   choices=["cusparse", "pruneSp"])
    return p.parse_args()

def build(kind: str, in_f: int, hid_f: int, out_f: int, args) -> torch.nn.Module:
    kind = kind.upper()
    if   kind == "GCN": net = models.GCN(args, in_f, hid_f, out_f)
    elif kind == "GIN": net = models.GIN(args, in_f, hid_f, out_f)
    elif kind == "GAT": net = models.GAT(args, in_f, hid_f, out_f)
    else: raise ValueError(f"unknown model_type: {kind}")
    return net.to(args.device)

def main() -> None:
    a      = parse_args()
    device = torch.device(a.device)

    use_compile = "compile" in a.optimization
    use_amp     = "amp"     in a.optimization

    X, A_pack, y = load_dataset(a.dataset_name, a.dataset_dir)
    n_feat  = X.shape[1]
    n_class = int(y.max()) + 1

    A_csr, A_dgl, row_ptr, col_idx, deg = A_pack
    X, y   = X.to(device), y.to(device)
    A_pack = (A_csr.to(device), A_dgl, row_ptr.to(device),
              col_idx.to(device), deg.to(device))

    model = build(a.model_type, n_feat, a.hidden_dim, n_class, a).eval()
    model.requires_grad_(False)
    if use_compile:
        model = torch.compile(model, mode="default", fullgraph=False)

    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, a.epochs + 1):
        ctx = torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext()
        t0  = time.time()
        with ctx:
            logits = model(X, A_pack, a)
            _ = F.nll_loss(F.log_softmax(logits, dim=-1), y)
        torch.cuda.synchronize(device)
        elapsed_ms = (time.time() - t0) * 1_000
        peak_mb    = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"[{epoch}/{a.epochs}] {elapsed_ms:.2f} ms  peak GPU {peak_mb:.1f} MB")

    metrics = {
        "epoch":        epoch,
        "inference_ms": elapsed_ms,
        "gpu_mem_MB":   peak_mb
    }
    with (Path(__file__).parent / "last_run_metrics.json").open("w") as fp:
        json.dump(metrics, fp)

if __name__ == "__main__":
    main()
