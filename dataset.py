#!/usr/bin/env python3
"""
Dataset loader utilities for the inference experiments.

`get_dataset()`  – returns the full six‑tuple (for scripts that need
                   every field).

`load_dataset()` – backward‑compatibility shim that returns exactly the
                   three objects older versions of main.py expect:
                   (X, A_package, y).  It also honours an optional
                   `device=` argument.
"""

import os, torch
from torch_geometric.datasets import Planetoid, Yelp, Flickr, NELL


# ───────────────────────────────────────────────────────────────────────────
def get_dataset(name: str, root: str):
    """
    Loads a graph dataset and constructs a CSR adjacency tensor.

    Returns
    -------
    N : int
        number of nodes
    F : int
        number of input features
    C : int
        number of classes
    A_pkg : tuple
        (A_csr, None, crow, col, deg) – everything main.py may need
    X : torch.Tensor [N, F]  (dense)
    y : torch.Tensor [N]     (labels)
    """
    root = os.path.expanduser(root)

    # ---------- choose dataset -------------------------------------------------
    if name in ("Cora", "Pubmed", "Citeseer"):
        ds = Planetoid(root, name)
    elif name == "Yelp":
        ds = Yelp(root)
    elif name == "Flickr":
        ds = Flickr(root)
    elif name == "NELL":
        ds = NELL(root)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    data = ds[0]

    N = data.num_nodes
    X = data.x.to_dense() if data.x.is_sparse else data.x

    # build CSR adjacency -------------------------------------------------------
    A = torch.sparse_coo_tensor(
            data.edge_index,
            torch.ones(data.edge_index.size(1), device=data.edge_index.device),
            (N, N)
        ).to_sparse_csr()

    crow = A.crow_indices()
    col  = A.col_indices()
    deg  = crow[1:] - crow[:-1]

    return N, data.num_node_features, ds.num_classes, (A, None, crow, col, deg), X, data.y


# ───────────────────────────────────────────────────────────────────────────
# Back‑compat shim (older main.py calls this)
def load_dataset(name: str, root: str, *_, device=None, **__):
    """
    Parameters
    ----------
    name, root
        Same as `get_dataset`.
    device : torch.device or str, optional
        If given, moves X and y to that device.

    Returns
    -------
    X, A_pkg, y   (exactly three objects, as expected by main.py)
    """
    _, _F, _C, A_pkg, X, y = get_dataset(name, root)

    if device is not None:
        X = X.to(device)
        y = y.to(device)
        # A_pkg contains CSR indices that are already on the same device
        # as X / y, so no extra move is needed.

    return X, A_pkg, y
