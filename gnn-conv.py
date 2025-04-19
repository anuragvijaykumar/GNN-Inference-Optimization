"""Lightweight custom GCN / GIN / GAT layers used **only for inference**.
They wrap the CUDA kernels that ship with the original PruneGNN repo.  We keep
forward‑path autograd.Functions (backward omitted) and place NVTX ranges
("comb", "agg") around the two logical phases so Nsight Compute can group
kernels under them.
"""
import math, torch, torch.nn as nn, nvtx, prune_gnn
from utils import prune_irr
import dgl.sparse

# -------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------

def _dense_mm(X, W):
    """Matmul that gracefully falls back if W is sparse."""
    try:
        return torch.matmul(X, W)
    except RuntimeError:
        return torch.matmul(X, W.to_dense())

# -------------------------------------------------------------------------
#       Autograd.Functions (forward only)
# -------------------------------------------------------------------------
class _GCNFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, A, args, lname):
        A_csr, _, row_ptr, col_idx, deg = A
        if args.sparsity_type == "irregular":
            with nvtx.annotate("comb"): C = prune_irr(args, _dense_mm(X, W), args.sparsity_rate)
            with nvtx.annotate("agg"):
                try: return torch.matmul(A_csr, C).to_dense()
                except RuntimeError: return torch.matmul(A_csr, C)
        # structured
        with nvtx.annotate("comb"):
            C = (_dense_mm(X, W) if X.is_sparse_csr else prune_gnn.cublas_gemm(X, W))
        with nvtx.annotate("agg"):
            if args.kernel_type == "pruneSp":
                return prune_gnn.prune_spmm(C, row_ptr, col_idx, deg, args.tpw, args.gin_flag, args.epsilon)
            return prune_gnn.cusparse_spmm_row(C, row_ptr, col_idx, deg)

class _GINFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, A, args, lname):
        A_csr, _, row_ptr, col_idx, deg = A
        if args.sparsity_type == "irregular":
            with nvtx.annotate("agg"):
                if X.is_sparse_csr:
                    agg = torch.matmul(A_csr, X).to_dense() + (1+args.epsilon)*X
                else:
                    agg = prune_gnn.cusparse_spmm_row(X, row_ptr, col_idx, deg) + (1+args.epsilon)*X
            agg = prune_irr(args, agg, args.sparsity_rate)
            with nvtx.annotate("comb"):
                return _dense_mm(agg, W)
        # structured
        with nvtx.annotate("agg"):
            if args.kernel_type == "pruneSp":
                agg = prune_gnn.prune_spmm(X, row_ptr, col_idx, deg, args.tpw, args.gin_flag, args.epsilon)
            else:
                agg = prune_gnn.cusparse_spmm_row(X, row_ptr, col_idx, deg) + (1+args.epsilon)*X
        with nvtx.annotate("comb"):
            return prune_gnn.cublas_gemm(agg, W)

class _GATFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, Wr, Wl, A, args, lname):
        A_csr, A_dgl, row_ptr, col_idx, deg = A
        leaky = nn.LeakyReLU(0.2)
        if args.sparsity_type == "irregular":
            with nvtx.annotate("comb"):
                C = prune_irr(args, _dense_mm(X, W), args.sparsity_rate)
            with nvtx.annotate("agg"):
                h_r = _dense_mm(X, Wr); h_l = _dense_mm(X, Wl)
                a = dgl.sparse.sddmm(A_dgl, h_l, h_r.T); A_csr.values = leaky(a.val)
                return torch.matmul(A_csr, C).to_dense()
        # structured
        with nvtx.annotate("comb"):
            C = (_dense_mm(X, W) if X.is_sparse_csr else prune_gnn.cublas_gemm(X, W))
        with nvtx.annotate("agg"):
            if X.is_sparse_csr:
                h_r = torch.matmul(X, Wr); h_l = torch.matmul(X, Wl)
            else:
                h_r = prune_gnn.cublas_gemm(X, Wr); h_l = prune_gnn.cublas_gemm(X, Wl)
            a = dgl.sparse.sddmm(A_dgl, h_l, h_r.T); deg = leaky(a.val)
            if args.kernel_type == "pruneSp":
                return prune_gnn.prune_spmm(C, row_ptr, col_idx, deg, args.tpw, args.gin_flag, args.epsilon)
            return prune_gnn.cusparse_spmm_row(C, row_ptr, col_idx, deg)

# -------------------------------------------------------------------------
#       nn.Module wrappers
# -------------------------------------------------------------------------

def _init_w(in_dim, out_dim):
    w = nn.Parameter(torch.empty(in_dim, out_dim)); nn.init.xavier_uniform_(w); return w

class GCNConv(nn.Module):
    def __init__(self, args, in_d, out_d, lname):
        super().__init__(); self.W=_init_w(in_d,out_d); self.args=args; self.lname=lname
    def forward(self,X,A,args):
        W = prune_irr(args,self.W,args.sparsity_rate) if args.sparsity_type=="irregular" else self.W
        return _GCNFunc.apply(X,W,A,args,self.lname)

class GINConv(nn.Module):
    def __init__(self,args,in_d,out_d,lname):
        super().__init__(); self.W=_init_w(in_d,out_d); self.args=args; self.lname=lname
    def forward(self,X,A,args):
        W = prune_irr(args,self.W,args.sparsity_rate) if args.sparsity_type=="irregular" else self.W
        return _GINFunc.apply(X,W,A,args,self.lname)

class GATConv(nn.Module):
    def __init__(self,args,in_d,out_d,heads,layer_name):
        super().__init__(); self.W=_init_w(in_d,out_d); self.Wr=_init_w(in_d,out_d); self.Wl=_init_w(in_d,out_d)
        self.lname=layer_name; self.args=args
    def forward(self,X,A,args):
        if args.sparsity_type=="irregular":
            W  = prune_irr(args,self.W,args.sparsity_rate)
            Wr = prune_irr(args,self.Wr,args.sparsity_rate)
            Wl = prune_irr(args,self.Wl,args.sparsity_rate)
        else:
            W,Wr,Wl = self.W,self.Wr,self.Wl
        return _GATFunc.apply(X,W,Wr,Wl,A,args,self.lname)