from __future__ import annotations
import re, numpy as np, pandas as pd, matplotlib.pyplot as plt, random, warnings


# ===== utils =====
def need_scanpy():
    import scanpy as sc
    return sc

def preprocess_sc(ad):
    sc = need_scanpy()
    sc.pp.filter_cells(ad, min_genes=200)
    sc.pp.filter_genes(ad, min_cells=3)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

def pseudobulk_by_sample_ct(ad, ct_col: str, sample_col: str) -> pd.DataFrame:
    obs = ad.obs[[ct_col, sample_col]].astype(str)
    groups = obs.groupby([sample_col, ct_col]).indices
    cols, names = [], []
    for (s, ct), idx in groups.items():
        Xi = ad.X[idx]
        cols.append(np.asarray(Xi.mean(axis=0)).ravel()); names.append(f"{s}|{ct}")
    return pd.DataFrame(np.column_stack(cols), index=ad.var_names.astype(str), columns=names)

def combat_on_pb(pb: pd.DataFrame) -> pd.DataFrame:
    sc = need_scanpy()
    ad = sc.AnnData(pb.T.copy())
    ad.obs["batch"] = pd.Categorical([c.split("|",1)[0] for c in pb.columns])  # batch = Sample
    sc.pp.combat(ad, key="batch")
    return pd.DataFrame(ad.X.T, index=pb.index, columns=pb.columns)

def stratified_marker_union(pb: pd.DataFrame, topk_per_ct: int, ct_order: list[str]) -> list[str]:
    cts = sorted({c.split("|",1)[1] for c in pb.columns})
    order = [ct for ct in ct_order if ct in cts] + [ct for ct in cts if ct not in ct_order]
    mean_all = pb.mean(axis=1) + 1e-9
    sel = set()
    for ct in order:
        cols = [c for c in pb.columns if c.split("|",1)[1]==ct]
        if not cols: continue
        m_ct = pb[cols].mean(axis=1)
        rest = (mean_all * len(cts) - m_ct) / max(1e-12, (len(cts)-1))
        score = np.log2((m_ct+1e-9)/(rest+1e-9)).sort_values(ascending=False)
        sel.update(score.index[:topk_per_ct])
    return list(sel)

def cpm10k(df: pd.DataFrame) -> pd.DataFrame:
    lib = df.sum(axis=0).astype(float); lib[lib==0.0] = 1.0
    return df.div(lib, axis=1) * 1e4

def winsorize_per_gene_pairwise(Btr: pd.DataFrame, Bte: pd.DataFrame, q: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    joint = pd.concat([Btr, Bte], axis=1)
    up = joint.quantile(q, axis=1)
    return Btr.clip(lower=0.0, upper=up, axis=0), Bte.clip(lower=0.0, upper=up, axis=0)

def asinh_vst(df: pd.DataFrame, c: float = 1.0) -> pd.DataFrame:
    return np.arcsinh(df / float(c))

def parse_sample_from_bulk_col(col: str) -> str:
    m = re.match(r"^(s\d+)", str(col).lower())
    return m.group(1) if m else str(col)

def simplex_proj(v: np.ndarray) -> np.ndarray:
    u = np.sort(v)[::-1]; cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)

# ILR
def helmert_basis(K: int) -> np.ndarray:
    H = np.zeros((K, K-1))
    for i in range(1, K):
        a = np.ones(i)/i
        H[:i, i-1] = a
        H[i,   i-1] = -1.0
    for j in range(K-1):
        col = H[:, j]; H[:, j] = col / (np.linalg.norm(col) + 1e-12)
    return H

def ilr(p: np.ndarray, H: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.maximum(p, eps)
    p = p / p.sum(axis=-1, keepdims=True)
    return np.log(p) @ H

def ilr_inv(z: np.ndarray, H: np.ndarray) -> np.ndarray:
    logp = z @ H.T
    x = np.exp(logp)
    return x / x.sum(axis=-1, keepdims=True)
