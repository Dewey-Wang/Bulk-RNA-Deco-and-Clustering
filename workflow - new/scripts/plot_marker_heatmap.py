import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.sparse import issparse
from math import ceil

# ---------- builders & detectors ----------
def build_pb_log(adata, stratum_col: str, celltype_col: str) -> pd.DataFrame:
    """AnnData(log1p+Combat) -> genes × (sample|CT)."""
    X = adata.X.A if issparse(adata.X) else np.asarray(adata.X)
    df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    meta = adata.obs[[stratum_col, celltype_col]].astype(str)
    df[stratum_col]  = meta[stratum_col].values
    df[celltype_col] = meta[celltype_col].values
    pb = (df.groupby([stratum_col, celltype_col]).mean()
            .drop(columns=[stratum_col, celltype_col], errors="ignore")).T
    pb.columns = [f"{s}|{ct}" for (s, ct) in pb.columns.to_list()]
    return pb.loc[:, pb.sum(axis=0) != 0]

def is_sample_ct(columns) -> bool:
    """列名是否包含 sample 維度（以 '|' 判斷）"""
    return any(("|" in str(c)) for c in columns)

def ensure_pb(pb_or_adata, stratum_col="Sample", celltype_col="highLevelType") -> pd.DataFrame:
    """AnnData -> pb_log；DataFrame -> 原樣返回。"""
    if isinstance(pb_or_adata, pd.DataFrame):
        return pb_or_adata
    return build_pb_log(pb_or_adata, stratum_col=stratum_col, celltype_col=celltype_col)

def to_pb_ct_mean(pb_any: pd.DataFrame, agg: str = "mean") -> pd.DataFrame:
    """genes × (sample|CT) -> genes × CT；若已是 genes × CT 直接回傳。"""
    if not is_sample_ct(pb_any.columns):
        return pb_any
    def _ct(c): return str(c).split("|", 1)[1]
    parts = []
    for ct in sorted({_ct(c) for c in pb_any.columns}):
        cols = [c for c in pb_any.columns if _ct(c) == ct]
        ser = (pb_any[cols].median(axis=1) if agg=="median" else pb_any[cols].mean(axis=1)).rename(ct)
        parts.append(ser)
    return pd.concat(parts, axis=1)  # genes × CT

# ---------- stats helpers ----------
def pvals_by_ct_any(pb_any: pd.DataFrame, ct_list: list[str]) -> pd.DataFrame:
    """
    若有 sample 維度 (sample|CT) → Welch t-test：該 CT vs 其他 CT。
    若已是 genes × CT（無重複）→ 回 NaN。
    """
    if not is_sample_ct(pb_any.columns):
        return pd.DataFrame(np.nan, index=pb_any.index, columns=ct_list)
    tags = [c.split("|",1)[1] for c in pb_any.columns]
    res = {}
    for ct in ct_list:
        in_cols  = [c for c,t in zip(pb_any.columns, tags) if t==ct]
        out_cols = [c for c,t in zip(pb_any.columns, tags) if t!=ct]
        if len(in_cols) < 2 or len(out_cols) < 2:
            res[ct] = pd.Series(np.nan, index=pb_any.index); continue
        p = []
        for g in pb_any.index:
            a = pb_any.loc[g, in_cols].astype(float).values
            b = pb_any.loc[g, out_cols].astype(float).values
            _, pv = ttest_ind(a, b, equal_var=False, nan_policy="omit")
            p.append(pv)
        res[ct] = pd.Series(p, index=pb_any.index)
    return pd.DataFrame(res)

# ---------- CT 聚合平均（畫圖用） ----------
def mean_by_ct(pb_any: pd.DataFrame, agg: str = "mean") -> pd.DataFrame:
    """回 genes × CT；若 pb_any 已是 genes × CT，直接回傳。"""
    if not is_sample_ct(pb_any.columns):
        return pb_any
    tags = [c.split("|",1)[1] for c in pb_any.columns]
    out = {}
    for ct in sorted(set(tags)):
        cols = [c for c,t in zip(pb_any.columns, tags) if t==ct]
        if cols:
            out[ct] = (pb_any[cols].median(axis=1) if agg=="median" else pb_any[cols].mean(axis=1))
    return pd.DataFrame(out)

# ---------- 把 union 基因平均分配回 CT（視覺化排序） ----------
def pack_markers_per_ct(cts: list[str], marker_list: list[str], pb_ct_mean: pd.DataFrame) -> dict[str, list[str]]:
    genes = [g for g in marker_list if g in pb_ct_mean.index]
    res = {ct: [] for ct in cts}
    if not genes: return res
    M = pb_ct_mean.loc[genes, cts]
    mean_all = M.mean(axis=1) + 1e-9
    k = max(1, ceil(len(genes) / max(1, len(cts))))
    for ct in cts:
        m_ct = M[ct] + 1e-9
        rest = (mean_all * len(cts) - m_ct) / max(1, (len(cts)-1))
        score = np.log2(m_ct / rest)
        res[ct] = score.sort_values(ascending=False).index[:k].tolist()
    return res

# ---------- heatmap ----------
def plot_marker_heatmap(pb_any: pd.DataFrame,
                        markers_by_ct: dict[str, list[str]],
                        column_ct_order: list[str],
                        title: str,
                        show_pbars: bool = True,
                        pval_df: pd.DataFrame | None = None,
                        figsize=(9,8),
                        agg_for_mean: str = "mean"):
    M = mean_by_ct(pb_any, agg=agg_for_mean)     # genes × CT
    col_order = [ct for ct in column_ct_order if ct in M.columns]
    rows = [g for ct in column_ct_order for g in markers_by_ct.get(ct, []) if g in M.index]
    if not rows:
        print(f"[{title}] 沒有可用的 marker 可畫。"); return
    mat = M.loc[rows, col_order].astype(float).values
    mat = (mat - mat.mean(axis=1, keepdims=True)) / (mat.std(axis=1, keepdims=True) + 1e-9)

    block_sizes = [len([g for g in markers_by_ct.get(ct, []) if g in M.index]) for ct in column_ct_order]
    block_pos = np.cumsum([0] + block_sizes)

    plt.figure(figsize=figsize)
    im = plt.imshow(mat, aspect="auto")
    plt.xticks(range(len(col_order)), col_order, rotation=45, ha="right")
    ytick_pos, ytick_lab = [], []
    for i,ct in enumerate(column_ct_order):
        if block_sizes[i] > 0: ytick_pos.append(block_pos[i]); ytick_lab.append(ct)
    plt.yticks(ytick_pos, ytick_lab)
    for y in block_pos: plt.axhline(y-0.5, linestyle=":", linewidth=0.8)
    plt.title(title); plt.colorbar(im, fraction=0.025, pad=0.02)

    if show_pbars and (pval_df is not None):
        ax = plt.gca(); ax2 = ax.inset_axes([1.02, 0.0, 0.15, 1.0])
        # row_labels 與 bars 對齊
        row_labels = [(ct, g) for ct in column_ct_order for g in markers_by_ct.get(ct, []) if g in M.index]
        bars = []
        for (ct,g) in row_labels:
            pv = pval_df.loc[g, ct] if (g in pval_df.index and ct in pval_df.columns) else np.nan
            bars.append(-np.log10(pv) if (pv is not None and np.isfinite(pv) and pv>0) else 0.0)
        y = np.arange(len(bars)); ax2.barh(y, bars)
        ax2.set_ylim(-0.5, len(bars)-0.5); ax2.set_yticks([]); ax2.set_xlabel(r"-log$_{10}$(p)")
        for yy in block_pos: ax2.axhline(yy-0.5, linestyle=":", linewidth=0.8)

    plt.tight_layout(); plt.show()

# ---------- 統一入口（同時支援 pb_log / pb_ct_mean + dict / list） ----------
def plot_from_markers(pb_or_adata,
                      markers,                   # dict: {ct:[...]} 或 list: union
                      ct_order: list[str],
                      title: str,
                      stratum_col: str = "Sample",
                      celltype_col: str = "highLevelType",
                      agg_for_mean: str = "mean",
                      pval_out_csv: str | None = None):
    # 1) 拿到 pb_any（保留樣本維度以便有 p 值）與 pb_ct_mean（繪圖排序）
    pb_any = ensure_pb(pb_or_adata, stratum_col=stratum_col, celltype_col=celltype_col)
    pb_ct  = to_pb_ct_mean(pb_any, agg=agg_for_mean)

    # 2) per-CT markers
    if isinstance(markers, dict):
        markers_by_ct = {ct: [g for g in (genes or []) if g in pb_ct.index] for ct, genes in markers.items()}
        union = sorted({g for gs in markers_by_ct.values() for g in gs})
    else:
        union = [g for g in markers if g in pb_ct.index]
        markers_by_ct = pack_markers_per_ct(ct_order, union, pb_ct)

    # 3) p 值（只有 pb_any 含樣本維度時才有）
    pvals = pvals_by_ct_any(pb_any, ct_order)
    if pval_out_csv and union:
        pvals.loc[sorted(set(union).intersection(pvals.index))].to_csv(pval_out_csv)

    # 4) 畫圖
    plot_marker_heatmap(pb_any, markers_by_ct, ct_order, title,
                        show_pbars=True, pval_df=pvals, agg_for_mean=agg_for_mean)
