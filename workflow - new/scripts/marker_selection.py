# ==================== markers on log1p + Combat space ====================
import numpy as np
import pandas as pd
from collections import Counter
from scipy.sparse import issparse  # <<< 用 scipy.sparse

def build_pb_log_balanced(
    adata,
    celltype_col: str = "highLevelType",
    stratum_col: str = "Sample",
    min_cells_per_pair: int = 20,           # 建議 >0，例如 20/50
    downsample_to_n: int | None = None,     # 每個 (Sample,CT) 下採樣到固定 n；None=不下
    sample_agg: str = "median",             # "mean" 或 "median"
    random_state: int = 42,
):
    """
    Return:
      pb_sc : DataFrame, genes × (sample|CT)  —— 用於 per-sample 視覺化/統計
      pb_ct : DataFrame, genes × CT          —— 用於打分/選標（等權樣本聚合）
      pair_counts : Series, 細胞數（index=(sample,CT)）
    需求：
      adata.X 已在 log1p+Combat 空間（你目前 ad_tr_pp）
      adata.obs 需有 [stratum_col, celltype_col]
    """
    # --- to dense (避免 pandas 對稀疏 mean 慢/不穩) ---
    X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X, dtype=float)
    genes = adata.var_names
    obs   = adata.obs[[stratum_col, celltype_col]].astype(str).copy()

    # --- cell-level DF ---
    df = pd.DataFrame(X, index=adata.obs_names, columns=genes)
    df[stratum_col]  = obs[stratum_col].values
    df[celltype_col] = obs[celltype_col].values

    # --- 計數與過濾 (Sample,CT) 群 ---
    counts = df.groupby([stratum_col, celltype_col]).size().rename("n_cells")
    if min_cells_per_pair and min_cells_per_pair > 0:
        keep_pairs = counts[counts >= int(min_cells_per_pair)].index
        key = pd.MultiIndex.from_arrays([df[stratum_col], df[celltype_col]])
        df = df.loc[key.isin(keep_pairs)].copy()
        # 重新計數（過濾後）
        counts = df.groupby([stratum_col, celltype_col]).size().rename("n_cells")

    # --- 可選：群內下採樣到固定 n ---
    if downsample_to_n is not None:
        rng = np.random.default_rng(random_state)
        parts = []
        for (s, ct), sub in df.groupby([stratum_col, celltype_col], sort=False):
            if len(sub) > downsample_to_n:
                idx = rng.choice(sub.index.values, size=downsample_to_n, replace=False)
                parts.append(sub.loc[idx])
            else:
                parts.append(sub)
        df = pd.concat(parts, axis=0)

    # --- (Sample,CT) 細胞平均（僅數值列） ---
    num_cols = df.select_dtypes(include=[np.number]).columns
    sc_means = df.groupby([stratum_col, celltype_col])[num_cols].mean()

    # --- genes × (sample|CT) ---
    pb_sc = sc_means.T
    pb_sc.columns = [f"{s}|{ct}" for (s, ct) in pb_sc.columns.to_list()]
    # 去掉全 0 欄（保險）
    pb_sc = pb_sc.loc[:, pb_sc.sum(axis=0).values != 0]

    # --- genes × CT（樣本等權聚合：mean/median）---
    def _col_ct(col): return col.split("|", 1)[1]
    ct_labels = pd.Index([_col_ct(c) for c in pb_sc.columns])
    pb_ct_parts = []
    for ct in sorted(ct_labels.unique()):
        cols = [c for c, lab in zip(pb_sc.columns, ct_labels) if lab == ct]
        if not cols:
            continue
        ser = (pb_sc[cols].median(axis=1) if sample_agg == "median" else pb_sc[cols].mean(axis=1)).rename(ct)
        pb_ct_parts.append(ser)
    pb_ct = pd.concat(pb_ct_parts, axis=1) if pb_ct_parts else pd.DataFrame(index=pb_sc.index)

    return pb_sc, pb_ct, counts.sort_values(ascending=False)


# ========== 取得每個 CT 的 marker → dict(CT -> [genes])，並維持給定順序 ==========
def unique_markers_drop(per_ct_markers: dict[str, list[str]]):
    """
    嚴格唯一：凡在 >=2 個 CT 出現的基因直接移除。
    返回: (per_ct_unique, union_unique, gene_counts)
    """
    all_genes = [g for genes in per_ct_markers.values() for g in genes]
    counts = Counter(all_genes)

    per_ct_unique = {ct: [g for g in genes if counts[g] == 1]
                     for ct, genes in per_ct_markers.items()}
    union_unique = sorted({g for genes in per_ct_unique.values() for g in genes})
    gene_counts = pd.Series(counts).sort_values(ascending=False)
    return per_ct_unique, union_unique, gene_counts

# ---- 2) per-CT markers with log-space difference score ----
def select_markers_log(pb_log: pd.DataFrame,
                       topk_per_ct: int,
                       ct_order: list[str]) -> tuple[dict[str, list[str]], list[str], pd.DataFrame]:
    """
    Per-CT Top-K markers on log space using score = m_ct - rest.
    Returns: (per_ct_markers, union_markers, score_table)
    """
    cts_present = sorted({c.split("|", 1)[1] for c in pb_log.columns})
    order = [ct for ct in ct_order if ct in cts_present] + [ct for ct in cts_present if ct not in ct_order]

    mean_all = pb_log.mean(axis=1)
    per_ct = {}
    score_mat = {}

    for ct in order:
        cols = [c for c in pb_log.columns if c.split("|", 1)[1] == ct]
        if not cols:
            continue
        m_ct  = pb_log[cols].mean(axis=1)
        rest  = (mean_all * len(cts_present) - m_ct) / max(1, (len(cts_present) - 1))
        score = (m_ct - rest)  # why: log 空間加性差值 ≈ 幾何平均 logFC
        topk  = list(score.sort_values(ascending=False).index[:topk_per_ct])
        per_ct[ct] = topk
        score_mat[ct] = score

    union_markers = sorted(set().union(*per_ct.values())) if per_ct else []
    score_table = pd.DataFrame(score_mat) if score_mat else pd.DataFrame(index=pb_log.index)

    return per_ct, union_markers, score_table



import numpy as np
import pandas as pd
from collections import Counter

# --------- Mean-Ratio：直接吃 pb_ct_mean（genes × CT） ----------
def mean_ratio_markers_ctmean(
    pb_ct_mean: pd.DataFrame,
    topk_per_ct: int,
    ct_order: list[str],
    *,
    pseudocount: float = 1e-9,
) -> tuple[dict[str, list[str]], list[str], pd.DataFrame]:
    """
    score(g, ct) = (mean_ct + eps) / (max_{ct'!=ct} mean_ct' + eps)
    Returns: (per_ct_markers, union_markers, score_table)
    """
    cols_present = list(pb_ct_mean.columns)
    order = [ct for ct in ct_order if ct in cols_present] + [ct for ct in cols_present if ct not in ct_order]
    M = pb_ct_mean.loc[:, order].astype(float)

    if M.shape[1] < 2:
        empty_scores = pd.DataFrame(index=M.index, columns=order)
        return {ct: [] for ct in order}, [], empty_scores

    score_mat: dict[str, pd.Series] = {}
    per_ct: dict[str, list[str]] = {}

    for i, ct in enumerate(order):
        m_ct = M[ct]
        other_cols = order[:i] + order[i+1:]
        m_other_max = M[other_cols].max(axis=1)
        score = (m_ct + pseudocount) / (m_other_max + pseudocount)
        score_mat[ct] = score
        per_ct[ct] = list(score.sort_values(ascending=False).index[:topk_per_ct])

    union = sorted(set().union(*per_ct.values())) if per_ct else []

    score_table = pd.DataFrame(score_mat, index=M.index)
    return per_ct, union, score_table


# --------- Log-diff：直接吃 pb_ct_mean（genes × CT） ----------
def select_markers_log_from_ctmean(
    pb_ct_mean: pd.DataFrame,
    topk_per_ct: int,
    ct_order: list[str],
) -> tuple[dict[str, list[str]], list[str], pd.DataFrame]:
    """
    score(g, ct) = mean_ct - mean_rest
    Returns: (per_ct_markers, union_markers, score_table)
    """
    cts_present = [ct for ct in ct_order if ct in pb_ct_mean.columns] + \
                  [ct for ct in pb_ct_mean.columns if ct not in ct_order]
    M = pb_ct_mean.loc[:, cts_present].astype(float)

    mean_all = M.mean(axis=1)
    per_ct, score_mat = {}, {}

    C = M.shape[1]
    for ct in cts_present:
        m_ct = M[ct]
        rest = (mean_all * C - m_ct) / max(1, (C - 1))
        score = (m_ct - rest)
        score_mat[ct] = score
        per_ct[ct] = list(score.sort_values(ascending=False).index[:topk_per_ct])

    union = sorted(set().union(*per_ct.values())) if per_ct else []
    score_table = pd.DataFrame(score_mat, index=M.index)
    return per_ct, union, score_table


# --------- 單一入口：支援 genes×CT 或 genes×(sample|CT) ----------
def select_markers_any(
    pb: pd.DataFrame,
    topk_per_ct: int,
    ct_order: list[str],
    *,
    method: str = "logdiff",         # "logdiff" 或 "meanratio"
    pseudocount: float = 1e-9,
) -> tuple[dict[str, list[str]], list[str], pd.DataFrame]:
    """
    pb 可以是：
      - genes × CT
      - genes × (sample|CT)  → 會先等權聚合為 genes × CT 再打分
    method:
      - "logdiff": score = mean_ct - mean_rest
      - "meanratio": score = (mean_ct+eps)/(max_other+eps)
    Returns: (per_ct_markers, union_markers, score_table)
    """
    # 若為 sample|CT 先聚合到 CT
    if any(("|" in str(c)) for c in pb.columns):
        def _col_ct(c): return str(c).split("|", 1)[1]
        ct_labels = [_col_ct(c) for c in pb.columns]
        parts = []
        for ct in sorted(set(ct_labels)):
            cols = [c for c, lbl in zip(pb.columns, ct_labels) if lbl == ct]
            parts.append(pb[cols].mean(axis=1).rename(ct))
        pb_ct_mean = pd.concat(parts, axis=1)
    else:
        pb_ct_mean = pb

    method = method.lower().strip()
    if method == "logdiff":
        return select_markers_log_from_ctmean(pb_ct_mean, topk_per_ct, ct_order)
    elif method == "meanratio":
        return mean_ratio_markers_ctmean(
            pb_ct_mean, topk_per_ct, ct_order,
            pseudocount=pseudocount
        )
    else:
        raise ValueError("method 必須為 'logdiff' 或 'meanratio'")
