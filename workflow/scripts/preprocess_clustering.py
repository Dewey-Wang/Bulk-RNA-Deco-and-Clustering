# === strata-aware 版本：按 (Chemo, Tumor status) 分組跑同一套預處理與特徵 ===
from typing import List, Sequence, Tuple, Dict
import numpy as np
import pandas as pd
import scanpy as sc


def strata_aware_preproc_and_features(
    ad_tr_raw: sc.AnnData,
    ad_te_raw: sc.AnnData,
    *,
    # —— preprocess 與特徵參數 ——
    do_winsor=True, q=0.995,
    do_asinh=True, c=1.0,
    n_top_hvg=2000, n_pcs_h=50, n_pcs_m=50,
    marker_topk_ct=120, target_order=None,
    # —— meta 欄位 —— 
    label_col="highLevelType", sample_col="Sample",
    batch_key_all="set",
    # ✅ 這裡可自訂 strata 欄位（預設同你之前設定）
    strata_cols=("chemo","Tumor status"),
    # —— 輸出嵌入 key —— 
    out_obsm_key="X_clust_like_deconv",
):
    if target_order is None:
        target_order = ["T","B","Endothelial","Fibroblast","Plasmablast","Myofibroblast","NK","Myeloid","Mast"]

    ad_tr = ad_tr_raw.copy()
    ad_te = ad_te_raw.copy()

    # 0) 檢查欄位
    for col in (label_col, sample_col):
        if col not in ad_tr.obs:
            raise KeyError(f"train.obs 缺少 {col}")
    for col in strata_cols:
        if col not in ad_tr.obs or col not in ad_te.obs:
            raise KeyError(f"train/test.obs 都需要 strata 欄位：{col}")

    # 1) 建 strata label
    def _group_label_row(row): return " | ".join(str(row[c]) for c in strata_cols)
    ad_tr.obs["_strata_"] = ad_tr.obs.apply(_group_label_row, axis=1)
    ad_te.obs["_strata_"] = ad_te.obs.apply(_group_label_row, axis=1)

    # 2) 基因內插（共同基因）
    genes = ad_tr.var_names.astype(str).intersection(ad_te.var_names.astype(str))
    if len(genes) == 0:
        raise ValueError("train/test 沒有共同基因")
    ad_tr = ad_tr[:, genes].copy()
    ad_te = ad_te[:, genes].copy()

    # 3) 預留嵌入容器
    Z_tr = np.zeros((ad_tr.n_obs, n_pcs_h + n_pcs_m), dtype=np.float32)
    Z_te = np.zeros((ad_te.n_obs, n_pcs_h + n_pcs_m), dtype=np.float32)

    # 小工具
    def _dense32(X):
        X = X.A if hasattr(X,"A") else (X.toarray() if hasattr(X,"toarray") else X)
        X = np.asarray(X, dtype=np.float32)
        return np.ascontiguousarray(X)

    def _stdz(A):
        mu = A.mean(axis=0, keepdims=True)
        sd = A.std(axis=0, keepdims=True) + 1e-9
        return (A - mu) / sd

    def _harmony(ad, genes_sel, key, n_comps):
        ad2 = ad[:, genes_sel].copy()
        sc.pp.scale(ad2, max_value=10)
        sc.tl.pca(ad2, n_comps=min(n_comps, ad2.n_vars, ad2.n_obs), svd_solver="arpack")
        sc.external.pp.harmony_integrate(ad2, key=key)
        return ad2.obsm["X_pca_harmony"]

    strata_levels = sorted(pd.Index(ad_tr.obs["_strata_"]).union(ad_te.obs["_strata_"]).unique())

    for s in strata_levels:
        tr_idx = np.where(ad_tr.obs["_strata_"].values == s)[0]
        te_idx = np.where(ad_te.obs["_strata_"].values == s)[0]
        if len(tr_idx)==0 and len(te_idx)==0:
            continue

        ad_tr_s = ad_tr[tr_idx].copy() if len(tr_idx)>0 else None
        ad_te_s = ad_te[te_idx].copy() if len(te_idx)>0 else None

        # 3.1 CP10k
        if ad_tr_s is not None and ad_tr_s.n_obs>0: sc.pp.normalize_total(ad_tr_s, target_sum=1e4)
        if ad_te_s is not None and ad_te_s.n_obs>0: sc.pp.normalize_total(ad_te_s, target_sum=1e4)

        # 3.2 dense
        Xtr = _dense32(ad_tr_s.X) if (ad_tr_s is not None and ad_tr_s.n_obs>0) else None
        Xte = _dense32(ad_te_s.X) if (ad_te_s is not None and ad_te_s.n_obs>0) else None

        # 3.3 strata 內 winsor
        if do_winsor and (Xtr is not None or Xte is not None):
            pool = [x for x in (Xtr, Xte) if x is not None]
            joint = pool[0] if len(pool)==1 else np.concatenate(pool, axis=0)
            up = np.quantile(joint, q, axis=0).astype(np.float32)
            if Xtr is not None:
                np.minimum(Xtr, up, out=Xtr); np.clip(Xtr, 0.0, None, out=Xtr)
            if Xte is not None:
                np.minimum(Xte, up, out=Xte); np.clip(Xte, 0.0, None, out=Xte)

        # 3.4 asinh
        if do_asinh:
            invc = 1.0/float(c)
            if Xtr is not None: Xtr = np.arcsinh(Xtr * invc).astype(np.float32)
            if Xte is not None: Xte = np.arcsinh(Xte * invc).astype(np.float32)

        if ad_tr_s is not None and ad_tr_s.n_obs>0: ad_tr_s.X = Xtr
        if ad_te_s is not None and ad_te_s.n_obs>0: ad_te_s.X = Xte

        # 3.5 只用 train(該 strata) 選 markers（若不足則空）
        markers_union = []
        if ad_tr_s is not None and ad_tr_s.n_obs > 50:
            # 用原始 train 的同 strata（用 CP10k+log1p）建 pseudobulk→Combat→per-CT topK
            mask = ad_tr_raw.obs.apply(lambda r: " | ".join(str(r[c]) for c in strata_cols), axis=1) == s
            ad_mark = ad_tr_raw[mask][:, genes].copy()
            if ad_mark.n_obs > 0:
                sc.pp.normalize_total(ad_mark, target_sum=1e4)
                sc.pp.log1p(ad_mark)
                # pseudobulk
                obs = ad_mark.obs[[sample_col, label_col]].astype(str)
                groups = obs.groupby([sample_col, label_col]).indices
                cols, names = [], []
                for (sm, ct), idx in groups.items():
                    Xi = ad_mark.X[idx,:]
                    Xi = Xi.A if hasattr(Xi,"A") else (Xi.toarray() if hasattr(Xi,"toarray") else Xi)
                    cols.append(np.asarray(Xi).mean(axis=0)); names.append(f"{sm}|{ct}")
                if cols:
                    pb = pd.DataFrame(np.column_stack(cols), index=ad_mark.var_names.astype(str), columns=names)
                    ad_pb = sc.AnnData(pb.T.copy()); ad_pb.obs["batch"] = [c.split("|",1)[0] for c in pb.columns]
                    sc.pp.combat(ad_pb, key="batch")
                    pb_c = pd.DataFrame(ad_pb.X.T, index=pb.index, columns=pb.columns)
                    # per-CT topK
                    cts = sorted({c.split("|",1)[1] for c in pb_c.columns})
                    mean_all = pb_c.mean(axis=1) + 1e-9
                    union = set()
                    denom = max(1, len(cts)-1)
                    for ct in (ct for ct in target_order if ct in cts):
                        cols_ct = [c for c in pb_c.columns if c.endswith("|"+ct)]
                        m_ct = pb_c[cols_ct].mean(axis=1)
                        rest = (mean_all*len(cts) - m_ct)/denom
                        score = np.log2((m_ct+1e-9)/(rest+1e-9)).sort_values(ascending=False)
                        union.update(score.index[:marker_topk_ct])
                    markers_union = sorted([g for g in union if g in genes])

        # 3.6 只拼有資料的 pieces → HVG（強制納入 markers）
        pieces = {}
        if ad_tr_s is not None and ad_tr_s.n_obs>0: pieces["train"] = ad_tr_s
        if ad_te_s is not None and ad_te_s.n_obs>0: pieces["test"] = ad_te_s
        if not pieces:
            continue

        ad_joint = sc.concat(pieces, label=batch_key_all, join="inner")
        sc.pp.highly_variable_genes(ad_joint, n_top_genes=n_top_hvg, flavor="seurat_v3",
                                    batch_key=batch_key_all, inplace=True)
        hv = ad_joint.var["highly_variable"].values.copy()
        name_to_idx = {g:i for i,g in enumerate(ad_joint.var_names.astype(str))}
        for g in markers_union:
            if g in name_to_idx:
                hv[name_to_idx[g]] = True
        genes_use = ad_joint.var_names[hv].astype(str).tolist()
        if len(genes_use) == 0:
            # fallback：用全部基因
            genes_use = ad_joint.var_names.astype(str).tolist()

        # 3.7 Harmony (HVG∪markers) & markers-only Harmony（markers 不足時取 HVG 的前 200）
        genes_mark_for_harmony = markers_union if len(markers_union)>0 else genes_use[:min(200, len(genes_use))]
        Z_h = _harmony(ad_joint, genes_use,              batch_key_all, n_pcs_h)
        Z_m = _harmony(ad_joint, genes_mark_for_harmony, batch_key_all, n_pcs_m)
        Z = np.concatenate([_stdz(Z_h), _stdz(Z_m)], axis=1).astype(np.float32)

        # 3.8 回填
        mask_train = (ad_joint.obs[batch_key_all].astype(str)=="train").values
        rows_train = np.where(mask_train)[0]
        rows_test  = np.where(~mask_train)[0]
        if len(tr_idx) == len(rows_train):
            Z_tr[tr_idx,:] = Z[rows_train,:]
        if len(te_idx) == len(rows_test):
            Z_te[te_idx,:] = Z[rows_test,:]

    ad_tr.obsm[out_obsm_key] = Z_tr
    ad_te.obsm[out_obsm_key] = Z_te
    return ad_tr, ad_te

def preproc_and_features_joint(
    ad_tr_raw: sc.AnnData,
    ad_te_raw: sc.AnnData,
    *,
    # --- preprocess params ---
    do_winsor: bool = True,
    q: float = 0.995,
    do_asinh: bool = True,
    c: float = 1.0,
    # --- features params ---
    n_top_hvg: int = 2000,
    marker_topk_ct: int = 120,
    target_order: Sequence[str] = ("T","B","Endothelial","Fibroblast","Plasmablast","Myofibroblast","NK","Myeloid","Mast"),
    # --- meta columns ---
    label_col: str = "highLevelType",
    sample_col: str = "Sample",
    # 可傳一個或多個 obs 欄位名（例如 ["set"], ["Sample"], ["Sample","Patient"]）
    harmony_batch_keys: Sequence[str] = ("set",),
    # --- output ---
    out_obsm_key: str = "X_clust_like_deconv",
) -> Tuple[sc.AnnData, sc.AnnData]:

    # --- normalize harmony_batch_keys to a list of column names ---
    def _to_key_list(x):
        if isinstance(x, str):
            return [x]
        try:
            return list(x)
        except Exception:
            raise TypeError("harmony_batch_keys 必須是字串或字串序列")

    harmony_keys = _to_key_list(harmony_batch_keys)

    # ---------- 0) 基本檢查 ----------
    for need in (label_col, sample_col):
        if need not in ad_tr_raw.obs:
            raise KeyError(f"train AnnData.obs 缺少欄位 '{need}'")

    # 檢查 batch keys 是否存在於 train/test（"set" 會在 concat 後自動提供）
    missing = [k for k in harmony_keys
               if k != "set" and (k not in ad_tr_raw.obs or k not in ad_te_raw.obs)]
    if missing:
        raise KeyError(
            f"Harmony batch key(s) 不在 train/test .obs：{missing}\n"
            f"train.obs: {list(ad_tr_raw.obs.columns)}\n"
            f"test.obs : {list(ad_te_raw.obs.columns)}"
        )

    # ---------- 1) 對齊共同基因 ----------
    genes = ad_tr_raw.var_names.astype(str).intersection(ad_te_raw.var_names.astype(str))
    if len(genes) == 0:
        raise ValueError("train/test 沒有共同基因")
    ad_tr = ad_tr_raw[:, genes].copy()
    ad_te = ad_te_raw[:, genes].copy()

    # ---------- 2) CP10k ----------
    sc.pp.normalize_total(ad_tr, target_sum=1e4)
    sc.pp.normalize_total(ad_te, target_sum=1e4)

    # ---------- 3) dense float32 + C-order ----------
    def _dense32(X):
        X = X.A if hasattr(X, "A") else (X.toarray() if hasattr(X, "toarray") else X)
        X = np.asarray(X, dtype=np.float32)
        return np.ascontiguousarray(X)

    Xtr = _dense32(ad_tr.X)
    Xte = _dense32(ad_te.X)

    # ---------- 4) joint winsor ----------
    if do_winsor:
        if not (0.5 < q < 1.0):
            raise ValueError(f"winsor 分位數 q 應在 (0.5,1.0)，目前 q={q}")
        joint = np.concatenate([Xtr, Xte], axis=0)
        up = np.quantile(joint, q, axis=0).astype(np.float32)
        np.minimum(Xtr, up, out=Xtr); np.clip(Xtr, 0.0, None, out=Xtr)
        np.minimum(Xte, up, out=Xte); np.clip(Xte, 0.0, None, out=Xte)

    # ---------- 5) asinh ----------
    if do_asinh:
        invc = 1.0/float(c)
        Xtr = np.arcsinh(Xtr * invc).astype(np.float32)
        Xte = np.arcsinh(Xte * invc).astype(np.float32)

    ad_tr.X = Xtr
    ad_te.X = Xte

    # ---------- 6) train-only markers（CP10k+log1p 副本） ----------
    ad_mark = ad_tr_raw[:, genes].copy()
    sc.pp.normalize_total(ad_mark, target_sum=1e4)
    sc.pp.log1p(ad_mark)
    sc.pp.scale(ad_mark, max_value=10)

    obs = ad_mark.obs[[sample_col, label_col]].astype(str)
    groups = obs.groupby([sample_col, label_col]).indices
    cols, names = [], []
    for (sm, ct), idx in groups.items():
        Xi = ad_mark.X[idx, :]
        Xi = Xi.A if hasattr(Xi, "A") else (Xi.toarray() if hasattr(Xi, "toarray") else Xi)
        cols.append(np.asarray(Xi).mean(axis=0))
        names.append(f"{sm}|{ct}")
    if not cols:
        raise ValueError("pseudobulk 為空：檢查 train 中是否有 Sample×celltype 的交集。")

    pb = pd.DataFrame(np.column_stack(cols), index=ad_mark.var_names.astype(str), columns=names)
    ad_pb = sc.AnnData(pb.T.copy())
    ad_pb.obs["batch"] = [c.split("|",1)[0] for c in pb.columns]  # batch=Sample
    sc.pp.combat(ad_pb, key="batch")
    pb_c = pd.DataFrame(ad_pb.X.T, index=pb.index, columns=pb.columns)

    cts = sorted({c.split("|",1)[1] for c in pb_c.columns})
    mean_all = pb_c.mean(axis=1) + 1e-9
    union = set()
    denom = max(1, len(cts)-1)
    for ct in (ct for ct in target_order if ct in cts):
        cols_ct = [c for c in pb_c.columns if c.endswith("|"+ct)]
        if not cols_ct: continue
        m_ct = pb_c[cols_ct].mean(axis=1)
        rest = (mean_all * len(cts) - m_ct) / denom
        score = np.log2((m_ct+1e-9)/(rest+1e-9)).sort_values(ascending=False)
        union.update(score.index[:marker_topk_ct])
    markers_union = sorted([g for g in union if g in genes])

    # ---------- 7) HVG（joint）+ force-in markers ----------
    ad_joint = sc.concat({"train": ad_tr, "test": ad_te}, label="set", join="inner")
    sc.pp.highly_variable_genes(
        ad_joint,
        n_top_genes=n_top_hvg,
        flavor="seurat_v3",
        batch_key="set",
        inplace=True
    )
    hv = ad_joint.var["highly_variable"].values.copy()
    name_to_idx = {g: i for i, g in enumerate(ad_joint.var_names.astype(str))}
    for g in markers_union:
        if g in name_to_idx:
            hv[name_to_idx[g]] = True
    genes_use = ad_joint.var_names[hv].astype(str).tolist()
    if len(genes_use) == 0:
        genes_use = ad_joint.var_names.astype(str).tolist()

    # ---------- 8) Harmony：把多個 batch keys 合成單一欄 ----------
    def _mk_harmony_batch(df_obs, keys):
        if len(keys) == 0:
            return df_obs["set"].astype(str)
        if len(keys) == 1:
            return df_obs[keys[0]].astype(str)
        return df_obs[keys[0]].astype(str).str.cat(
            [df_obs[k].astype(str) for k in keys[1:]], sep="|"
        )

    ad_joint.obs["__harm_batch__"] = _mk_harmony_batch(ad_joint.obs, harmony_keys)

    def _harmony(ad: sc.AnnData, genes_sel: List[str], key: str) -> np.ndarray:
        ad2 = ad[:, genes_sel].copy()
        sc.pp.log1p(ad2)
        sc.pp.scale(ad2, max_value=10)
        sc.tl.pca(ad2, svd_solver="arpack")
        sc.external.pp.harmony_integrate(ad2, key=key)
        if "X_pca_harmony" not in ad2.obsm:
            raise RuntimeError("Harmony failed: 'X_pca_harmony' not found.")
        return ad2.obsm["X_pca_harmony"]

    Z_h = _harmony(ad_joint, genes_use, "__harm_batch__")
    genes_mark_for_harmony = markers_union if len(markers_union) > 0 else genes_use[:min(200, len(genes_use))]
    Z_m = _harmony(ad_joint, genes_mark_for_harmony, "__harm_batch__")

    # ---------- 9) 標準化後拼接，拆回 train/test ----------
    def _stdz(A: np.ndarray) -> np.ndarray:
        mu = A.mean(axis=0, keepdims=True)
        sd = A.std(axis=0, keepdims=True) + 1e-9
        return (A - mu) / sd

    Z = np.concatenate([_stdz(Z_h), _stdz(Z_m)], axis=1).astype(np.float32)

    mask_train = (ad_joint.obs["set"].astype(str) == "train").values
    rows_tr = np.where(mask_train)[0]
    rows_te = np.where(~mask_train)[0]

    ad_tr_feat = ad_tr.copy()
    ad_te_feat = ad_te.copy()
    ad_tr_feat.obsm[out_obsm_key] = Z[rows_tr, :]
    ad_te_feat.obsm[out_obsm_key] = Z[rows_te, :]

    return ad_tr_feat, ad_te_feat
