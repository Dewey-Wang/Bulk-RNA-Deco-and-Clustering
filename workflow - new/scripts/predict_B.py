# import numpy as np
# import pandas as pd
# import scanpy as sc
# from scipy.sparse import issparse
# from scipy.optimize import nnls
# from sklearn.model_selection import KFold

# def deconv_ridge_nnls_cpm_zscore_te(
#     ad_tr: sc.AnnData,
#     ad_te: sc.AnnData,
#     bulk_tr_raw: pd.DataFrame,   # genes × train_samples
#     bulk_te_raw: pd.DataFrame,   # genes × test_samples
#     truth_tr: pd.DataFrame,      # celltypes × train_samples
#     marker_genes: list[str],
#     *,
#     ct_keep: list[str],                  # 僅用於CV/評估/回傳
#     target_sum: float = 1_000_000.0,
#     min_cells_per_pair: int = 50,
#     downsample_to_n: int | None = None,
#     alpha_grid = None,                   # None → np.logspace(-6, 3, 28)
#     n_splits: int = 5,
#     random_state: int = 42,
#     celltype_col: str = "highLevelType",
#     sample_col: str = "Sample",
# ):
#     """
#     單細胞(train+test；若 test 無 celltype 則只用 train) → 建簽名 S 與 z-score 參考。
#     在 train bulk 上僅以 ct_keep 做 CV 挑 α，最後：
#       - 回傳 test bulk 的 ct_keep 預測 pred_te_keep
#       - 回傳 train bulk 上 ct_keep 的 per-CT RMSE 與 mean RMSE（用 alpha_best、同前處理）
#       - 回傳 CV 的 out-of-fold 預測 oof_pred_keep（ct_keep × train_samples）
#     Returns
#     -------
#     pred_te_keep    : DataFrame (ct_keep × test_samples)
#     rmse_ct_keep    : Series   (ct_keep)               # on train
#     rmse_mean_keep  : float                            # on train
#     alpha_best      : float
#     oof_pred_keep   : DataFrame (ct_keep × train_samples)  # CV驗證折預測
#     """

#     # ---------- helpers ----------
#     def _to_dense(X):
#         return X.toarray() if issparse(X) else np.asarray(X)

#     def _cpm(df: pd.DataFrame | None):
#         if df is None: return None
#         s = df.sum(axis=0).replace(0, 1.0)
#         return df.divide(s, axis=1) * 1_000_000

#     def _build_signature_linear(ad_: sc.AnnData):
#         ad = ad_.copy()
#         sc.pp.normalize_total(ad, target_sum=target_sum)
#         X = _to_dense(ad.X)
#         df = pd.DataFrame(X, index=ad.obs_names, columns=ad.var_names)
#         obs = ad.obs[[sample_col, celltype_col]].astype(str)
#         df[sample_col]   = obs[sample_col].values
#         df[celltype_col] = obs[celltype_col].values

#         counts = df.groupby([sample_col, celltype_col]).size()
#         keep_pairs = counts[counts >= min_cells_per_pair].index
#         key = pd.MultiIndex.from_frame(df[[sample_col, celltype_col]])
#         df = df.loc[key.isin(keep_pairs)].copy()

#         if downsample_to_n is not None:
#             rng = np.random.default_rng(random_state)
#             parts = []
#             for (s, ct), sub in df.groupby([sample_col, celltype_col], sort=False):
#                 if len(sub) > downsample_to_n:
#                     idx = rng.choice(sub.index.values, size=downsample_to_n, replace=False)
#                     parts.append(sub.loc[idx])
#                 else:
#                     parts.append(sub)
#             df = pd.concat(parts, axis=0)

#         pair_means = df.groupby([sample_col, celltype_col]).mean()
#         ct_means   = pair_means.groupby(level=1).mean()
#         return ct_means.T  # genes × CT

#     def _zscore_to_ref(S_ref: pd.DataFrame, B: pd.DataFrame):
#         common = S_ref.index.intersection(B.index)
#         S2 = S_ref.loc[common].fillna(0.0).copy()
#         B2 = B.loc[common].fillna(0.0).copy()
#         mu = S2.mean(axis=1)
#         sd = S2.std(axis=1)
#         sd = sd.where(sd > 0, 1.0)  # 避免除0
#         Sz = (S2.subtract(mu, axis=0)).divide(sd, axis=0)
#         Bz = (B2.subtract(mu, axis=0)).divide(sd, axis=0)
#         return Sz, Bz

#     def _drop_nonfinite(*dfs: pd.DataFrame):
#         mask = np.isfinite(dfs[0].values).all(axis=1)
#         for D in dfs[1:]:
#             mask &= np.isfinite(D.values).all(axis=1)
#         return [D.loc[mask].copy() for D in dfs], int(mask.sum())

#     def _ridge_nnls_single(S_df: pd.DataFrame, y: pd.Series, alpha: float) -> pd.Series:
#         # min ||S w - y||^2 + alpha||w||^2, s.t. w>=0  → NNLS on augmented system
#         k = S_df.shape[1]
#         A = np.vstack([S_df.values, np.sqrt(alpha) * np.eye(k)])
#         b = np.concatenate([y.values, np.zeros(k)])
#         w, _ = nnls(A, b)
#         s = w.sum()
#         if s > 0: w = w / s  # 和=1（soft）
#         return pd.Series(w, index=S_df.columns, name=y.name)

#     def _ridge_nnls_matrix(S_df: pd.DataFrame, B_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
#         return pd.concat([_ridge_nnls_single(S_df, B_df[c], alpha) for c in B_df.columns], axis=1)

#     def _cv_select_alpha(S_df: pd.DataFrame, B_df: pd.DataFrame, truth_df: pd.DataFrame, ct_eval: list[str]):
#         alphas = np.array(alpha_grid, dtype=float) if alpha_grid is not None else np.logspace(-6, 3, 28)
#         cols = B_df.columns.intersection(truth_df.columns)
#         if len(cols) < 2: cols = list(cols)
#         Bc   = B_df[cols]
#         Tref = truth_df.loc[ct_eval, cols]
#         kf = KFold(n_splits=min(n_splits, max(2, len(cols))), shuffle=True, random_state=random_state)
#         scores = []
#         for a in alphas:
#             rmses = []
#             for _, val_idx in kf.split(cols):
#                 val_cols = cols[val_idx]
#                 P = _ridge_nnls_matrix(S_df, Bc[val_cols], a).loc[ct_eval]
#                 rmses.append(np.sqrt(((P - Tref[val_cols])**2).mean().mean()))
#             scores.append((float(a), float(np.mean(rmses))))
#         best_alpha, _ = min(scores, key=lambda x: x[1])
#         return best_alpha

#     def _oof_predict_keep(S_df: pd.DataFrame, B_df: pd.DataFrame, truth_df: pd.DataFrame, ct_eval: list[str], alpha_best: float):
#         cols = B_df.columns.intersection(truth_df.columns)
#         if len(cols) < 2: cols = list(cols)
#         Bc   = B_df[cols]
#         Tref = truth_df.loc[ct_eval, cols]
#         kf = KFold(n_splits=min(n_splits, max(2, len(cols))), shuffle=True, random_state=random_state)
#         oof = pd.DataFrame(index=ct_eval, columns=cols, dtype=float)
#         for tr_idx, va_idx in kf.split(cols):
#             tr_cols = cols[tr_idx]
#             va_cols = cols[va_idx]
#             # 擬合於訓練折，再預測驗證折
#             P_va = _ridge_nnls_matrix(S_df, Bc[va_cols], alpha_best).loc[ct_eval]
#             oof.loc[ct_eval, va_cols] = P_va.values
#         # RMSE（僅 ct_keep）
#         rmse_ct = np.sqrt(((oof - Tref)**2).mean(axis=1))
#         rmse_mean = float(rmse_ct.mean())
#         return oof, rmse_ct, rmse_mean

#     # ---------- 1) 簽名的單細胞 ----------
#     use_ad_for_signature = ad_tr
#     if (celltype_col in ad_te.obs.columns) and (ad_te.obs[celltype_col].notna().any()):
#         use_ad_for_signature = sc.concat([ad_tr, ad_te], join="inner", label="split", keys=["train","test"])

#     # ---------- 2) 簽名（線性空間, CPM） ----------
#     S_all = _build_signature_linear(use_ad_for_signature)

#     mg = pd.Index([g for g in marker_genes if g in S_all.index])
#     if mg.empty:
#         raise ValueError("marker_genes 與 SC 簽名無交集。")
#     S0 = S_all.loc[mg].copy()

#     # ---------- 3) Bulk：CPM + 對齊 ----------
#     B_tr = _cpm(bulk_tr_raw)
#     B_te = _cpm(bulk_te_raw)
#     common = S0.index.intersection(B_tr.index).intersection(B_te.index)
#     if len(common) == 0:
#         raise ValueError("SC 簽名與 train/test bulk 無共同基因。")
#     S0, B_tr, B_te = S0.loc[common], B_tr.loc[common], B_te.loc[common]
#     print(f"[Align] genes={len(common)}, CT(all)={S0.shape[1]}, bulk_train={B_tr.shape[1]}, bulk_test={B_te.shape[1]}")

#     # ---------- 4) CPM+z-score（參考 = S0） + 有限性 ----------
#     S_prep, B_tr_prep = _zscore_to_ref(S0, B_tr)
#     _,      B_te_prep = _zscore_to_ref(S0, B_te)
#     (S_prep, B_tr_prep, B_te_prep), n_ok = _drop_nonfinite(S_prep, B_tr_prep, B_te_prep)
#     if n_ok == 0:
#         raise ValueError("z-score 後無可用基因（非有限值過多）。")
#     if n_ok < len(common):
#         print(f"[Warn] 去除非有限基因：{len(common) - n_ok} / {len(common)}")

#     # ---------- 5) CV α（只在 ct_keep 上打分） ----------
#     ct_all  = list(S_prep.columns)
#     ct_eval = list(pd.Index(ct_keep).intersection(ct_all))
#     if not ct_eval:
#         raise ValueError("ct_keep 與簽名 CT 無交集。")
#     truth_eval = truth_tr.loc[ct_eval]
#     alpha_best = _cv_select_alpha(S_prep, B_tr_prep, truth_eval, ct_eval)
#     print(f"[CV] best alpha = {alpha_best}")

#     # ---------- 6) 產生 OOF 預測（用 alpha_best）並計算 RMSE ----------
#     oof_pred_keep, rmse_ct_keep, rmse_mean_keep = _oof_predict_keep(
#         S_prep, B_tr_prep, truth_eval, ct_eval, alpha_best
#     )
#     print("Per-CT OOF RMSE (ct_keep):")
#     for ct, v in rmse_ct_keep.sort_values().items():
#         print(f"  {ct}: {v:.4f}")
#     print("Mean OOF RMSE (ct_keep):", round(rmse_mean_keep, 4))

#     # ---------- 7) 用 alpha_best 預測 test（輸出 ct_keep） ----------
#     pred_te_all  = _ridge_nnls_matrix(S_prep, B_te_prep, alpha_best)  # all CT × test_samples
#     pred_te_keep = pred_te_all.loc[ct_eval]                            # ct_keep × test_samples

#     return pred_te_keep, oof_pred_keep
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from scipy.optimize import nnls
from sklearn.model_selection import KFold

def deconv_ridge_nnls_cpm_zscore_te(
    ad_tr: sc.AnnData,
    ad_te: sc.AnnData,
    bulk_tr_raw: pd.DataFrame,   # genes × train_samples
    bulk_te_raw: pd.DataFrame,   # genes × test_samples
    truth_tr: pd.DataFrame,      # celltypes × train_samples
    marker_genes: list[str],
    *,
    ct_keep: list[str],                  # 僅用於CV/評估/回傳
    target_sum: float = 1_000_000.0,
    min_cells_per_pair: int = 50,
    downsample_to_n: int | None = None,
    alpha_grid = None,                   # None → np.logspace(-6, 3, 28)
    n_splits: int = 5,
    random_state: int = 42,
    celltype_col: str = "highLevelType",
    sample_col: str = "Sample",
):
    """
    建 S（單細胞簽名，線性空間 CPM）、用 S 對 bulk 做 z-score（gene-wise；參考為 S），
    僅在 ct_keep 上做 CV 選 α。做 OOF 時『同時』對 test 做預測並收集，最後將所有 fold 的
    test 預測取平均後回傳（僅 ct_keep）。

    Returns
    -------
    pred_te_keep   : DataFrame (ct_keep × test_samples)     # test 的平均預測（跨 OOF folds）
    oof_pred_keep  : DataFrame (ct_keep × train_samples)    # OOF 預測（驗證折）
    rmse_ct_keep   : Series   (ct_keep)                     # OOF 每 CT RMSE
    rmse_mean_keep : float                                  # OOF 平均 RMSE
    alpha_best     : float                                  # 以 ct_keep 選到的 α
    """

    # ---------- helpers ----------
    def _to_dense(X):
        return X.toarray() if issparse(X) else np.asarray(X)

    def _cpm(df: pd.DataFrame | None):
        if df is None: return None
        s = df.sum(axis=0).replace(0, 1.0)
        return df.divide(s, axis=1) * 1_000_000

    def _build_signature_linear(ad_: sc.AnnData):
        ad = ad_.copy()
        sc.pp.normalize_total(ad, target_sum=target_sum)
        X = _to_dense(ad.X)
        df = pd.DataFrame(X, index=ad.obs_names, columns=ad.var_names)
        obs = ad.obs[[sample_col, celltype_col]].astype(str)
        df[sample_col]   = obs[sample_col].values
        df[celltype_col] = obs[celltype_col].values

        counts = df.groupby([sample_col, celltype_col]).size()
        keep_pairs = counts[counts >= min_cells_per_pair].index
        key = pd.MultiIndex.from_frame(df[[sample_col, celltype_col]])
        df = df.loc[key.isin(keep_pairs)].copy()

        if downsample_to_n is not None:
            rng = np.random.default_rng(random_state)
            parts = []
            for (s, ct), sub in df.groupby([sample_col, celltype_col], sort=False):
                if len(sub) > downsample_to_n:
                    idx = rng.choice(sub.index.values, size=downsample_to_n, replace=False)
                    parts.append(sub.loc[idx])
                else:
                    parts.append(sub)
            df = pd.concat(parts, axis=0)

        pair_means = df.groupby([sample_col, celltype_col]).mean()
        ct_means   = pair_means.groupby(level=1).mean()
        return ct_means.T  # genes × CT

    def _zscore_to_ref(S_ref: pd.DataFrame, B: pd.DataFrame):
        common = S_ref.index.intersection(B.index)
        S2 = S_ref.loc[common].fillna(0.0).copy()
        B2 = B.loc[common].fillna(0.0).copy()
        mu = S2.mean(axis=1)
        sd = S2.std(axis=1)
        sd = sd.where(sd > 0, 1.0)  # 避免除0
        Sz = (S2.subtract(mu, axis=0)).divide(sd, axis=0)
        Bz = (B2.subtract(mu, axis=0)).divide(sd, axis=0)
        return Sz, Bz

    def _drop_nonfinite(*dfs: pd.DataFrame):
        mask = np.isfinite(dfs[0].values).all(axis=1)
        for D in dfs[1:]:
            mask &= np.isfinite(D.values).all(axis=1)
        return [D.loc[mask].copy() for D in dfs], int(mask.sum())

    def _ridge_nnls_single(S_df: pd.DataFrame, y: pd.Series, alpha: float) -> pd.Series:
        # min ||S w - y||^2 + alpha||w||^2, s.t. w>=0  → NNLS on augmented system
        k = S_df.shape[1]
        A = np.vstack([S_df.values, np.sqrt(alpha) * np.eye(k)])
        b = np.concatenate([y.values, np.zeros(k)])
        w, _ = nnls(A, b)
        s = w.sum()
        if s > 0: w = w / s  # 和=1（soft）
        return pd.Series(w, index=S_df.columns, name=y.name)

    def _ridge_nnls_matrix(S_df: pd.DataFrame, B_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
        return pd.concat([_ridge_nnls_single(S_df, B_df[c], alpha) for c in B_df.columns], axis=1)

    def _cv_select_alpha(S_df: pd.DataFrame, B_df: pd.DataFrame, truth_df: pd.DataFrame, ct_eval: list[str]):
        alphas = np.array(alpha_grid, dtype=float) if alpha_grid is not None else np.logspace(-6, 3, 28)
        cols = B_df.columns.intersection(truth_df.columns)
        if len(cols) < 2: cols = list(cols)
        Bc   = B_df[cols]
        Tref = truth_df.loc[ct_eval, cols]
        kf = KFold(n_splits=min(n_splits, max(2, len(cols))), shuffle=True, random_state=random_state)
        scores = []
        for a in alphas:
            rmses = []
            for _, val_idx in kf.split(cols):
                val_cols = cols[val_idx]
                P = _ridge_nnls_matrix(S_df, Bc[val_cols], a).loc[ct_eval]
                rmses.append(np.sqrt(((P - Tref[val_cols])**2).mean().mean()))
            scores.append((float(a), float(np.mean(rmses))))
        best_alpha, _ = min(scores, key=lambda x: x[1])
        return best_alpha

    # ---------- 1) 簽名的單細胞 ----------
    use_ad_for_signature = ad_tr
    if (celltype_col in ad_te.obs.columns) and (ad_te.obs[celltype_col].notna().any()):
        # test 沒有 celltype 時，不拼接；有的話才拼接
        use_ad_for_signature = sc.concat([ad_tr, ad_te], join="inner", label="split", keys=["train","test"])

    # ---------- 2) 簽名（線性空間, CPM） ----------
    S_all = _build_signature_linear(use_ad_for_signature)

    mg = pd.Index([g for g in marker_genes if g in S_all.index])
    if mg.empty:
        raise ValueError("marker_genes 與 SC 簽名無交集。")
    S0 = S_all.loc[mg].copy()

    # ---------- 3) Bulk：CPM + 對齊 ----------
    B_tr = _cpm(bulk_tr_raw)
    B_te = _cpm(bulk_te_raw)
    common = S0.index.intersection(B_tr.index).intersection(B_te.index)
    if len(common) == 0:
        raise ValueError("SC 簽名與 train/test bulk 無共同基因。")
    S0, B_tr, B_te = S0.loc[common], B_tr.loc[common], B_te.loc[common]
    print(f"[Align] genes={len(common)}, CT(all)={S0.shape[1]}, bulk_train={B_tr.shape[1]}, bulk_test={B_te.shape[1]}")

    # ---------- 4) CPM+z-score（參考 = S0） + 有限性 ----------
    S_prep, B_tr_prep = _zscore_to_ref(S0, B_tr)
    _,      B_te_prep = _zscore_to_ref(S0, B_te)
    (S_prep, B_tr_prep, B_te_prep), n_ok = _drop_nonfinite(S_prep, B_tr_prep, B_te_prep)
    if n_ok == 0:
        raise ValueError("z-score 後無可用基因（非有限值過多）。")
    if n_ok < len(common):
        print(f"[Warn] 去除非有限基因：{len(common) - n_ok} / {len(common)}")

    # ---------- 5) CV α（只在 ct_keep 上打分） ----------
    ct_all  = list(S_prep.columns)
    ct_eval = list(pd.Index(ct_keep).intersection(ct_all))
    if not ct_eval:
        raise ValueError("ct_keep 與簽名 CT 無交集。")
    truth_eval = truth_tr.loc[ct_eval]
    alpha_best = _cv_select_alpha(S_prep, B_tr_prep, truth_eval, ct_eval)
    print(f"[CV] best alpha = {alpha_best}")

    # ---------- 6) OOF 同步預測 test，最後對 test 取平均 ----------
    cols = B_tr_prep.columns.intersection(truth_eval.columns)
    if len(cols) < 2: cols = list(cols)
    Bc   = B_tr_prep[cols]
    Tref = truth_eval.loc[ct_eval, cols]

    kf = KFold(n_splits=min(n_splits, max(2, len(cols))), shuffle=True, random_state=random_state)
    oof = pd.DataFrame(index=ct_eval, columns=cols, dtype=float)

    # 收集每個 fold 的 test 預測（全部 CT，最後再抽 ct_keep）
    test_preds_folds = []

    for tr_idx, va_idx in kf.split(cols):
        # 1) 這個方法每個樣本是一個 NNLS 解，因此不需要「擬合」；直接在驗證折上解
        va_cols = cols[va_idx]
        P_va = _ridge_nnls_matrix(S_prep, Bc[va_cols], alpha_best).loc[ct_eval]
        oof.loc[ct_eval, va_cols] = P_va.values

        # 2) 同一個 fold 直接對 test 做預測並保存
        P_te_fold_all = _ridge_nnls_matrix(S_prep, B_te_prep, alpha_best)  # all CT × test_samples
        test_preds_folds.append(P_te_fold_all)  # 先保留全部 CT，平均時再一起平均

    # OOF RMSE（僅 ct_keep）
    rmse_ct_keep = np.sqrt(((oof - Tref)**2).mean(axis=1))
    rmse_mean_keep = float(rmse_ct_keep.mean())
    print("Per-CT OOF RMSE (ct_keep):")
    for ct, v in rmse_ct_keep.sort_values().items():
        print(f"  {ct}: {v:.4f}")
    print("Mean OOF RMSE (ct_keep):", round(rmse_mean_keep, 4))

    # 對 test 的多折預測做平均（先 all CT，最後再挑 ct_keep）
    if len(test_preds_folds) == 1:
        pred_te_all = test_preds_folds[0]
    else:
        # 對齊所有 fold 的 index/columns（保險起見）
        idx = test_preds_folds[0].index
        cols_te = test_preds_folds[0].columns
        stack = np.stack([tp.reindex(index=idx, columns=cols_te).values for tp in test_preds_folds], axis=0)
        pred_te_all = pd.DataFrame(stack.mean(axis=0), index=idx, columns=cols_te)

    pred_te_keep = pred_te_all.loc[ct_eval]

    return pred_te_keep, oof, rmse_ct_keep, rmse_mean_keep, alpha_best
