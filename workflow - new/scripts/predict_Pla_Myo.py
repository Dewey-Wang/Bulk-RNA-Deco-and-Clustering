import numpy as np
import pandas as pd

def deconv_Pla_Myo(
    bulk_tr_raw: pd.DataFrame,
    truth: pd.DataFrame,
    union_unique: list[str],
    bulk_te_raw: pd.DataFrame | None = None,
    alpha: float = 1.0,
    clip0: bool = True,
    ct_keep: tuple[str, str] = ("Plasmablast", "Myofibroblast"),
):
    """
    LOSO Ridge deconv（僅 Ridge，無 z-score），只針對指定 CT（預設 Plasmablast、Myofibroblast）。
    會印出每個 CT 的 RMSE，並回傳 (pred_loso, per_ct_rmse, overall_rmse, pred_test)。

    Parameters
    ----------
    bulk_tr_raw : DataFrame, genes × train_samples（原始 bulk）
    truth       : DataFrame, celltypes × train_samples（真實比例）
    union_unique: list[str]，已選的 marker genes
    bulk_te_raw : DataFrame | None，genes × test_samples（可省略）
    alpha       : float，Ridge 強度
    clip0       : bool，預測裁為 >=0
    normalize_rows : bool，是否把每個樣本預測向量規一化到和=1
    ct_keep     : tuple[str,str]，要預測的細胞型

    Returns
    -------
    pred_loso     : DataFrame, (CT × train_samples) 的 LOSO 預測
    per_ct_rmse   : Series, 各 CT 的 RMSE
    overall_rmse  : float, 全體 RMSE
    pred_test     : DataFrame | None, (CT × test_samples) 的最終模型預測
    """
    # ---- 對齊特徵 / 細胞型 / 樣本 ----
    genes = pd.Index([g for g in union_unique if g in bulk_tr_raw.index])
    if genes.empty:
        raise ValueError("marker_genes 與 bulk 無交集。")
    cts = [ct for ct in ct_keep if ct in truth.index]
    if not cts:
        raise ValueError("指定的細胞型在 truth 中不存在。")

    B_tr_raw = bulk_tr_raw.loc[genes].copy()
    Y_df = truth.loc[cts].copy()

    cols = B_tr_raw.columns.intersection(Y_df.columns)
    if len(cols) < 2:
        raise ValueError("可監督的訓練樣本太少（LOSO 需要 ≥2）。")
    B_tr_raw = B_tr_raw[cols]
    Y_df = Y_df[cols]

    B_te_raw = None
    if isinstance(bulk_te_raw, pd.DataFrame):
        B_te_raw = bulk_te_raw.loc[bulk_te_raw.index.intersection(genes)].copy()

    # ---- CPM(1e6) ----
    def _cpm(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None: return None
        s = df.sum(axis=0).replace(0, 1.0)
        return df.divide(s, axis=1) * 1_000_000

    B_tr = _cpm(B_tr_raw)
    B_te = _cpm(B_te_raw) if B_te_raw is not None else None

    # ---- Ridge ----
    def _fit_ridge(X: np.ndarray, Y: np.ndarray, a: float) -> np.ndarray:
        XT = X.T
        A = XT @ X + a * np.eye(X.shape[1])
        B = XT @ Y
        return np.linalg.solve(A, B)  # p×k

    def _predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
        P = X @ W
        if clip0:
            P = np.maximum(P, 0.0)
        return P

    # ---- LOSO ----
    samples  = list(B_tr.columns)
    ct_names = list(Y_df.index)
    pred_cols = []
    for held in samples:
        tr = [c for c in samples if c != held]
        X_tr = B_tr[tr].T.values          # n_tr × p
        T_tr = Y_df[tr].T.values          # n_tr × k
        W    = _fit_ridge(X_tr, T_tr, alpha)
        X_va = B_tr[[held]].T.values      # 1 × p
        P_va = _predict(X_va, W)          # 1 × k
        pred_cols.append(pd.Series(P_va[0], index=ct_names, name=held))
    pred_loso = pd.DataFrame(pred_cols).T  # ct × sample

    # ---- 評估（印出每個 CT 的 RMSE）----
    T_eval = Y_df.loc[ct_names, samples].values.T
    P_eval = pred_loso.loc[ct_names, samples].values.T
    per_ct_rmse = pd.Series(
        np.sqrt(((P_eval - T_eval) ** 2).mean(axis=0)),
        index=ct_names
    )
    overall_rmse = float(np.sqrt(np.mean((P_eval - T_eval) ** 2)))

    print("[LOSO-RIDGE] per-CT RMSE:")
    for ct, v in per_ct_rmse.sort_values().items():
        print(f"  {ct}: {v:.4f}")
    print(f"[LOSO-RIDGE] overall RMSE = {overall_rmse:.4f}")

    # ---- Test 推論（可選）----
    pred_test = None
    if B_te is not None and B_te.shape[1] > 0:
        genes_order = list(B_tr.index)
        B_te_aligned = B_te.reindex(genes_order).fillna(0.0)
        X_all = B_tr.T.values
        T_all = Y_df.T.values
        W_all = _fit_ridge(X_all, T_all, alpha)
        P_te  = _predict(B_te_aligned.T.values, W_all)
        pred_test = pd.DataFrame(P_te.T, index=ct_names, columns=B_te_aligned.columns)

    return pred_loso, pred_test
