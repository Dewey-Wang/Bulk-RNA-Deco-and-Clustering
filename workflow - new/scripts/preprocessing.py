import pandas as pd
from IPython.display import display

def report_removed_cells(mask,
                         ad_tr,
                         cols=None,
                         preview_n=10,
                         normalize_pct=True):
    """
    Input:
      mask: True=kept, False=removed
      ad_tr: AnnData
    Behavior:
      - 若 removed_n==0：僅回傳 {"stats":..., "no_remove": True}
      - 否則：輸出預覽與分佈；可回傳各表
    """
    mask = pd.Series(mask, index=ad_tr.obs_names).astype(bool)
    if mask.shape[0] != ad_tr.n_obs:
        raise ValueError("mask length mismatch with ad_tr.n_obs")

    kept_n = int(mask.sum())
    removed_n = int((~mask).sum())
    total_n = int(len(mask))
    stats = {
        "total": total_n,
        "kept": kept_n,
        "removed": removed_n,
        "kept_ratio": kept_n / total_n if total_n else 0.0,
        "removed_ratio": removed_n / total_n if total_n else 0.0,
    }

    # 無移除：只回傳，不顯示
    if removed_n == 0:
        return {"stats": stats, "no_remove": True}

    # 以下為有移除時的處理
    if cols is None:
        cols = ["Sample","Patient","Tumor status","highLevelType","chemo",
                "n_genes_by_counts","total_counts","pct_counts_mt"]
    existing = [c for c in cols if c in ad_tr.obs.columns]
    if not existing:
        raise ValueError("None of the requested columns exist in ad_tr.obs")

    removed_obs = ad_tr.obs.loc[~mask, existing].copy()

    if normalize_pct and "pct_counts_mt" in removed_obs.columns:
        if ad_tr.obs["pct_counts_mt"].astype(float).max() <= 1.0:
            removed_obs["pct_counts_mt"] = removed_obs["pct_counts_mt"].astype(float) * 100.0

    print(stats)
    print("Removed cells preview:")
    display(removed_obs.head(preview_n))

    def _vc(df, col):
        return df[col].value_counts().to_frame("n") if col in df.columns else pd.DataFrame(columns=["n"])

    print("\nCounts by Sample (removed):")
    counts_by_sample = _vc(removed_obs, "Sample"); display(counts_by_sample)

    print("\nCounts by Patient (removed):")
    counts_by_patient = _vc(removed_obs, "Patient"); display(counts_by_patient)

    print("\nCounts by Tumor status (removed):")
    counts_by_tumor = _vc(removed_obs, "Tumor status"); display(counts_by_tumor)

    print("\nCounts by highLevelType (removed):")
    counts_by_hlt = _vc(removed_obs, "highLevelType"); display(counts_by_hlt)

    print("\nCounts by chemo (removed):")
    counts_by_chemo = _vc(removed_obs, "chemo"); display(counts_by_chemo)

    print("\nCrosstab: Tumor status × highLevelType (removed)")
    xtab_tumor_hlt = (pd.crosstab(removed_obs["Tumor status"], removed_obs["highLevelType"])
                      if {"Tumor status","highLevelType"}.issubset(removed_obs.columns) else pd.DataFrame())
    display(xtab_tumor_hlt)

    print("\nCrosstab: Sample × Tumor status (removed)")
    xtab_sample_tumor = (pd.crosstab(removed_obs["Sample"], removed_obs["Tumor status"])
                         if {"Sample","Tumor status"}.issubset(removed_obs.columns) else pd.DataFrame())
    display(xtab_sample_tumor)

    return {
            "stats": stats,
            "no_remove": False,
            "removed_obs": removed_obs,
            "counts_by_sample": counts_by_sample,
            "counts_by_patient": counts_by_patient,
            "counts_by_tumor": counts_by_tumor,
            "counts_by_highLevelType": counts_by_hlt,
            "counts_by_chemo": counts_by_chemo,
            "xtab_tumor_highLevelType": xtab_tumor_hlt,
            "xtab_sample_tumor": xtab_sample_tumor,
        }
