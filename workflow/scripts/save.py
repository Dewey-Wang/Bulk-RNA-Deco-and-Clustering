# === Save df_test_pred in ",index,..." format (fixed EXAMPLE_ROW_ORDER) ===
from typing import List
import pandas as pd

EXAMPLE_ROW_ORDER: List[str] = [
    "T","Endothelial","Fibroblast","Plasmablast","B",
    "Myofibroblast","NK","Myeloid","Mast"
]

def save_props_as_index_wide_fixed(
    df_test_pred: pd.DataFrame,
    out_path: str
) -> pd.DataFrame:
    """
    固定以 EXAMPLE_ROW_ORDER 做列順序，輸出為:
    ,index,s5_0,s5_1,...
    0,T,....
    1,Endothelial,...
    ...
    - 若列不是 cell types 但欄是，會自動轉置。
    - 多出的 cell types 會排在固定順序之後。
    """
    df = df_test_pred.copy()

    # why: 僅在欄位含有固定 cell types、列不含時轉置
    has_idx = all(ct in df.index for ct in EXAMPLE_ROW_ORDER)
    has_col = all(ct in df.columns for ct in EXAMPLE_ROW_ORDER)
    if not has_idx and has_col:
        df = df.T

    # 依固定順序排，剩餘的置後
    keep = [ct for ct in EXAMPLE_ROW_ORDER if ct in df.index]
    others = [ct for ct in df.index if ct not in keep]
    df = df.loc[keep + others]

    # 產出 ",index,..." 樣式（保留數字索引做行號）
    out_df = df.reset_index()
    out_df.index = range(len(out_df))
    out_df.index.name = None

    out_df.to_csv(out_path, index=True)
    print(f"[OK] Wrote to: {out_path}")
    return out_df

# 使用範例
# out_path = "outputs/pred_props.csv"
# _preview = save_props_as_index_wide_fixed(df_test_pred, out_path)
# print(_preview.head())
