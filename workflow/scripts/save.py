import os
import stat
import pandas as pd
from typing import List

EXAMPLE_ROW_ORDER: List[str] = ['T', 'Endothelial', 'Fibroblast', 'Plasmablast', 'B', 'Myofibroblast',
       'NK', 'Myeloid', 'Mast']

def save_props_as_index_wide_fixed_strict(
    df_test_pred: pd.DataFrame,
    out_path: str = "../workflow/outputs/result_final/pred_props.csv"
) -> pd.DataFrame:
    """
    嚴格寫入 out_path（不加時間、不改名）。若已有同名檔案：
      1) 嘗試移除唯讀屬性
      2) 刪除舊檔
      3) 重新寫入
    任一步驟失敗會 raise，方便你修正環境問題。
    """
    # ---- 準備資料 ----
    df = df_test_pred.copy()
    has_idx = all(ct in df.index for ct in EXAMPLE_ROW_ORDER)
    has_col = all(ct in df.columns for ct in EXAMPLE_ROW_ORDER)
    if not has_idx and has_col:
        df = df.T

    keep = [ct for ct in EXAMPLE_ROW_ORDER if ct in df.index]
    others = [ct for ct in df.index if ct not in keep]
    df = df.loc[keep + others]

    out_df = df.reset_index()
    out_df.index = range(len(out_df))
    out_df.index.name = None  # 讓最左索引欄標頭為空 -> ",index,..."

    # ---- 準備目錄 ----
    out_dir = os.path.dirname(out_path) or "."
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # ---- 若舊檔存在：移除唯讀並刪除 ----
    if os.path.exists(out_path):
        try:
            # 嘗試移除唯讀 (Windows 常見)
            try:
                os.chmod(out_path, stat.S_IWRITE | stat.S_IREAD)
            except Exception:
                pass
            os.remove(out_path)
        except Exception as e:
            raise PermissionError(f"無法覆寫既有檔案：{out_path}，請關閉占用程式或調整權限。原始錯誤：{e!r}")

    # ---- 寫入 ----
    try:
        out_df.to_csv(out_path, index=True)
    except Exception as e:
        raise PermissionError(f"寫入失敗：{out_path}。請確認資料夾寫入權限與檔案未被佔用。原始錯誤：{e!r}")

    print(f"[OK] Wrote to: {out_path}")
    return out_df
