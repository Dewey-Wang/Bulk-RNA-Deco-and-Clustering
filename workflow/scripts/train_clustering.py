import os
from typing import Tuple, Dict, List, Optional, Iterable
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    adjusted_rand_score, v_measure_score
)
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline

import lightgbm as lgb  # base LGBM stays

# ---------------- Config ----------------workflow - new/outputs_features_csv/seperate_chemo/features_post.csv
LABEL_COL = "cell_type"
POS_LABEL = "T"
GROUP_COL = "Sample"
N_SPLITS = 5
SEED = 42

# Threshold
FIXED_THRESHOLD = 0.55
FORCE_T_BY_BINARY = True

# LightGBM (base)
EARLY_STOP_ROUNDS = 50
LGB_PARAMS_MULTI = dict(
    objective="multiclass",
    boosting_type='gbdt',
    learning_rate=0.0208,
    num_leaves=227,
    max_depth=14,
    min_child_samples=35,
    min_child_weight=0.0044,
    min_split_gain=0.3021,
    reg_alpha=0.0,
    reg_lambda=0.0,
    max_bin=418,
    feature_fraction=0.6154,
    bagging_fraction=0.9876,
    bagging_freq=10,
    n_estimators=500,
    # subsample=0.6,
    # colsample_bytree=0.6,
    random_state=SEED,
    n_jobs=-1,
)

LGB_PARAMS_BIN = dict(
    objective="binary",
    boosting_type='gbdt',
    learning_rate=0.0208,
    num_leaves=227,
    max_depth=14,
    min_child_samples=35,
    min_child_weight=0.0044,
    min_split_gain=0.3021,
    reg_alpha=0.0,
    reg_lambda=0.0,
    max_bin=418,
    feature_fraction=0.6154,
    bagging_fraction=0.9876,
    bagging_freq=10,
    n_estimators=500,
    # subsample=0.6,
    # colsample_bytree=0.6,
    random_state=SEED,
    n_jobs=-1,
)

# Arbiter (multinomial Logistic Regression)
ARB_PARAMS_LR = dict(
    penalty="l2", solver="lbfgs", max_iter=2000, random_state=SEED, multi_class="multinomial"
)

# KNN
K_NEIGHBORS = 30
KNN_METRIC = "cosine"
KNN_WEIGHTED_VOTE = False

# RIDGE (new base)
RIDGE_ALPHA = 10.0  # tune if needed

# Imbalance → sample weights
USE_SAMPLE_WEIGHTS = True
WEIGHT_SMOOTH_ALPHA = 0.0
WEIGHT_MAX = 5.0

# ---------------- Utils ----------------
def load_full(path: str) -> pd.DataFrame:
    if not os.path.exists(path): raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0)
    if df.empty: raise ValueError(f"Empty: {path}")
    return df


def split_masks(
    df: pd.DataFrame,
    label_col: str,
    unk_values: tuple[str, ...] = ("UNK", "UNKNOWN")
) -> Tuple[pd.Series, pd.Series]:
    """
    回傳 (is_train, is_test)：
      - train: 有標註且不為 UNK/UNKNOWN
      - test : 其餘（包含 UNK、缺值、空字串、'nan'）
    """
    idx = df.index
    if label_col not in df.columns:
        # 沒有這個欄位 → 全部當 test
        return pd.Series(False, index=idx), pd.Series(True, index=idx)

    s = df[label_col].astype(str).str.strip()
    # 缺標註的條件（NaN / 空字串 / 字串 'nan'）
    is_missing = df[label_col].isna() | s.eq("") | s.str.lower().eq("nan")
    # UNK 類型（大小寫無關）
    unk_set = {u.upper() for u in unk_values}
    is_unk = s.str.upper().isin(unk_set)

    is_train = ~(is_missing | is_unk)
    is_test  = ~is_train
    return is_train, is_test

def numeric_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    Xdf = df.select_dtypes(include=[np.number]).copy()
    keep = (Xdf.nunique(dropna=True) > 1)
    Xdf = Xdf.loc[:, keep]
    if Xdf.shape[1] == 0: raise ValueError("No usable numeric features.")
    return Xdf.fillna(0.0)

def make_sample_weights(y_like: Iterable,
                        smooth_alpha: float = WEIGHT_SMOOTH_ALPHA,
                        max_w: Optional[float] = WEIGHT_MAX) -> np.ndarray:
    ser = pd.Series(list(y_like)); vc = ser.value_counts()
    K = float(vc.shape[0]); N = float(len(ser))
    w_c = (N / (K * (vc + smooth_alpha))).to_dict()
    w = ser.map(w_c).to_numpy(dtype=float)
    return np.minimum(w, float(max_w)) if max_w is not None else w

def metrics_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return dict(
        ACC=accuracy_score(y_true, y_pred),
        Balanced_ACC=balanced_accuracy_score(y_true, y_pred),
        F1_macro=f1_score(y_true, y_pred, average="macro"),
        ARI=adjusted_rand_score(y_true, y_pred),
        V=v_measure_score(y_true, y_pred),
    )

def entropy_row(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())

def softmax_rows(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    a = a - np.max(a, axis=1, keepdims=True)  # stabilize
    e = np.exp(a)
    s = e.sum(axis=1, keepdims=True)
    s = np.clip(s, eps, None)
    return e / s

# ---------------- KNN helpers ----------------
def knn_predict_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    n_classes: int,
    k: int,
    metric: str = "euclidean",
    weighted: bool = True,
) -> np.ndarray:
    k = max(1, min(k, len(X_train)))
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1)
    nbrs.fit(X_train)
    dists, idxs = nbrs.kneighbors(X_query, return_distance=True)

    eps = 1e-12
    proba = np.zeros((X_query.shape[0], n_classes), dtype=float)
    if weighted:
        weights = 1.0 / (dists + eps)
        for i in range(X_query.shape[0]):
            cls = y_train[idxs[i]]; w = weights[i]
            for c, wj in zip(cls, w): proba[i, c] += float(wj)
    else:
        for i in range(X_query.shape[0]):
            cnt = Counter(y_train[idxs[i]])
            for c, v in cnt.items(): proba[i, c] += float(v)
    return softmax_rows(proba)

# ---------------- LGBM two-stage (per fold) ----------------
def train_predict_lgb_two_stage_fold(
    X_tr: np.ndarray, y_tr_str: np.ndarray,
    X_va: np.ndarray, y_va_str: np.ndarray,
    X_test: np.ndarray,
    classes_all: List[str],
    sw_bin_tr: Optional[np.ndarray],
    sw_multi_tr: Optional[np.ndarray],
) -> Dict[str, any]:
    le = LabelEncoder().fit(classes_all)
    y_tr = le.transform(y_tr_str); y_va = le.transform(y_va_str)
    K = len(classes_all); idx_T = int(classes_all.index(POS_LABEL))

    # Binary
    yb_tr = (pd.Series(y_tr_str) == POS_LABEL).astype(int).to_numpy()
    yb_va = (pd.Series(y_va_str) == POS_LABEL).astype(int).to_numpy()
    bin_clf = lgb.LGBMClassifier(**LGB_PARAMS_BIN)
    bin_clf.fit(
        X_tr, yb_tr, sample_weight=sw_bin_tr,
        eval_set=[(X_va, yb_va)], eval_metric="logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOP_ROUNDS, verbose=False)],
    )
    best_it_bin = bin_clf.best_iteration_ or LGB_PARAMS_BIN["n_estimators"]
    p_va_T = bin_clf.predict_proba(X_va, num_iteration=best_it_bin)[:, 1]
    p_test_T = bin_clf.predict_proba(X_test, num_iteration=best_it_bin)[:, 1] if X_test.size else np.array([])

    # Multiclass
    multi_clf = lgb.LGBMClassifier(num_class=K, **LGB_PARAMS_MULTI)
    multi_clf.fit(
        X_tr, y_tr, sample_weight=sw_multi_tr,
        eval_set=[(X_va, y_va)], eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOP_ROUNDS, verbose=False)],
    )
    best_it_multi = multi_clf.best_iteration_ or LGB_PARAMS_MULTI["n_estimators"]
    proba_va_k = multi_clf.predict_proba(X_va, num_iteration=best_it_multi)
    proba_test_k = multi_clf.predict_proba(X_test, num_iteration=best_it_multi) if X_test.size else np.zeros((0, K))

    # Mask on validation
    yhat_va = np.argmax(proba_va_k, axis=1)
    yhat_va[p_va_T >= FIXED_THRESHOLD] = idx_T

    return dict(
        p_va_T=p_va_T, p_test_T=p_test_T,
        proba_va_k=proba_va_k, proba_test_k=proba_test_k,
        yhat_va=yhat_va, best_it_bin=best_it_bin, best_it_multi=best_it_multi
    )

# ---------------- KNN two-stage (ALL classes; mask by binary) ----------------
def train_predict_knn_two_stage_fold(
    X_tr: np.ndarray, y_tr_str: np.ndarray,
    X_va: np.ndarray, y_va_str: np.ndarray,
    X_test: np.ndarray,
    classes_all: List[str],
    sw_bin_tr: Optional[np.ndarray],
) -> Dict[str, any]:
    K = len(classes_all); idx_T = classes_all.index(POS_LABEL)

    # Binary for mask
    yb_tr = (pd.Series(y_tr_str) == POS_LABEL).astype(int).to_numpy()
    yb_va = (pd.Series(y_va_str) == POS_LABEL).astype(int).to_numpy()
    bin_clf = lgb.LGBMClassifier(**LGB_PARAMS_BIN)
    bin_clf.fit(
        X_tr, yb_tr, sample_weight=sw_bin_tr,
        eval_set=[(X_va, yb_va)], eval_metric="logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOP_ROUNDS, verbose=False)],
    )
    best_it_bin = bin_clf.best_iteration_ or LGB_PARAMS_BIN["n_estimators"]
    p_va_T = bin_clf.predict_proba(X_va, num_iteration=best_it_bin)[:, 1]
    p_test_T = bin_clf.predict_proba(X_test, num_iteration=best_it_bin)[:, 1] if X_test.size else np.array([])

    # KNN multiclass on ALL
    le = LabelEncoder().fit(classes_all)
    y_tr = le.transform(y_tr_str)
    proba_va_knn = knn_predict_proba(
        X_tr, y_tr, X_va, n_classes=K, k=K_NEIGHBORS, metric=KNN_METRIC, weighted=KNN_WEIGHTED_VOTE
    )
    proba_test_knn = knn_predict_proba(
        X_tr, y_tr, X_test, n_classes=K, k=K_NEIGHBORS, metric=KNN_METRIC, weighted=KNN_WEIGHTED_VOTE
    ) if X_test.size else np.zeros((0, K))

    # Hard label with mask
    yhat_va = np.argmax(proba_va_knn, axis=1)
    yhat_va[p_va_T >= FIXED_THRESHOLD] = idx_T

    return dict(
        p_va_T_knn=p_va_T, p_test_T_knn=p_test_T,
        proba_va_full_knn=proba_va_knn, proba_test_full_knn=proba_test_knn,
        yhat_va_knn=yhat_va
    )

# ---------------- RIDGE two-stage (ALL classes; mask by binary) ----------------
def train_predict_ridge_two_stage_fold(
    X_tr: np.ndarray, y_tr_str: np.ndarray,
    X_va: np.ndarray, y_va_str: np.ndarray,
    X_test: np.ndarray,
    classes_all: List[str],
    sw_bin_tr: Optional[np.ndarray],
    sw_multi_tr: Optional[np.ndarray],
) -> Dict[str, any]:
    """Multiclass via RidgeClassifier(decision_function→softmax), masked by LGBM-binary."""
    K = len(classes_all); idx_T = classes_all.index(POS_LABEL)

    # Binary for mask (same as others)
    yb_tr = (pd.Series(y_tr_str) == POS_LABEL).astype(int).to_numpy()
    yb_va = (pd.Series(y_va_str) == POS_LABEL).astype(int).to_numpy()
    bin_clf = lgb.LGBMClassifier(**LGB_PARAMS_BIN)
    bin_clf.fit(
        X_tr, yb_tr, sample_weight=sw_bin_tr,
        eval_set=[(X_va, yb_va)], eval_metric="logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOP_ROUNDS, verbose=False)],
    )
    best_it_bin = bin_clf.best_iteration_ or LGB_PARAMS_BIN["n_estimators"]
    p_va_T = bin_clf.predict_proba(X_va, num_iteration=best_it_bin)[:, 1]
    p_test_T = bin_clf.predict_proba(X_test, num_iteration=best_it_bin)[:, 1] if X_test.size else np.array([])

    # Multiclass by RidgeClassifier (scores→softmax)
    le = LabelEncoder().fit(classes_all)
    y_tr = le.transform(y_tr_str)
    ridge = RidgeClassifier(alpha=RIDGE_ALPHA)
    ridge.fit(X_tr, y_tr, sample_weight=sw_multi_tr)
    scores_va = ridge.decision_function(X_va)  # (n_va, K)
    proba_va = softmax_rows(scores_va)
    scores_te = ridge.decision_function(X_test) if X_test.size else np.zeros((0, K))
    proba_te = softmax_rows(scores_te) if X_test.size else np.zeros((0, K))

    # Mask on hard labels
    yhat_va = np.argmax(proba_va, axis=1)
    yhat_va[p_va_T >= FIXED_THRESHOLD] = idx_T

    return dict(
        p_va_T_ridge=p_va_T, p_test_T_ridge=p_test_T,
        proba_va_full_ridge=proba_va, proba_test_full_ridge=proba_te,
        yhat_va_ridge=yhat_va
    )

# ---------------- Meta feature helpers ----------------
def features_from_prob(proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    top1 = np.max(proba, axis=1)
    sorted_p = -np.sort(-proba, axis=1)
    margin = sorted_p[:, 0] - sorted_p[:, 1]
    entropy = np.array([entropy_row(p) for p in proba])
    return top1, margin, entropy

def build_arbiter_meta3(
    proba_lgb: np.ndarray, proba_knn: np.ndarray, proba_ridge: np.ndarray, pT: np.ndarray
) -> np.ndarray:
    l_top1, l_margin, l_ent = features_from_prob(proba_lgb)
    k_top1, k_margin, k_ent = features_from_prob(proba_knn)
    r_top1, r_margin, r_ent = features_from_prob(proba_ridge)
    agree_lk = (np.argmax(proba_lgb, axis=1) == np.argmax(proba_knn, axis=1)).astype(int)
    agree_lr = (np.argmax(proba_lgb, axis=1) == np.argmax(proba_ridge, axis=1)).astype(int)
    agree_kr = (np.argmax(proba_knn, axis=1) == np.argmax(proba_ridge, axis=1)).astype(int)
    return np.column_stack([
        pT,
        l_top1, l_margin, l_ent,
        k_top1, k_margin, k_ent,
        r_top1, r_margin, r_ent,
        agree_lk, agree_lr, agree_kr
    ])

def build_target_meta3(
    y_true_enc: np.ndarray,
    yhat_lgb: np.ndarray, l_margin: np.ndarray,
    yhat_knn: np.ndarray, k_margin: np.ndarray,
    yhat_rdg: np.ndarray, r_margin: np.ndarray,
) -> np.ndarray:
    l_ok = (yhat_lgb == y_true_enc).astype(int)
    k_ok = (yhat_knn == y_true_enc).astype(int)
    r_ok = (yhat_rdg == y_true_enc).astype(int)
    margins = np.column_stack([l_margin, k_margin, r_margin])     # (n,3)
    correct = np.column_stack([l_ok, k_ok, r_ok])                 # (n,3)
    # pick among correct by max margin; if none correct, pick global max margin
    pick_correct = np.argmax(np.where(correct == 1, margins, -np.inf), axis=1)
    has_correct = (correct.sum(axis=1) > 0)
    pick_global = np.argmax(margins, axis=1)
    return np.where(has_correct, pick_correct, pick_global).astype(int)  # 0:LGBM,1:KNN,2:RIDGE
# ---- NEW generic helpers: build meta from variable base models ----
def _build_meta_generic(proba_dict: dict, pT: np.ndarray) -> np.ndarray:
    """
    proba_dict: {name -> (n, K) ndarray}，只會用到你真的有跑的模型
    X_meta: [pT,  每個模型的(top1, margin, entropy),   所有成對的hard-agreement]
    """
    def _triples(P):
        top1 = np.max(P, axis=1)
        sorted_p = -np.sort(-P, axis=1)
        margin = sorted_p[:, 0] - sorted_p[:, 1]
        ent = np.array([entropy_row(p) for p in P])
        return top1, margin, ent

    names = list(proba_dict.keys())
    feats = [pT.reshape(-1,1)]
    margins = []
    hard = {}
    for nm in names:
        t, m, e = _triples(proba_dict[nm]); margins.append(m)
        feats += [t.reshape(-1,1), m.reshape(-1,1), e.reshape(-1,1)]
        hard[nm] = np.argmax(proba_dict[nm], axis=1)
    # pairwise agreement（有兩個以上模型才加）
    if len(names) >= 2:
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                agree = (hard[names[i]] == hard[names[j]]).astype(int).reshape(-1,1)
                feats.append(agree)
    X_meta = np.column_stack(feats)
    return X_meta

def _build_target_generic(y_true_enc: np.ndarray, yhat_list: list[np.ndarray], margin_list: list[np.ndarray]) -> np.ndarray:
    """
    yhat_list: [yhat_model0, yhat_model1, ...]
    margin_list: [margin_model0, margin_model1, ...]（跟 meta 同順序）
    回傳：每列要選哪個模型（0..M-1）
    """
    M = len(yhat_list)
    n = y_true_enc.shape[0]
    margins = np.column_stack(margin_list)          # (n, M)
    correct = np.column_stack([(yh == y_true_enc).astype(int) for yh in yhat_list])  # (n, M)
    has_correct = (correct.sum(axis=1) > 0)
    # 先把錯的設為 -inf，只在正確者裡挑最大 margin
    pick_correct = np.argmax(np.where(correct==1, margins, -np.inf), axis=1)
    pick_global  = np.argmax(margins, axis=1)
    return np.where(has_correct, pick_correct, pick_global).astype(int)

# ---------------- Main orchestration ----------------
from typing import Tuple, Dict, List, Optional, Iterable
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def run_ensemble_arbiter(
    df: pd.DataFrame,
    fname: str,
    *,
    base_models: Tuple[str, ...] = ("lgbm","knn","ridge"),
    cv_mode: str = "group",            # "group" | "stratified" | "kfold"
    cv_shuffle: bool = True,           # 只影響 random 類型
    cv_random_state: int = 42,         # 只影響 random 類型
) -> pd.DataFrame:
    # ---- 前置與切分（沿用你原本的） ----
    is_lab, is_unlab = split_masks(df, LABEL_COL)
    if not is_lab.any(): raise ValueError(f"{fname}: no labeled rows.")
    if GROUP_COL not in df.columns: raise ValueError(f"{fname}: required group column '{GROUP_COL}' not found.")

    X_all   = numeric_feature_matrix(df).to_numpy(dtype=float)
    y_all   = df[LABEL_COL].astype(str).values
    idx_all = df.index.to_numpy()

    X_lab, y_lab_str, idx_lab = X_all[is_lab.values], y_all[is_lab.values], idx_all[is_lab.values]
    X_test, idx_test          = X_all[is_unlab.values], idx_all[is_unlab.values]

    groups_series = df.loc[idx_lab, GROUP_COL]
    if groups_series.isna().all(): raise ValueError(f"{fname}: '{GROUP_COL}' all NaN for labeled rows.")
    groups = groups_series.fillna("NA").astype(str).values

    le_all = LabelEncoder().fit(y_lab_str)
    classes_all = le_all.classes_.tolist()
    if POS_LABEL not in classes_all: raise ValueError(f"{fname}: POS_LABEL '{POS_LABEL}' not in labels {classes_all}")
    K = len(classes_all); idx_T = classes_all.index(POS_LABEL)

    # ==== 建立交叉驗證折 ====
    cv_mode = cv_mode.lower()
    if cv_mode not in {"group", "stratified", "kfold"}:
        raise ValueError("cv_mode 必須是 'group' / 'stratified' / 'kfold'")

    if cv_mode == "group":
        n_groups = pd.Series(groups).nunique()
        n_splits = min(N_SPLITS, n_groups)
        if n_splits < 2: raise ValueError(f"{fname}: need ≥2 groups for GroupKFold, got {n_groups}.")
        splitter = GroupKFold(n_splits=n_splits)
        folds = list(splitter.split(X_lab, y_lab_str, groups))
        print(f"{fname}: GroupKFold ({n_splits} folds) on {n_groups} groups.")
    elif cv_mode == "stratified":
        # StratifiedKFold 要求每個類別樣本數 ≥ n_splits；自動調整到可行
        vc = pd.Series(y_lab_str).value_counts()
        max_folds_by_class = int(vc.min())
        n_splits = max(2, min(N_SPLITS, max_folds_by_class))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=cv_shuffle, random_state=cv_random_state)
        folds = list(splitter.split(X_lab, y_lab_str))
        print(f"{fname}: StratifiedKFold ({n_splits} folds, shuffle={cv_shuffle}, seed={cv_random_state}).")
    else:  # "kfold"
        n_splits = max(2, min(N_SPLITS, len(y_lab_str)))
        splitter = KFold(n_splits=n_splits, shuffle=cv_shuffle, random_state=cv_random_state)
        folds = list(splitter.split(X_lab))
        print(f"{fname}: KFold ({n_splits} folds, shuffle={cv_shuffle}, seed={cv_random_state}).")

    # 權重
    y_bin_lbl = np.where(pd.Series(y_lab_str) == POS_LABEL, POS_LABEL, "nonT")
    sw_bin_all   = make_sample_weights(y_bin_lbl) if USE_SAMPLE_WEIGHTS else np.ones(len(y_lab_str), dtype=float)
    sw_multi_all = make_sample_weights(y_lab_str) if USE_SAMPLE_WEIGHTS else np.ones(len(y_lab_str), dtype=float)

    # ---- 依「選到的模型」動態開 buffer ----
    valid_names = {"lgbm","knn","ridge"}
    chosen = [m.lower() for m in base_models]
    for nm in chosen:
        if nm not in valid_names:
            raise ValueError(f"Unknown base model '{nm}', must be in {sorted(valid_names)}")
    if len(chosen) == 0:
        raise ValueError("base_models 不能為空")

    oof_pred = {nm: np.full(len(y_lab_str), -1, dtype=int) for nm in chosen}
    oof_prob = {nm: np.zeros((len(y_lab_str), K), dtype=float) for nm in chosen}
    oof_pT   = np.zeros(len(y_lab_str), dtype=float)  # 仍用 LGBM-binary 的 p(T)

    test_fold_pT   = []
    test_fold_prob = {nm: [] for nm in chosen}

    rows = []
    rows_ens = []

    # ---- outer folds ----
    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        X_tr, X_va = X_lab[tr_idx], X_lab[va_idx]
        y_tr_str_f, y_va_str_f = y_lab_str[tr_idx], y_lab_str[va_idx]

        # 收集 meta
        fold_yhat, fold_margin = [], []
        fold_prob_for_meta = {}

        # 1) LGBM
        if "lgbm" in chosen:
            lgb_res = train_predict_lgb_two_stage_fold(
                X_tr, y_tr_str_f, X_va, y_va_str_f, X_test, classes_all,
                sw_bin_tr=sw_bin_all[tr_idx] if USE_SAMPLE_WEIGHTS else None,
                sw_multi_tr=sw_multi_all[tr_idx] if USE_SAMPLE_WEIGHTS else None,
            )
            p_va_T_lgb   = lgb_res["p_va_T"]
            proba_va_lgb = lgb_res["proba_va_k"]
            yhat_va_lgb  = lgb_res["yhat_va"]

            oof_pred["lgbm"][va_idx] = yhat_va_lgb
            oof_prob["lgbm"][va_idx] = proba_va_lgb
            oof_pT[va_idx] = p_va_T_lgb

            sorted_p = -np.sort(-proba_va_lgb, axis=1)
            fold_margin.append(sorted_p[:,0]-sorted_p[:,1])
            fold_yhat.append(yhat_va_lgb)
            fold_prob_for_meta["lgbm"] = proba_va_lgb

            if X_test.size:
                test_fold_pT.append(lgb_res["p_test_T"])
                test_fold_prob["lgbm"].append(lgb_res["proba_test_k"])

            y_true_enc = le_all.transform(y_va_str_f)
            m = metrics_all(y_true_enc, yhat_va_lgb); m["Score"] = 0.5*(m["ARI"]+m["V"])
            rows.append({"fold": fold, "model": "LGBM", **m})

        # 2) KNN
        if "knn" in chosen:
            knn_res = train_predict_knn_two_stage_fold(
                X_tr, y_tr_str_f, X_va, y_va_str_f, X_test, classes_all,
                sw_bin_tr=sw_bin_all[tr_idx] if USE_SAMPLE_WEIGHTS else None,
            )
            proba_va_knn = knn_res["proba_va_full_knn"]
            yhat_va_knn  = knn_res["yhat_va_knn"]

            oof_pred["knn"][va_idx] = yhat_va_knn
            oof_prob["knn"][va_idx] = proba_va_knn

            sorted_p = -np.sort(-proba_va_knn, axis=1)
            fold_margin.append(sorted_p[:,0]-sorted_p[:,1])
            fold_yhat.append(yhat_va_knn)
            fold_prob_for_meta["knn"] = proba_va_knn

            if X_test.size:
                test_fold_prob["knn"].append(knn_res["proba_test_full_knn"])

            y_true_enc = le_all.transform(y_va_str_f)
            m = metrics_all(y_true_enc, yhat_va_knn); m["Score"] = 0.5*(m["ARI"]+m["V"])
            rows.append({"fold": fold, "model": "KNN", **m})

        # 3) RIDGE
        if "ridge" in chosen:
            rdg_res = train_predict_ridge_two_stage_fold(
                X_tr, y_tr_str_f, X_va, y_va_str_f, X_test, classes_all,
                sw_bin_tr=sw_bin_all[tr_idx] if USE_SAMPLE_WEIGHTS else None,
                sw_multi_tr=sw_multi_all[tr_idx] if USE_SAMPLE_WEIGHTS else None,
            )
            proba_va_rdg = rdg_res["proba_va_full_ridge"]
            yhat_va_rdg  = rdg_res["yhat_va_ridge"]

            oof_pred["ridge"][va_idx] = yhat_va_rdg
            oof_prob["ridge"][va_idx] = proba_va_rdg

            sorted_p = -np.sort(-proba_va_rdg, axis=1)
            fold_margin.append(sorted_p[:,0]-sorted_p[:,1])
            fold_yhat.append(yhat_va_rdg)
            fold_prob_for_meta["ridge"] = proba_va_rdg

            if X_test.size:
                test_fold_prob["ridge"].append(rdg_res["proba_test_full_ridge"])

            y_true_enc = le_all.transform(y_va_str_f)
            m = metrics_all(y_true_enc, yhat_va_rdg); m["Score"] = 0.5*(m["ARI"]+m["V"])
            rows.append({"fold": fold, "model": "RIDGE", **m})

        # ---- Ensemble OOF ----
        y_true_enc = le_all.transform(y_va_str_f)

        # 這一折的 OOF meta（當前折的各模型 OOF 概率）
        X_meta_va = _build_meta_generic(
            {nm: oof_prob[nm][va_idx] for nm in chosen},
            oof_pT[va_idx]
        )
        y_meta_va = _build_target_generic(y_true_enc, fold_yhat, fold_margin)

        if len(chosen) >= 2:
            # --- 準備仲裁器訓練資料（不洩漏當前 val）---
            X_meta_tr = _build_meta_generic({nm: oof_prob[nm][tr_idx] for nm in chosen}, oof_pT[tr_idx])
            y_true_tr = le_all.transform(y_lab_str[tr_idx])

            # 用訓練折的 OOF hard label + margin 組仲裁器目標
            yhat_tr_list = [oof_pred[nm][tr_idx] for nm in chosen]
            margin_tr_list = []
            for nm in chosen:
                P = oof_prob[nm][tr_idx]
                sorted_p = -np.sort(-P, axis=1)
                margin_tr_list.append(sorted_p[:, 0] - sorted_p[:, 1])
            y_meta_tr = _build_target_generic(y_true_tr, yhat_tr_list, margin_tr_list)

            # --- 目標只有單一類別時的 fallback ---
            uniq = np.unique(y_meta_tr)
            if uniq.size >= 2:
                arb = Pipeline([("scaler", StandardScaler()),
                                ("lr", LogisticRegression(**ARB_PARAMS_LR))])
                arb.fit(X_meta_tr, y_meta_tr)
                choice = arb.predict(X_meta_va)  # 0..len(chosen)-1
            else:
                # 無法訓練 LR → 直接用「validation 端各基模的 margin 最大者」做仲裁
                # fold_margin 是你上面已經用 val 的機率算好的 list，順序對應 chosen
                margin_val_mat = np.column_stack(fold_margin)  # (n_val, M)
                choice = np.argmax(margin_val_mat, axis=1)

            base_stack = np.vstack([oof_pred[nm][va_idx] for nm in chosen]).T
            ens_oof = base_stack[np.arange(base_stack.shape[0]), choice]
            m = metrics_all(y_true_enc, ens_oof); m["Score"] = 0.5*(m["ARI"] + m["V"])
        else:
            # 只有一個基模，仲裁器退化為直接用它
            ens_oof = oof_pred[chosen[0]][va_idx]
            m = metrics_all(y_true_enc, ens_oof); m["Score"] = 0.5*(m["ARI"] + m["V"])


        # 列出本折的 train/val 的 Sample（unique）
        train_samples = ",".join(pd.unique(df.loc[idx_lab[tr_idx], GROUP_COL].astype(str)))
        val_samples   = ",".join(pd.unique(df.loc[idx_lab[va_idx], GROUP_COL].astype(str)))

        rows_ens.append({
            "fold": fold,
            "model": "ENSEMBLE",
            "ARI": m["ARI"], "V": m["V"], "Score": m["Score"],
            "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
            "train_samples": train_samples,
            "val_samples": val_samples,
            "cv_mode": cv_mode,
        })

    # ---- CV 彙總 ----
    df_cv = pd.DataFrame(rows + rows_ens)

    print("\n[CV metrics per fold — Score]")
    try:
        tbl = df_cv.pivot(index="fold", columns="model", values="Score")
        print(tbl.to_string())
    except Exception:
        print(df_cv[["fold","model","Score"]].to_string(index=False))

    if len(rows_ens) > 0:
        df_folds = pd.DataFrame(rows_ens)[
            ["fold", "cv_mode", "n_train", "n_val", "train_samples", "val_samples"]
        ].sort_values("fold")
        print("\n[Per-fold train/val samples]")
        for _, r in df_folds.iterrows():
            f = int(r["fold"])
            print(f"fold {f} [{r['cv_mode']}]: n_train={int(r['n_train'])}, n_val={int(r['n_val'])}")
            print(f"  train_samples: {r['train_samples']}")
            print(f"  val_samples  : {r['val_samples']}")

    def _summ(name: str) -> Dict[str, float]:
        sub = df_cv[df_cv["model"] == name]
        return dict(ARI=sub["ARI"].mean(), V=sub["V"].mean(), Score=sub["Score"].mean()) if len(sub) else dict(ARI=np.nan, V=np.nan, Score=np.nan)

    for tag in (["LGBM"] if "lgbm" in chosen else []) + (["KNN"] if "knn" in chosen else []) + (["RIDGE"] if "ridge" in chosen else []) + ["ENSEMBLE"]:
        s = _summ(tag)
        print(f"{tag:<8}: ARI={s['ARI']:.6f}  V={s['V']:.6f}  Score={s['Score']:.6f}")

    # ---- 最終仲裁 + 測試 ----
    if X_test.size and any(len(v)>0 for v in test_fold_prob.values()):
        proba_test_mean = {}
        for nm in chosen:
            if len(test_fold_prob[nm]) > 0:
                proba_test_mean[nm] = np.mean(np.stack(test_fold_prob[nm], axis=0), axis=0)
            else:
                proba_test_mean[nm] = np.zeros((len(idx_test), K))
        pT_mean = np.mean(np.vstack(test_fold_pT), axis=0) if len(test_fold_pT)>0 else np.zeros(len(idx_test))

        X_meta_test = _build_meta_generic(proba_test_mean, pT_mean)

        if len(chosen) >= 2:
            X_meta_oof_all = _build_meta_generic({nm: oof_prob[nm] for nm in chosen}, oof_pT)
            y_true_enc_all = le_all.transform(y_lab_str)
            yhat_all_list   = [oof_pred[nm] for nm in chosen]
            margin_all_list = []
            for nm in chosen:
                P = oof_prob[nm]
                sorted_p = -np.sort(-P, axis=1)
                margin_all_list.append(sorted_p[:,0]-sorted_p[:,1])
            y_meta_all = _build_target_generic(y_true_enc_all, yhat_all_list, margin_all_list)

            arb_final = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(**ARB_PARAMS_LR))])
            arb_final.fit(X_meta_oof_all, y_meta_all)
            choice_test = arb_final.predict(X_meta_test)

            base_stack = np.vstack([np.argmax(proba_test_mean[nm], axis=1) for nm in chosen]).T
            pred_final_enc = base_stack[np.arange(base_stack.shape[0]), choice_test]
        else:
            only = chosen[0]
            pred_final_enc = np.argmax(proba_test_mean[only], axis=1)

        if FORCE_T_BY_BINARY and len(test_fold_pT)>0:
            pred_final_enc[pT_mean >= FIXED_THRESHOLD] = idx_T

        out = pd.DataFrame({"cell_id": idx_test, "pred_final": le_all.inverse_transform(pred_final_enc)})
        for nm in chosen:
            for i, cls in enumerate(classes_all):
                out[f"proba_{nm}_{cls}"] = proba_test_mean[nm][:, i]
        if len(test_fold_pT)>0:
            out["pT_mean"] = pT_mean
            out["thr_used"] = FIXED_THRESHOLD

        print("\n[Test prediction distribution (%), ENSEMBLE]")
        print(out["pred_final"].value_counts(normalize=True).sort_index().mul(100).round(2).astype(str) + "%")
    else:
        print("\n[Info] No unlabeled rows to predict.")
        out = pd.DataFrame(columns=["cell_id", "pred_final"])

    return out

def combine_pred_triplet(pred_1: pd.DataFrame,
                         pred_2: pd.DataFrame,
                         pred_3: pd.DataFrame,
                         *,
                         id_col: str = "cell_id",
                         col_name: str = "pred_final") -> pd.DataFrame:
    """
    多數決融合三份預測：
      - 先以 id_col 合併出 pred_1 / pred_2 / pred_3 三欄
      - 若三者有兩者以上相同 -> 採多數
      - 若三者全不同 -> 印出 cell_id，並採 pred_1
    回傳：含 pred_1/pred_2/pred_3/final_pred 的 DataFrame（以 cell_id 排序）
    """
    # 準備三個乾淨的 (cell_id, predX) 表
    def _clean(df, tag):
        d = df[[id_col, col_name]].copy()
        d = d.rename(columns={col_name: f"pred_{tag}"})
        return d

    p1 = _clean(pred_1, "1")
    p2 = _clean(pred_2, "2")
    p3 = _clean(pred_3, "3")

    # 外連接，確保不遺漏任何 cell
    merged = p1.merge(p2, on=id_col, how="outer").merge(p3, on=id_col, how="outer")

    # 定義逐列決策
    def _decide(row):
        a, b, c = row["pred_1"], row["pred_2"], row["pred_3"]
        vals = [a, b, c]
        # 去掉缺失
        vals_non_null = [v for v in vals if pd.notna(v)]
        if len(vals_non_null) == 0:
            return pd.NA
        # 多數決：若有兩票以上相同
        vc = pd.Series(vals_non_null).value_counts()
        if vc.iloc[0] >= 2:
            if vc.iloc[0] == 2:
                print(f"[Tie 2-different] cell_id={row[id_col]} | "
                f"pred_1={a}, pred_2={b}, pred_3={c} -> use { vc.index[0]}")
            return vc.index[0]
        # 三者皆不同 or 只有一票各異
        # 印出 cell_id，並採用 pred_1
        print(f"[Tie 3-different] cell_id={row[id_col]} | "
              f"pred_1={a}, pred_2={b}, pred_3={c} -> use pred_1")
        return a

    merged["final_pred"] = merged.apply(_decide, axis=1)

    # 依 id 排序（可選）
    try:
        merged = merged.sort_values(by=id_col)
    except Exception:
        pass

    return merged