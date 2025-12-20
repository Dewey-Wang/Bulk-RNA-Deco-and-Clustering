
# scRNA-seq Clustering & Classification Pipeline

## Overview & Results

This project presents a **leakage-aware clustering and classification pipeline for scRNA-seq data**, built on **strong preprocessing, biologically informed feature selection, and simple yet robust machine learning models**.

### Final performance
- **Local Train (outer CV) score:** **0.857**
- **Online Test score:** **0.855**
- **Final ranking:** **3rd place**

The evaluation score is defined as:
```

Score = 0.5 × Adjusted Rand Index (ARI) + 0.5 × V-measure (V)

```

Notably, the **top two methods relied on scVI-based batch correction**.

---

## Overall Pipeline (One-glance)

```text
train_adata.h5ad + test_adata.h5ad
        |
        v
┌──────────────────────────────────────────────┐
│ Stage 1: Preprocessing & Representation      │
│  - Gene alignment (train ∩ test)             │
│  - CP10k normalization                       │
│  - HVGs ∪ train-only markers                 │
│  - PCA + Harmony batch correction            │
│                                              │
│  OUTPUT: obsm["X_clust_like_deconv"]          │
└──────────────────────────────────────────────┘
        |
        |  (Stage 1 output is the ONLY input)
        v
┌──────────────────────────────────────────────┐
│ Stage 2: Classification (Ensemble + Arbiter) │
│  - Base models: LGBM / KNN / Ridge            │
│  - Sample-aware CV (no leakage)               │
│  - Arbiter selects best model per cell        │
│                                              │
│  OUTPUT: pred_final (per run)                 │
└──────────────────────────────────────────────┘
        |
        v
┌──────────────────────────────────────────────┐
│ Final Step: Combine 3 Prediction Sets         │
│  - Majority vote across 3 conditions          │
│  - Tie-breaking rules applied                 │
│                                              │
│  FINAL OUTPUT: final cell_type prediction     │
└──────────────────────────────────────────────┘
````

---

## Stage 1 — Preprocessing & Representation Learning

**Goal:**
Construct a **stable, batch-corrected low-dimensional embedding** that can be reused for clustering and classification.

### Inputs

* `train_adata.h5ad` (labeled scRNA-seq)
* `test_adata.h5ad` (unlabeled scRNA-seq)

### Output

* `AnnData.obsm["X_clust_like_deconv"]`
  A shared embedding for **both train and test cells**.

### Key steps (compact)

```text
align genes
 → CP10k normalization
 → select features = HVGs (joint) ∪ markers (train-only)
 → PCA
 → Harmony batch correction (Sample / Patient)
 → concatenate embeddings
```

This embedding is **frozen** and reused in Stage 2.

---

## Stage 2 — Classification (Ensemble + Arbiter)

> **Important:**
> Stage 2 uses **only the embedding produced in Stage 1** as input.
> No raw expression data is used at this stage.

### Input

* Numeric feature matrix derived from:

  * `obsm["X_clust_like_deconv"]`
* Metadata:

  * `cell_type` (labels for training rows; `UNK/NaN` = test)
  * `Sample` (used for leakage-aware CV)

---

### Base models

Each fold trains the following models:

1. **LightGBM**

   * Binary classifier: T vs non-T → produces `p(T)`
   * Multiclass classifier: all cell types
2. **KNN**

   * Cosine distance, multiclass probabilities
3. **RidgeClassifier**

   * Linear multiclass classifier (scores → softmax)

#### Binary T-gating

If `p(T) ≥ threshold (0.55)`, the predicted label is **forced to `"T"`**.
This stabilizes predictions for the most ambiguous / dominant class.

---

### Arbiter (meta-model)

Instead of averaging predictions, an **arbiter** decides **which base model to trust per cell**.

* Inputs:

  * Per-model confidence statistics (top-1 prob, margin, entropy)
  * Agreement indicators between models
  * Binary `p(T)`
* Model:

  * `StandardScaler + Multinomial LogisticRegression`
* Training:

  * Uses **out-of-fold (OOF) predictions only**
  * Completely leakage-free

---

## Three Classification Conditions (Stage 2 is run 3 times)

To improve robustness, Stage 2 is executed under **three different conditions**:

| Run | Base models used   | Cross-validation strategy |
| --: | ------------------ | ------------------------- |
|  #1 | LGBM + KNN + Ridge | Stratified K-Fold         |
|  #2 | LGBM + KNN + Ridge | Standard K-Fold           |
|  #3 | Ridge + LGBM       | GroupKFold by `Sample`    |

Each run produces an independent `pred_final` for all test cells.

---

## Final Step — Combine 3 Prediction Sets (Majority Vote)

The three prediction sets are merged **cell by cell**:

Decision rule:

1. If **≥ 2 predictions agree** → take the majority label
2. If **all three disagree**:

   * Log the cell ID
   * Fall back to **Run #1 prediction**

This final combination further reduces variance and guards against instability from any single CV setup.
