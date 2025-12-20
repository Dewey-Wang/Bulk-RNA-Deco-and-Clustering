## Deconvolution (Bulk → Cell-type proportions)

This section describes the deconvolution pipeline used to estimate the proportions of **nine tumor microenvironment cell types** (T, B, Endothelial, Fibroblast, Plasmablast, Myofibroblast, NK, Myeloid, Mast) from bulk RNA-seq profiles. The core idea is to learn a mapping from **bulk gene expression → cell-type proportions**, while respecting (1) *patient-level separation* and (2) the *compositional nature* of proportions.

### Inputs

The pipeline uses the following files:

* **Training scRNA-seq reference**

  * `train_adata.h5ad`
    Contains single-cell expression (raw counts) and metadata including:

    * `highLevelType` (cell type label)
    * `Sample` (sample ID)
    * `Patient` (patient ID)
    * treatment condition (`chemo`), tumor status, etc.
   This could select the marker genes for the further usage.

* **Bulk expression matrices**

  * `train_bulk.csv` (genes × n_train_bulk_samples)
  * `test_bulk.csv` (genes × n_test_bulk_samples)
    These are the bulk RNA-seq measurements to be deconvolved.

* **Training ground-truth proportions**

  * `train_bulk_trueprops.csv` (cell types × n_train_bulk_samples)
    Used only for training / evaluation, not available for the real test set.

> Note: `test_adata.h5ad` is loaded but not used for marker selection (kept for potential extensions).

---

## Method (step-by-step with rationale)

### 1) Preprocess the training scRNA-seq (reference preparation)

**What we do**

* Filter low-quality cells / genes
* Normalize per cell to a fixed library size
* Log-transform

**Why**
scRNA-seq raw counts are heavily affected by sequencing depth and dropouts. A standard preprocessing (QC → normalize → log1p) makes cell-to-cell expression more comparable and reduces noise, which is essential because downstream **feature selection (marker genes)** depends on stable expression differences across cell types.

---

### 2) Build pseudobulk profiles per (Sample, Cell Type)

**What we do**

* For each `(Sample, highLevelType)` group, compute the **mean expression across cells**
* This yields columns like `Sample|CellType`

**Why**
Marker detection is more reliable on aggregated signals than on individual cells. Pseudobulk reduces single-cell technical noise and makes “cell-type signatures” more robust and interpretable—especially when later used to select markers that generalize across patients.

---

### 3) Correct batch effects in pseudobulk using ComBat (batch = Sample)

**What we do**

* Apply ComBat on the pseudobulk matrix, treating `Sample` as a batch variable.

**Why**
Even within the same study, different samples can differ due to processing time, technician, chemistry, etc. If these technical effects correlate with patient identity, marker selection can accidentally pick “patient markers” instead of “cell-type markers.” ComBat helps reduce this risk by removing sample-specific systematic shifts before marker scoring.

---

### 4) Feature selection (Marker genes) inside CV: stratified per-cell-type union

**What we do**

* For each cell type, compute a discrimination score vs the rest (log-ratio–like)
* Select the **top K markers per cell type**
* Take the **union** of markers across cell types (balanced marker set)

**Why (most important part)**
Bulk deconvolution is fundamentally **feature-driven**: if the chosen genes do not separate cell types well, the regression model cannot recover proportions reliably.

Key design points:

* **Stratified per-cell-type selection** prevents dominant cell types from monopolizing the marker set.
* **Marker selection is performed only using inner-training patients** (see LOPO below), preventing leakage where markers “peek” into the held-out patient.
* Using a union of top markers yields a compact, informative feature space without requiring PCA.

> In practice, **feature selection is one of the two most critical steps** (together with preprocessing). Good markers can make a simple model (Ridge) perform very well.

---

### 5) Joint preprocessing of bulk train + bulk test (NO PCA)

**What we do**

* Normalize each bulk sample to **CP10k** (counts per 10k)
* Apply **winsorization per gene** using a quantile (e.g., 0.995) computed on train+test jointly
* Apply **asinh variance-stabilizing transform**: `asinh(x / c)`
* Do **not** perform PCA

**Why (also a critical step)**
Bulk data often exhibits strong distribution shifts between datasets (or between different sample batches). Joint preprocessing is designed to reduce train–test mismatch and stabilize regression.

* **CP10k** adjusts for library size differences between bulk samples.
* **Winsorization** limits the impact of extreme outliers (often technical artifacts or highly over-dispersed genes) that can dominate a linear model.
* **asinh** behaves like log at high counts while being well-defined near zero; it stabilizes variance while keeping low-expression structure.

> In practice, **bulk preprocessing + marker selection** dominate performance. The model itself is relatively simple.

---

### 6) Compositional regression with ILR (Isometric Log-Ratio) + Ridge

**What we do**

* Cell-type proportions lie on a simplex (non-negative, sum to 1).
* Transform training proportions with **ILR** (Helmert basis).
* Fit a **multi-output Ridge regression** in ILR space.
* Convert predictions back with inverse ILR, then **project onto the simplex**.

**Why**
Direct regression on raw proportions can produce invalid predictions (negative values or sums ≠ 1). ILR provides a principled way to map compositions to Euclidean space where standard regression works, then return to the simplex. Ridge regularization is chosen because:

* it is stable with correlated gene features
* it reduces overfitting when the number of markers is moderate-to-large

---

### 7) Patient-aware evaluation: LOPO + inner K-fold + bagging

**What we do**

* **Outer loop: Leave-One-Patient-Out (LOPO)**
  Hold out all bulk samples from one patient.
* **Inner loop: K-fold over remaining patients**
  Each inner fold:

  * re-select markers using only inner-training patients
  * fit Ridge(ILR)
  * predict held-out samples
* Aggregate fold predictions by averaging **in the simplex** (after inverse ILR and simplex projection)

**Why**
Patients are the true units of generalization. Without LOPO, a model can unintentionally exploit patient-specific patterns and overestimate performance. Nesting marker selection inside the CV loop is especially important—feature selection is a common source of leakage.

Bagging across folds:

* reduces variance due to marker selection instability
* improves robustness across patients
* yields more stable final predictions

---

## What matters most

If you only remember two things about this pipeline:

1. **Preprocessing (especially joint bulk preprocessing)** is crucial to reduce distribution shift and make training consistent with test-time inputs.

2. **Feature selection (marker genes)** is crucial because deconvolution accuracy depends on genes that truly distinguish cell types across patients—not on patient- or batch-specific artifacts.

With strong preprocessing + robust marker selection, even a relatively simple linear model (Ridge) can produce competitive deconvolution performance.
很好，這一段其實**非常值得加上數學與方法定位**，會讓你的 README 從「工程實作」直接升級成「方法論清楚、有研究深度」。
下面我幫你**直接寫好可以無縫接在你現有 README 後面**的章節，用英文、偏 method / report 風格，但仍適合 GitHub。

你可以整段貼上，或只取你想要的部分。

---

## Mathematical Formulation and Method Positioning

### Problem formulation

Bulk RNA-seq deconvolution can be written as a linear mixture model:

[
\mathbf{B} \approx \mathbf{S}\mathbf{P}
]

where:

* (\mathbf{B} \in \mathbb{R}^{G \times N}) is the bulk expression matrix
  ( (G) genes, (N) bulk samples ),
* (\mathbf{S} \in \mathbb{R}^{G \times K}) is the cell-type–specific gene expression matrix
  (cell-type signatures),
* (\mathbf{P} \in \mathbb{R}^{K \times N}) is the matrix of cell-type proportions.

Each column (\mathbf{p}_n) of (\mathbf{P}) is a **composition**:

[
p_{nk} \ge 0, \quad \sum_{k=1}^{K} p_{nk} = 1
]

The goal of deconvolution is to estimate (\mathbf{P}) from (\mathbf{B}), given a reference derived from scRNA-seq.

Rather than explicitly estimating (\mathbf{S}) and solving a constrained optimization problem, this project adopts a **supervised regression approach**:

[
\mathbf{p}_n = f(\mathbf{b}_n)
]

where (f(\cdot)) is learned from bulk training samples with known ground-truth proportions.

---

### Why compositional modeling is necessary

Cell-type proportions live on a **simplex**, not in unconstrained Euclidean space.
Naively regressing proportions directly can lead to:

* negative predicted values,
* proportions that do not sum to 1,
* unstable behavior near the boundaries.

To address this, we use the **Isometric Log-Ratio (ILR) transformation**, which provides a bijection between the simplex and (\mathbb{R}^{K-1}).

---

### ILR transformation and inverse mapping

Given a proportion vector (\mathbf{p} = (p_1, \dots, p_K)), the ILR transform is:

[
\mathbf{z} = \mathrm{ILR}(\mathbf{p}) = \log(\mathbf{p}) \mathbf{H}
]

where (\mathbf{H} \in \mathbb{R}^{K \times (K-1)}) is an orthonormal Helmert basis.

This maps proportions to an unconstrained vector (\mathbf{z} \in \mathbb{R}^{K-1}), allowing the use of standard regression models.

After prediction, we apply the inverse transform:

[
\hat{\mathbf{p}} = \mathrm{ILR}^{-1}(\hat{\mathbf{z}})
]

followed by an explicit **simplex projection**, ensuring:

[
\hat{p}_k \ge 0, \quad \sum_k \hat{p}_k = 1
]

**Why this matters**

* Guarantees mathematically valid proportions
* Avoids ad-hoc post-hoc normalization
* Makes the regression problem well-posed

---

### Linear regression with strong preprocessing

After ILR transformation, the model becomes:

[
\mathbf{z}_n = \mathbf{X}_n \mathbf{W} + \boldsymbol{\epsilon}
]

where:

* (\mathbf{X}_n) is the vector of selected bulk gene expression features,
* (\mathbf{W}) is learned using **Ridge regression**.

Despite being linear, this model performs competitively because:

1. **Aggressive and carefully designed preprocessing** stabilizes distributions and removes technical artifacts.
2. **High-quality marker gene selection** ensures features are biologically informative.
3. Regularization controls overfitting in the low-sample regime.

This aligns with the classical deconvolution view that bulk expression is approximately a linear mixture of cell-type–specific signals.

---

### Why preprocessing and feature selection dominate performance

This method intentionally prioritizes **data preprocessing and feature engineering over model complexity**.

* The training set contains only **12 bulk samples**, making complex nonlinear models prone to overfitting.
* Strong preprocessing (CP10k + joint winsorization + asinh) minimizes train–test distribution shift.
* Marker selection focuses the regression on genes with maximal cell-type contrast.

In practice, these two steps contribute far more to performance than replacing Ridge regression with a more complex model.

---

### Method positioning and performance

This approach ranked **3rd overall** in the deconvolution task.

Key characteristics:

* Simple linear model
* No PCA or deep learning
* Heavy emphasis on preprocessing, compositional modeling, and leakage prevention

Performance:

* **Test set RMSE**: **0.043**
* **Training outer-loop (LOPO) RMSE**: **0.046**

The close agreement between training (outer-fold) and test performance suggests strong generalization and limited overfitting, despite the small training set size.

---

### Takeaway

This project demonstrates that, for bulk RNA-seq deconvolution in low-sample settings:

> **Strong preprocessing + principled compositional modeling + careful feature selection can outperform more complex models.**

A simple linear regressor, when combined with the right statistical and biological constraints, is sufficient to achieve competitive performance.
