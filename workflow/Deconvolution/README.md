## Deconvolution (Bulk → Cell-type proportions)

This project performs bulk RNA-seq deconvolution to estimate the proportions of **nine tumor microenvironment (TME) cell types**
(T, B, Endothelial, Fibroblast, Plasmablast, Myofibroblast, NK, Myeloid, Mast).

The pipeline is designed for **low-sample settings** (only 12 training bulk samples) and ranked **3rd overall**, achieving:

- **Test RMSE:** **0.043**
- **Train RMSE (LOPO outer-fold):** **0.046**

Rather than relying on complex models, the method emphasizes **strong preprocessing, careful feature selection, and patient-aware training** with a simple linear regression model.

---

## Inputs

- **`train_adata.h5ad`**  
  Training scRNA-seq reference with cell-type annotations (`highLevelType`), `Sample`, and `Patient`.

- **`train_bulk.csv`, `test_bulk.csv`**  
  Bulk RNA-seq expression matrices (genes × samples).

- **`train_bulk_trueprops.csv`**  
  Ground-truth cell-type proportions for training bulk samples (used only for training/evaluation).

---

## 1. Preprocessing

Preprocessing is applied separately to scRNA-seq data and bulk RNA-seq data, using transformations appropriate to each modality.

### 1a. scRNA-seq preprocessing (reference preparation)

Applied **only to the training scRNA-seq dataset**.

Steps:
- Cell filtering (`min_genes = 200`)
- Gene filtering (`min_cells = 3`)
- Library-size normalization (target sum = `1e4`)
- Log-transformation (`log1p`)

**Rationale**  
scRNA-seq data is highly noisy and affected by sequencing depth and dropout events. Standard preprocessing stabilizes expression values and ensures that downstream marker gene selection captures **biological cell-type differences rather than technical noise**.

---

### 1b. Bulk RNA-seq preprocessing (joint train + test)

Applied **jointly to training and test bulk samples**, without dimensionality reduction.

Steps:
- CPM normalization (CP10k)
- Per-gene clipping (winsorization) at the **99.5th percentile** across train + test
- Variance-stabilizing transformation using `asinh(x / c)` with `c = 1.0`

**Rationale**  
Bulk RNA-seq data often exhibits strong distribution shifts across datasets. Joint preprocessing reduces train–test mismatch and stabilizes the regression problem.

- CP10k corrects for library size differences
- Winsorization limits the influence of extreme outliers
- `asinh` behaves like a log transform for large values while remaining well-defined near zero

---

## 2. Marker Gene Selection (Patient-aware)

Marker gene selection is performed **inside a patient-aware loop** and repeated for each inner training split.

Steps:
- Convert scRNA-seq data into **pseudobulk profiles** by averaging expression per `(Sample, Cell Type)`
- Apply **ComBat** on pseudobulk profiles (batch = Sample)
- Select marker genes using a **stratified per-cell-type strategy**
  - Rank genes by their ability to distinguish each cell type from all others
  - Select the top *K* genes per cell type
  - Take the **union** of selected markers across cell types

**Rationale**  
Bulk deconvolution is fundamentally **feature-driven**. Reliable marker genes are essential for separating cell types in bulk mixtures.

Key design choices:
- Pseudobulk aggregation reduces single-cell noise
- Batch correction prevents sample- or patient-specific artifacts from dominating marker selection
- Stratified selection ensures balanced representation of all cell types
- Performing marker selection inside the patient-aware loop prevents information leakage

> **Marker gene selection is one of the two most critical steps in the pipeline**, together with bulk preprocessing.

---

## 3. Model Training (Patient-aware)

Model training is performed inside a **Leave-One-Patient-Out (LOPO)** outer loop, with an inner K-fold split over remaining patients.

### 3a. Ground-truth proportion transformation

- Cell-type proportions are transformed using **Isometric Log-Ratio (ILR)** transformation
- ILR maps compositional data (non-negative, sum-to-one) into an unconstrained space suitable for regression

A conceptual explanation of ILR is provided here:  
https://medium.com/@nextgendatascientist/a-guide-for-data-scientists-log-ratio-transformations-in-machine-learning-a2db44e2a455

---

### 3b. Regression model

Steps:
- Use **only the selected marker genes** as features
- Train a **linear regression model (Ridge)** to predict ILR-transformed proportions
- Model training is repeated for each inner fold

**Rationale**  
Linear regression is intentionally chosen:
- It is stable with small sample sizes
- It works well with correlated gene expression features
- When paired with strong preprocessing and feature selection, it can outperform more complex models

---

## 4. Prediction and Aggregation

Steps:
- Predict ILR-transformed proportions for held-out samples
- Apply inverse ILR transformation
- Project predictions onto the simplex (non-negative, sum-to-one)
- Aggregate predictions across folds by **averaging in simplex space**

**Rationale**  
This aggregation strategy:
- Reduces variance across folds
- Preserves the compositional constraints of proportions
- Produces stable and robust final predictions

---

## Key takeaways

- This pipeline ranked **3rd overall** using a **simple linear model**
- Performance is driven primarily by:
  1. **Strong preprocessing**
  2. **Careful, patient-aware marker selection**
- With limited data, **methodological rigor matters more than model complexity**

This pipeline demonstrates that, for bulk deconvolution in low-sample settings, a well-designed linear approach can outperform more complex models.
