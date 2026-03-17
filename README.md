# ML4G Project 2 — RNA-seq Deconvolution & Cell-type Classification

> Characterize the tumor microenvironment of esophageal adenocarcinoma  
> using joint bulk RNA-seq deconvolution and single-cell RNA-seq clustering.  
> Strong preprocessing, biologically informed features, and leakage-aware validation.

**Result:** Ranked **🥉 3rd** in **ETH Zürich – Machine Learning for Genomics (263-5351-00L, HS2025)** Project 2, supervised by **Prof. Valentina Boeva** (Head TA: **Lovro Rabuzin**).

See the full project description here: [ML4G_Project_2_deconv.pdf](./ML4G_Project_2_deconv.pdf)

---

## 🏆 Final Results & Robustness

- **Final ranking:** 🥉 **3rd place**

**Clustering task**
- **Local (train outer-CV) score:** **0.857**
- **Online (test) score:** **0.855**

**Deconvolution task**
- **Local (train outer-CV) score:** **0.046**
- **Online (test) score:** **0.043**

The **near-identical local and online scores** indicate that the proposed pipeline:
- generalizes well to unseen data
- effectively avoids information leakage
- is robust to distribution shifts between training and test samples
  
---

## High-level Method Summary

The pipeline is organized into **two main components**, each documented in detail in its own folder.

```text
Workflow/
├─► Clustering (representation learning → classification)
└─► Deconvolution (bulk → cell-type proportions)
└─► Prediction (results)
````

---

## 1. Clustering & Cell-type Classification

📁 **`Workflow/Clustering/`**

This component addresses the **single-cell clustering / classification task**.

**Key ideas**

* Learn a **shared, batch-corrected embedding** for train + test cells
* Perform classification in this low-dimensional space using leakage-aware validation

**Method summary**

* Feature construction:

  * Highly variable genes (HVGs)
  * ∪ train-only cell-type marker genes
* Dimension reduction & batch correction:

  * PCA + Harmony (using `Sample` / `Patient`)
* Classification:

  * LightGBM, KNN, Ridge
  * ensemble arbiter to select the most reliable model per cell
* Robustness:

  * Stage 2 is run under **three different CV / model conditions**
  * Final predictions are obtained by **majority vote across the three runs**

👉 See **`Workflow/Clustering/README.md`** for full details.

---

## 2. Bulk RNA-seq Deconvolution

📁 **`Workflow/Deconvolution/`**

This component estimates **cell-type proportions** from bulk RNA-seq samples.

**Key ideas**

* Strong joint preprocessing of bulk train + test data
* Linear regression in a compositional (proportion-aware) space
* Strict patient-aware evaluation

**Method summary**

* Bulk preprocessing:

  * CP10k normalization
  * Winsorization
  * asinh variance stabilization
* Feature construction:

  * Marker genes derived from scRNA-seq
* Model:

  * ILR-transformed Ridge regression
* Evaluation:

  * Leave-One-Patient-Out (LOPO) cross-validation
* Design focus:

  * robustness under **very limited training samples**

👉 See **`Workflow/Deconvolution/README.md`** for full details.


---

## Dataset (Hugging Face)

The full dataset (including both train and test data) is hosted on Hugging Face:

👉 https://huggingface.co/datasets/Dewey0115/ETH_ML4G_Project-2

### Download Options

#### Hugging Face CLI 

```bash
pip install -U huggingface_hub

# download entire dataset
hf download Dewey0115/ETH_ML4G_Project-2 \
  --repo-type dataset \
  --local-dir Dataset/
```

---

## Environment

* Exact versions are pinned in **`./environment.yml`**.

---

## License & Citation

**CC BY-NC 4.0** (non-commercial, attribution required).

Please cite:

> Wang, Ding-Yang. *ML4G Project 2 – Bulk deconvolution and single-cell clustering*. GitHub repository, 2025.

```bibtex
@misc{wang2025ml4g,
  author       = {Wang, Ding-Yang},
  title        = {ML4G Project 2 – Bulk deconvolution and single-cell clustering},
  year         = {2025},
  howpublished = {\url{https://github.com/Dewey-Wang/Bulk-RNA-Deco-and-Clustering.git}},
  note         = {Non-commercial use; citation required}
}
```
