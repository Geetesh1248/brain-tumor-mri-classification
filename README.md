# Brain Tumor MRI Classification via Mixture of Experts (MoE)

An end-to-end machine learning research pipeline for classifying brain tumors from MRI scans. This project bridges deep transfer learning with gradient-boosted decision trees (GBDT) and scales into a PyTorch-based Mixture of Experts (MoE) gating architecture.

Conducted as a research initiative at the Maulana Azad National Institute of Technology (MANIT).

## 🧠 Project Architecture

Our pipeline is designed to overcome the computational overhead of end-to-end CNN fine-tuning by leveraging a hybrid extraction-classification approach:

1. **Feature Extraction:** An ImageNet-pretrained ResNet-50 backbone (with the classification head removed) processes MRI scans to extract dense, 2048-dimensional global average pooled embeddings.
2. **Base Classifiers:** Three distinct GBDT models (XGBoost, CatBoost, LightGBM) are trained directly on the ResNet-50 feature space.
3. **Mixture of Experts (MoE):** A PyTorch linear gating network (`nn.Linear(2048, 3)`) evaluates incoming MRI features and dynamically assigns trust weights to the three frozen experts to compute a final, weighted 4-class prediction.

## 📊 Dataset

The dataset consists of 6,597 deduplicated MRI scans across four classes:

* **Glioma:** 1,621 images
* **Healthy:** 2,000 images
* **Meningioma:** 1,645 images
* **Pituitary:** 1,757 images

*Note: To preserve repository performance, the raw `dataset/` images and the 32MB `results/resnet50_features.npz` array file are explicitly `.gitignore`d and must be downloaded from the team's shared drive.*

## 🏆 Model Benchmarks (Test Set: n=990)

| Model Architecture | Accuracy | Macro F1 | Glioma F1 | Healthy F1 | Meningioma F1 | Pituitary F1 |
| --- | --- | --- | --- | --- | --- | --- |
| **ResNet-50 + LightGBM** | **91.92%** | **0.917** | **0.918** | 0.967 | **0.840** | 0.942 |
| **ResNet-50 + MoE Ensemble** | 91.82% | 0.916 | 0.915 | 0.965 | 0.838 | **0.943** |
| **ResNet-50 + XGBoost** | 90.91% | 0.910 | 0.900 | **0.970** | 0.820 | 0.940 |
| **ResNet-50 + CatBoost** | 88.79% | 0.884 | 0.862 | 0.959 | 0.785 | 0.929 |

**Key Findings:**

* **LightGBM** achieved the highest independent accuracy and the fastest training time (78.7s).
* **Healthy tissue** is highly distinct in the 2048-D feature space (F1 $\ge$ 0.959 across all models).
* **Meningioma** represents the hardest clinical edge case, frequently overlapping with glioma and pituitary profiles.
* The **MoE Gating Network** converged rapidly, indicating that the three tree-based experts share highly correlated error distributions on difficult cases.

## 🗂️ Repository Structure

```text
brain-tumor-mri-classification/
├── models/                      # Serialized model weights and JSON metric reports
│   ├── xgboost_resnet50.json
│   ├── lightgbm_resnet50.txt
│   └── catboost_resnet50.cbm
├── notebooks/                   # Interactive exploration and feature extraction
│   ├── 01_Dataset_Exploration.ipynb
│   └── 03_ResNet50_Features.ipynb
├── src/                         # Modular training scripts
│   ├── train_xgboost.py
│   ├── train_catboost.py
│   ├── train_lightgbm.py
│   └── train_moe.py
├── .gitignore                   # Excludes __pycache__/, .venv/, datasets, and .npz/.pt files
└── requirements.txt             # Pinned environment dependencies

```

## 🚀 Quickstart & Setup Instructions

To replicate this environment and run the pipeline locally:

**1. Clone the repository:**

```bash
git clone https://github.com/Geetesh1248/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification

```

**2. Install dependencies:**

```bash
pip install -r requirements.txt

```

**3. Download the Feature Data:**
Create a `results/` directory in the project root. Download `resnet50_features.npz` from the team Google Drive and place it inside the `results/` folder.

**4. Run the Pipeline:**
Execute any of the training scripts from the root directory. They will automatically load the features, train the model, evaluate against the test set, and output the classification report to your terminal.

```bash
python src/train_lightgbm.py
python src/train_moe.py

```
