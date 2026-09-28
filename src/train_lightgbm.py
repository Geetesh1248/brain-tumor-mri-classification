"""
LightGBM classifier trained on ResNet-50 extracted features.

Loads pre-extracted features from results/resnet50_features.npz,
trains on the training split with validation monitoring, evaluates
on the test split, and exports a metrics JSON.
"""

import json
import time
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── Config ─────────────────────────────────────────────────────────────────────
FEATURES_PATH = Path("results/resnet50_features.npz")
METRICS_PATH  = Path("models/lightgbm_metrics.json")
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

CLASS_NAMES  = {0: "glioma", 1: "healthy", 2: "meningioma", 3: "pituitary"}
LABEL_NAMES  = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]

# ── 1. Load features ───────────────────────────────────────────────────────────
print(f"Loading features from '{FEATURES_PATH}' ...")
data = np.load(FEATURES_PATH)

X_train, y_train = data["X_train"], data["y_train"]
X_val,   y_val   = data["X_val"],   data["y_val"]
X_test,  y_test  = data["X_test"],  data["y_test"]

print(f"  Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")
print(f"  Classes: {CLASS_NAMES}\n")

# ── 2. Initialise LGBMClassifier ───────────────────────────────────────────────
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.1,
    num_leaves=31,
    objective="multiclass",
    num_class=4,
    random_state=42,
    n_jobs=-1,
    verbose=-1,           # suppress LightGBM internal logs (use callbacks)
)

# ── 3. Train with validation monitoring ────────────────────────────────────────
print("Training LGBMClassifier ...")
t0 = time.time()
model.fit(
    X_train, y_train,
    eval_X=X_val,
    eval_y=y_val,
    eval_metric="multi_logloss",
    callbacks=[
        log_evaluation(period=50),   # print every 50 rounds
    ],
)
elapsed = time.time() - t0
best_iter = model.best_iteration_ if model.best_iteration_ > 0 else 299
print(f"\nTraining complete in {elapsed:.1f}s  |  Best iteration: {best_iter}\n")

# ── 4. Evaluate on test set ────────────────────────────────────────────────────
y_pred = model.predict(X_test).astype(int)

acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)

print("=" * 60)
print(f"  Test Accuracy : {acc * 100:.2f}%")
print("=" * 60)

# Pretty confusion matrix
col_w = 13
print(f"\nConfusion Matrix (rows = true, cols = predicted):")
header = " " * 14 + "".join(f"{n:>{col_w}}" for n in LABEL_NAMES)
print(header)
print(" " * 14 + "-" * (col_w * len(LABEL_NAMES)))
for i, row in enumerate(cm):
    true_label = f"T:{LABEL_NAMES[i]}"
    row_str = "".join(f"{v:>{col_w}}" for v in row)
    print(f"  {true_label:<12}|{row_str}")

print("\nClassification Report:")
report_str = classification_report(
    y_test, y_pred, target_names=LABEL_NAMES, digits=4
)
print(report_str)

# ── 5. Export metrics JSON ─────────────────────────────────────────────────────
report_dict = classification_report(
    y_test, y_pred, target_names=LABEL_NAMES, output_dict=True
)
summary = {
    "model": "LightGBM",
    "test_accuracy": round(acc, 6),
    "best_iteration": int(best_iter),
    "training_time_seconds": round(elapsed, 2),
    "confusion_matrix": cm.tolist(),
    "per_class": {
        cls: {
            "precision": round(report_dict[cls]["precision"], 4),
            "recall":    round(report_dict[cls]["recall"],    4),
            "f1_score":  round(report_dict[cls]["f1-score"],  4),
            "support":   int(report_dict[cls]["support"]),
        }
        for cls in LABEL_NAMES
    },
    "macro_avg": {
        k: round(report_dict["macro avg"][k], 4)
        for k in ("precision", "recall", "f1-score")
    },
    "weighted_avg": {
        k: round(report_dict["weighted avg"][k], 4)
        for k in ("precision", "recall", "f1-score")
    },
    "comparison": {
        "XGBoost":  {"test_accuracy": 0.9091, "macro_f1": 0.91,  "train_time_s": 135},
        "CatBoost": {"test_accuracy": 0.8879, "macro_f1": 0.884, "train_time_s": 189},
        "LightGBM": {"test_accuracy": round(acc, 4), "macro_f1": round(report_dict["macro avg"]["f1-score"], 4), "train_time_s": round(elapsed, 1)},
    },
}

METRICS_PATH.write_text(json.dumps(summary, indent=2))
print(f"Metrics saved -> {METRICS_PATH}")

# ── Summary banner ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Test Accuracy   : {acc * 100:.2f}%")
print(f"  Macro F1        : {report_dict['macro avg']['f1-score'] * 100:.2f}%")
print(f"  Best Iteration  : {best_iter}")
print(f"  Training Time   : {elapsed:.1f}s")
print(f"  Metrics saved   : {METRICS_PATH}")
print("=" * 60)

print("\n--- Classifier Comparison (ResNet-50 features) ---")
print(f"  {'Model':<12} {'Accuracy':>10} {'Macro F1':>10} {'Train(s)':>10}")
print(f"  {'-'*44}")
for name, v in summary["comparison"].items():
    print(f"  {name:<12} {v['test_accuracy']*100:>9.2f}% {v['macro_f1']*100:>9.2f}% {v['train_time_s']:>10}")
