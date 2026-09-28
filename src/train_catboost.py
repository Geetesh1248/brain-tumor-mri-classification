"""
CatBoostClassifier trained on ResNet-50 extracted features.

Loads pre-extracted features from results/resnet50_features.npz,
trains on the training split with validation monitoring, evaluates
on the test split, saves the model, and writes a summary JSON.
"""

import json
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_PATH = Path("results/resnet50_features.npz")
MODEL_DIR     = Path("models")
MODEL_PATH    = MODEL_DIR / "catboost_resnet50.cbm"
METRICS_PATH  = MODEL_DIR / "catboost_metrics.json"

CLASS_NAMES = {0: "glioma", 1: "healthy", 2: "meningioma", 3: "pituitary"}

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load features ──────────────────────────────────────────────────────────
print(f"Loading features from '{FEATURES_PATH}' …")
data = np.load(FEATURES_PATH)

X_train, y_train = data["X_train"], data["y_train"]
X_val,   y_val   = data["X_val"],   data["y_val"]
X_test,  y_test  = data["X_test"],  data["y_test"]

print(f"  Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")
print(f"  Classes: {CLASS_NAMES}\n")

# Build CatBoost Pool objects (no categorical features here)
train_pool = Pool(X_train, label=y_train)
val_pool   = Pool(X_val,   label=y_val)
test_pool  = Pool(X_test,  label=y_test)

# ── 2. Initialise CatBoostClassifier ─────────────────────────────────────────
model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    loss_function="MultiClass",
    random_seed=42,
    verbose=50,          # print every 50 rounds
)

# ── 3. Train with validation monitoring ──────────────────────────────────────
print("Training CatBoostClassifier …")
t0 = time.time()
model.fit(
    train_pool,
    eval_set=val_pool,
    use_best_model=True,   # keep weights at best val loss
)
elapsed = time.time() - t0
best_iter = model.get_best_iteration()
print(f"\nTraining complete in {elapsed:.1f}s  |  Best iteration: {best_iter}\n")

# ── 4. Evaluate on test set ───────────────────────────────────────────────────
y_pred = model.predict(test_pool).flatten().astype(int)

acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)
label_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]

print("=" * 60)
print(f"  Test Accuracy : {acc * 100:.2f}%")
print("=" * 60)

# Pretty-print confusion matrix
col_w = 12
print(f"\nConfusion Matrix (rows = true, cols = predicted):")
header = " " * 12 + "".join(f"{n:>{col_w}}" for n in label_names)
print(header)
print(" " * 12 + "-" * (col_w * len(label_names)))
for i, row in enumerate(cm):
    true_label = f"T:{label_names[i]}"
    row_str = "".join(f"{v:>{col_w}}" for v in row)
    print(f"  {true_label:<10}|{row_str}")

print(f"\nClassification Report:")
report_str = classification_report(
    y_test, y_pred, target_names=label_names, digits=4
)
print(report_str)

# ── 5. Save model & summary metrics ──────────────────────────────────────────
model.save_model(str(MODEL_PATH))
print(f"Model saved -> {MODEL_PATH}")

# Per-class metrics dict
report_dict = classification_report(
    y_test, y_pred, target_names=label_names, output_dict=True
)
summary = {
    "test_accuracy": round(acc, 6),
    "best_iteration": int(best_iter) if best_iter is not None else 299,
    "training_time_seconds": round(elapsed, 2),
    "confusion_matrix": cm.tolist(),
    "per_class": {
        cls: {
            "precision": round(report_dict[cls]["precision"], 4),
            "recall":    round(report_dict[cls]["recall"],    4),
            "f1_score":  round(report_dict[cls]["f1-score"],  4),
            "support":   int(report_dict[cls]["support"]),
        }
        for cls in label_names
    },
    "macro_avg": {
        k: round(report_dict["macro avg"][k], 4)
        for k in ("precision", "recall", "f1-score")
    },
    "weighted_avg": {
        k: round(report_dict["weighted avg"][k], 4)
        for k in ("precision", "recall", "f1-score")
    },
}

METRICS_PATH.write_text(json.dumps(summary, indent=2))
print(f"Metrics saved -> {METRICS_PATH}\n")

# ── Summary banner ────────────────────────────────────────────────────────────
print("=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Test Accuracy   : {acc * 100:.2f}%")
print(f"  Macro F1        : {summary['macro_avg']['f1-score'] * 100:.2f}%")
print(f"  Best Iteration  : {summary['best_iteration']}")
print(f"  Training Time   : {elapsed:.1f}s")
print(f"  Model saved to  : {MODEL_PATH}")
print(f"  Metrics saved to: {METRICS_PATH}")
print("=" * 60)
