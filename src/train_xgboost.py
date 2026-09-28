"""
XGBoost classifier trained on ResNet-50 extracted features.

Loads pre-extracted features from results/resnet50_features.npz,
trains on the training split, and evaluates on the test split.
"""

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

# ── 1. Load features ──────────────────────────────────────────────────────────
FEATURES_PATH = "results/resnet50_features.npz"
print(f"Loading features from '{FEATURES_PATH}' …")
data = np.load(FEATURES_PATH)

X_train, y_train = data["X_train"], data["y_train"]
X_val,   y_val   = data["X_val"],   data["y_val"]
X_test,  y_test  = data["X_test"],  data["y_test"]

print(f"  Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")
print(f"  Classes: {sorted(set(y_train.tolist()))}\n")

# ── 2. Train XGBoost ──────────────────────────────────────────────────────────
num_classes = len(set(y_train.tolist()))

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=num_classes,
    eval_metric="mlogloss",
    use_label_encoder=False,
    tree_method="hist",        # fast histogram method
    n_jobs=-1,
    random_state=42,
    verbosity=1,
)

print("Training XGBoost classifier …")
t0 = time.time()
xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,           # print every 50 rounds
)
elapsed = time.time() - t0
print(f"\nTraining complete in {elapsed:.1f}s\n")

# ── 3. Evaluate on test set ───────────────────────────────────────────────────
y_pred = xgb.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)

print("=" * 55)
print(f"  Test Accuracy : {acc * 100:.2f}%")
print("=" * 55)

print("\nConfusion Matrix (rows = true, cols = predicted):")
# Pretty-print with class headers
class_labels = [str(c) for c in sorted(set(y_test.tolist()))]
header = "       " + "  ".join(f"P{c:>2}" for c in class_labels)
print(header)
print("      " + "-" * (len(header) - 6))
for i, row in enumerate(cm):
    row_str = "  ".join(f"{v:>4}" for v in row)
    print(f"  T{class_labels[i]:>2} | {row_str}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                             target_names=[f"Class {c}" for c in class_labels]))
