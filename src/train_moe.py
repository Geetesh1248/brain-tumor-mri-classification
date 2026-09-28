"""
Mixture of Experts (MoE) ensemble for brain tumor MRI classification.

Architecture:
  - Experts  : XGBoost, CatBoost, LightGBM (frozen, pre-trained)
  - Gate     : nn.Linear(2048, 3) -> Softmax
  - Output   : weighted sum of expert 4-class probability vectors

Pipeline:
  1. Load / train-and-save the three base classifiers.
  2. Generate frozen predict_proba arrays for all splits.
  3. Train the gating network with CrossEntropyLoss + Adam.
  4. Evaluate on the test set and export metrics.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── XGBoost / CatBoost / LightGBM ─────────────────────────────────────────────
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, log_evaluation

# ── Config ─────────────────────────────────────────────────────────────────────
FEATURES_PATH   = Path("results/resnet50_features.npz")
MODEL_DIR       = Path("models")
XGB_PATH        = MODEL_DIR / "xgboost_resnet50.json"
CB_PATH         = MODEL_DIR / "catboost_resnet50.cbm"
LGB_PATH        = MODEL_DIR / "lightgbm_resnet50.txt"
MOE_PATH        = MODEL_DIR / "moe_gate.pt"
METRICS_PATH    = MODEL_DIR / "moe_metrics.json"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["glioma", "healthy", "meningioma", "pituitary"]
N_CLASSES   = 4
N_EXPERTS   = 3
FEAT_DIM    = 2048

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load features
# ═══════════════════════════════════════════════════════════════════════════════
print(f"Loading features from '{FEATURES_PATH}' ...")
data = np.load(FEATURES_PATH)
X_train, y_train = data["X_train"].astype(np.float32), data["y_train"]
X_val,   y_val   = data["X_val"].astype(np.float32),   data["y_val"]
X_test,  y_test  = data["X_test"].astype(np.float32),  data["y_test"]
print(f"  Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Load or train expert classifiers
# ═══════════════════════════════════════════════════════════════════════════════

def get_xgboost() -> XGBClassifier:
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softmax", num_class=N_CLASSES,
        eval_metric="mlogloss", tree_method="hist",
        n_jobs=-1, random_state=42, verbosity=0,
    )
    if XGB_PATH.exists():
        print(f"  [XGBoost]  Loading from {XGB_PATH}")
        clf.load_model(str(XGB_PATH))
    else:
        print("  [XGBoost]  Training ...")
        clf.fit(X_train, y_train,
                eval_set=[(X_val, y_val)], verbose=False)
        clf.save_model(str(XGB_PATH))
        print(f"  [XGBoost]  Saved -> {XGB_PATH}")
    return clf


def get_catboost() -> CatBoostClassifier:
    clf = CatBoostClassifier(
        iterations=300, learning_rate=0.1, depth=6,
        loss_function="MultiClass", random_seed=42, verbose=0,
    )
    if CB_PATH.exists():
        print(f"  [CatBoost] Loading from {CB_PATH}")
        clf.load_model(str(CB_PATH))
    else:
        print("  [CatBoost] Training ...")
        clf.fit(Pool(X_train, y_train), eval_set=Pool(X_val, y_val),
                use_best_model=True)
        clf.save_model(str(CB_PATH))
        print(f"  [CatBoost] Saved -> {CB_PATH}")
    return clf


def get_lightgbm() -> LGBMClassifier:
    clf = LGBMClassifier(
        n_estimators=300, learning_rate=0.1, num_leaves=31,
        objective="multiclass", num_class=N_CLASSES,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    if LGB_PATH.exists():
        print(f"  [LightGBM] Loading from {LGB_PATH}")
        clf._Booster = __import__("lightgbm").Booster(
            model_file=str(LGB_PATH))
        clf.fitted_ = True          # mark as fitted for predict
        # re-fit to get sklearn wrapper properly wired (fast, uses saved booster)
        import lightgbm as lgb
        bst = lgb.Booster(model_file=str(LGB_PATH))
        # Use booster directly for predict_proba
        return bst                  # return raw booster; handled below
    else:
        print("  [LightGBM] Training ...")
        clf.fit(X_train, y_train,
                eval_X=X_val, eval_y=y_val,
                eval_metric="multi_logloss",
                callbacks=[log_evaluation(period=100)])
        clf.booster_.save_model(str(LGB_PATH))
        print(f"  [LightGBM] Saved -> {LGB_PATH}")
    return clf


print("=== Loading / training expert classifiers ===")
t0 = time.time()
xgb_clf = get_xgboost()
cb_clf  = get_catboost()
lgb_clf = get_lightgbm()
print(f"Experts ready in {time.time()-t0:.1f}s\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 2b. Generate frozen predict_proba for all splits
# ═══════════════════════════════════════════════════════════════════════════════
print("=== Generating expert probability predictions (frozen) ===")

import lightgbm as lgb  # noqa: E402

def expert_proba(clf, X: np.ndarray) -> np.ndarray:
    """Return (N, 4) probability array from any of the three expert types."""
    if isinstance(clf, lgb.Booster):
        raw = clf.predict(X)          # returns (N, 4) for multiclass
        return raw.astype(np.float32)
    elif isinstance(clf, XGBClassifier):
        return clf.predict_proba(X).astype(np.float32)
    elif isinstance(clf, CatBoostClassifier):
        return clf.predict_proba(X).astype(np.float32)
    elif isinstance(clf, LGBMClassifier):
        return clf.predict_proba(X).astype(np.float32)
    raise TypeError(f"Unknown classifier type: {type(clf)}")

# Shape: (N_experts, N_samples, N_classes)
def stack_expert_probas(X):
    return np.stack([
        expert_proba(xgb_clf, X),
        expert_proba(cb_clf,  X),
        expert_proba(lgb_clf, X),
    ], axis=0)   # (3, N, 4)

P_train = stack_expert_probas(X_train)   # (3, 4617, 4)
P_val   = stack_expert_probas(X_val)     # (3, 990,  4)
P_test  = stack_expert_probas(X_test)    # (3, 990,  4)
print(f"  Expert proba tensors: train={P_train.shape}, val={P_val.shape}, test={P_test.shape}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. PyTorch MoE architecture
# ═══════════════════════════════════════════════════════════════════════════════
class MixtureOfExperts(nn.Module):
    """
    Gating Network: Linear(2048 -> 3) + Softmax
    Forward       : weighted sum of the three expert probability vectors
    """
    def __init__(self, feat_dim: int = FEAT_DIM,
                 n_experts: int = N_EXPERTS, n_classes: int = N_CLASSES):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim, n_experts),
            nn.Softmax(dim=-1),
        )
        self.n_experts  = n_experts
        self.n_classes  = n_classes

    def forward(self, features: torch.Tensor,
                expert_probas: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features      : (B, 2048) ResNet-50 feature vectors
            expert_probas : (B, n_experts, n_classes) frozen prob arrays
        Returns:
            logits        : (B, n_classes)  weighted mixture
        """
        gate_weights = self.gate(features)                 # (B, 3)
        gate_weights = gate_weights.unsqueeze(-1)          # (B, 3, 1)
        mixture      = (gate_weights * expert_probas).sum(dim=1)  # (B, 4)
        # Return log for NLLLoss, or raw probabilities — use log here
        return torch.log(mixture.clamp(min=1e-8))          # (B, 4) log-probs

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Prepare PyTorch datasets
# ═══════════════════════════════════════════════════════════════════════════════
def make_dataset(X, P, y):
    return TensorDataset(
        torch.from_numpy(X),                            # (N, 2048)
        torch.from_numpy(P.transpose(1, 0, 2)),         # (N, 3, 4)
        torch.from_numpy(y).long(),
    )

train_ds = make_dataset(X_train, P_train, y_train)
val_ds   = make_dataset(X_val,   P_val,   y_val)
test_ds  = make_dataset(X_test,  P_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Train the gating network
# ═══════════════════════════════════════════════════════════════════════════════
model     = MixtureOfExperts().to(DEVICE)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

N_EPOCHS       = 50
PATIENCE       = 10
best_val_loss  = float("inf")
best_state     = None
no_improve     = 0

print("=== Training Gating Network (frozen experts) ===")
print(f"  Architecture : Linear({FEAT_DIM}, {N_EXPERTS}) -> Softmax -> weighted sum")
print(f"  Optimizer    : Adam (lr=1e-3, wd=1e-4)  |  Scheduler: CosineAnnealingLR")
print(f"  Criterion    : NLLLoss  |  Epochs: {N_EPOCHS}  |  Patience: {PATIENCE}\n")
print(f"  {'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>10}  {'Val Acc':>8}")
print(f"  {'-'*42}")

t0 = time.time()
for epoch in range(1, N_EPOCHS + 1):
    # -- train --
    model.train()
    train_loss = 0.0
    for feats, probs, labels in train_loader:
        feats, probs, labels = feats.to(DEVICE), probs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        log_mix = model(feats, probs)
        loss    = criterion(log_mix, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * feats.size(0)
    train_loss /= len(train_ds)
    scheduler.step()

    # -- validate --
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for feats, probs, labels in val_loader:
            feats, probs, labels = feats.to(DEVICE), probs.to(DEVICE), labels.to(DEVICE)
            log_mix   = model(feats, probs)
            val_loss += criterion(log_mix, labels).item() * feats.size(0)
            preds     = log_mix.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
    val_loss /= len(val_ds)
    val_acc   = val_correct / len(val_ds) * 100

    if epoch % 5 == 0 or epoch == 1:
        marker = " *" if val_loss < best_val_loss else ""
        print(f"  {epoch:>5}  {train_loss:>11.6f}  {val_loss:>10.6f}  {val_acc:>7.2f}%{marker}")

    # -- early stopping --
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve    = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
            break

elapsed_train = time.time() - t0

# Restore best weights
model.load_state_dict(best_state)
torch.save({"model_state": best_state, "best_val_loss": best_val_loss}, MOE_PATH)
print(f"\n  Best val loss : {best_val_loss:.6f}")
print(f"  Gate model saved -> {MOE_PATH}")
print(f"  Training time : {elapsed_train:.1f}s\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Evaluate on test set
# ═══════════════════════════════════════════════════════════════════════════════
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for feats, probs, labels in test_loader:
        feats, probs = feats.to(DEVICE), probs.to(DEVICE)
        log_mix = model(feats, probs)
        preds   = log_mix.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

y_pred = np.array(all_preds)
y_true = np.array(all_labels)

acc = accuracy_score(y_true, y_pred)
cm  = confusion_matrix(y_true, y_pred)

print("=" * 62)
print(f"  MoE Test Accuracy : {acc * 100:.2f}%  "
      f"(LightGBM baseline: 91.92%,  delta: {(acc - 0.9192)*100:+.2f}pp)")
print("=" * 62)

col_w = 13
print(f"\nConfusion Matrix (rows = true, cols = predicted):")
header = " " * 14 + "".join(f"{n:>{col_w}}" for n in CLASS_NAMES)
print(header)
print(" " * 14 + "-" * (col_w * N_CLASSES))
for i, row in enumerate(cm):
    print(f"  {('T:'+CLASS_NAMES[i]):<12}|" + "".join(f"{v:>{col_w}}" for v in row))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Export metrics JSON
# ═══════════════════════════════════════════════════════════════════════════════
report_dict = classification_report(
    y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)

summary = {
    "model": "MixtureOfExperts",
    "experts": ["XGBoost", "CatBoost", "LightGBM"],
    "gate_architecture": f"Linear({FEAT_DIM}, {N_EXPERTS}) -> Softmax",
    "test_accuracy": round(acc, 6),
    "best_val_loss": round(best_val_loss, 6),
    "gate_training_time_seconds": round(elapsed_train, 2),
    "confusion_matrix": cm.tolist(),
    "per_class": {
        cls: {
            "precision": round(report_dict[cls]["precision"], 4),
            "recall":    round(report_dict[cls]["recall"],    4),
            "f1_score":  round(report_dict[cls]["f1-score"],  4),
            "support":   int(report_dict[cls]["support"]),
        }
        for cls in CLASS_NAMES
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
        "XGBoost":  {"test_accuracy": 0.9091, "macro_f1": 0.91},
        "CatBoost": {"test_accuracy": 0.8879, "macro_f1": 0.884},
        "LightGBM": {"test_accuracy": 0.9192, "macro_f1": 0.9166},
        "MoE":      {"test_accuracy": round(acc, 4),
                     "macro_f1": round(report_dict["macro avg"]["f1-score"], 4)},
    },
}
METRICS_PATH.write_text(json.dumps(summary, indent=2))
print(f"Metrics saved -> {METRICS_PATH}")

# Summary banner
print("\n" + "=" * 62)
print("  FINAL COMPARISON (ResNet-50 features, test set n=990)")
print("=" * 62)
print(f"  {'Model':<12} {'Accuracy':>10} {'Macro F1':>10}")
print(f"  {'-'*34}")
for name, v in summary["comparison"].items():
    marker = " <-- best" if v["test_accuracy"] == max(
        vv["test_accuracy"] for vv in summary["comparison"].values()) else ""
    print(f"  {name:<12} {v['test_accuracy']*100:>9.2f}% "
          f"{v['macro_f1']*100:>9.2f}%{marker}")
print("=" * 62)
