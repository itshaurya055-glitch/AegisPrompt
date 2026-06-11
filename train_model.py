"""
DistilBERT Fine-Tuning - Prompt Injection Classifier
=====================================================
Model  : distilbert-base-uncased  (binary classification head)
Labels : 0 = safe/benign   |   1 = prompt injection
Loss   : Weighted CrossEntropyLoss (handles class imbalance)
Epochs : 3   |   LR : 2e-5   |   Max tokens : 128
Best checkpoint saved to: models/best_model/
"""

import sys
# Force UTF-8 output on Windows (must be before any print calls)
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import os
import json
import warnings
import numpy as np
import pandas as pd

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
)

# --- reproducibility ---
set_seed(42)

# --- paths ---
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV  = os.path.join(BASE_DIR, "data", "processed", "train.csv")
VAL_CSV    = os.path.join(BASE_DIR, "data", "processed", "val.csv")
TEST_CSV   = os.path.join(BASE_DIR, "data", "processed", "test.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models", "best_model")
CKPT_DIR   = os.path.join(BASE_DIR, "models", "checkpoints")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

SEP = "-" * 60

# --- hyper-params ---
MODEL_NAME   = "distilbert-base-uncased"
MAX_LEN      = 128
BATCH_SIZE   = 16
NUM_EPOCHS   = 3
LEARNING_RATE = 2e-5
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01

print(SEP)
print("DISTILBERT FINE-TUNING - PROMPT INJECTION CLASSIFIER")
print(SEP)
print(f"  Model        : {MODEL_NAME}")
print(f"  Max tokens   : {MAX_LEN}")
print(f"  Batch size   : {BATCH_SIZE}")
print(f"  Epochs       : {NUM_EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device       : {device}")
print()

# =============================================================
# 1. LOAD DATA
# =============================================================
print(SEP)
print("1. LOADING DATA")
print(SEP)

train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)
test_df  = pd.read_csv(TEST_CSV)

print(f"  Train : {len(train_df)} rows  label dist: {train_df['label'].value_counts().to_dict()}")
print(f"  Val   : {len(val_df)} rows  label dist: {val_df['label'].value_counts().to_dict()}")
print(f"  Test  : {len(test_df)} rows  label dist: {test_df['label'].value_counts().to_dict()}")

# =============================================================
# 2. TOKENIZER
# =============================================================
print()
print(SEP)
print("2. LOADING TOKENIZER")
print(SEP)

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
print(f"  Loaded tokenizer: {MODEL_NAME}")

# =============================================================
# 3. DATASET CLASS
# =============================================================

class InjectionDataset(Dataset):
    """
    Wraps a DataFrame into a PyTorch Dataset.
    Tokenizes on-the-fly with caching via __getitem__.
    """
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts  = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids"      : encoding["input_ids"].squeeze(0),
            "attention_mask" : encoding["attention_mask"].squeeze(0),
            "labels"         : torch.tensor(self.labels[idx], dtype=torch.long),
        }


train_dataset = InjectionDataset(train_df, tokenizer, MAX_LEN)
val_dataset   = InjectionDataset(val_df,   tokenizer, MAX_LEN)
test_dataset  = InjectionDataset(test_df,  tokenizer, MAX_LEN)

print(f"  Train dataset : {len(train_dataset)} samples")
print(f"  Val dataset   : {len(val_dataset)} samples")
print(f"  Test dataset  : {len(test_dataset)} samples")

# =============================================================
# 4. MODEL — DistilBERT + Binary Classification Head
# =============================================================
print()
print(SEP)
print("3. LOADING MODEL")
print(SEP)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "safe", 1: "injection"},
    label2id={"safe": 0, "injection": 1},
)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters     : {total_params:,}")
print(f"  Trainable parameters : {trainable:,}")

# =============================================================
# 5. CLASS-WEIGHT COMPUTATION (handles 60/40 imbalance)
# =============================================================
n_benign    = (train_df["label"] == 0).sum()
n_injection = (train_df["label"] == 1).sum()
n_total     = len(train_df)

# Weight = total / (n_classes * count_per_class)
weight_benign    = n_total / (2 * n_benign)
weight_injection = n_total / (2 * n_injection)
class_weights = torch.tensor([weight_benign, weight_injection], dtype=torch.float).to(device)

print()
print(f"  Class weights  ->  benign: {weight_benign:.4f} | injection: {weight_injection:.4f}")

# =============================================================
# 6. CUSTOM TRAINER — weighted cross-entropy / BCE
# =============================================================

class WeightedTrainer(Trainer):
    """
    Trainer subclass that uses weighted CrossEntropyLoss to handle
    class imbalance. This acts as BCE-equivalent for binary labels
    while remaining compatible with the HuggingFace Trainer API.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits                          # (batch, 2)

        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# =============================================================
# 7. METRICS — F1, Precision, Recall, Accuracy  per epoch
# =============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    f1        = f1_score(labels, preds, average="binary", pos_label=1)
    precision = precision_score(labels, preds, average="binary", pos_label=1, zero_division=0)
    recall    = recall_score(labels, preds, average="binary", pos_label=1, zero_division=0)
    accuracy  = accuracy_score(labels, preds)

    return {
        "accuracy"  : round(accuracy, 4),
        "f1"        : round(f1, 4),
        "precision" : round(precision, 4),
        "recall"    : round(recall, 4),
    }


# =============================================================
# 8. TRAINING ARGUMENTS
# =============================================================
print()
print(SEP)
print("4. TRAINING")
print(SEP)

training_args = TrainingArguments(
    output_dir=CKPT_DIR,

    # --- core schedule ---
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,

    # --- batch sizes ---
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    # --- evaluation strategy: evaluate every epoch ---
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    # --- logging ---
    logging_dir=os.path.join(BASE_DIR, "models", "logs"),
    logging_strategy="epoch",
    report_to="none",           # no wandb / mlflow

    # --- misc ---
    fp16=False,                 # CPU — no fp16
    dataloader_num_workers=0,   # Windows-safe
    save_total_limit=2,
    seed=42,
)

# =============================================================
# 9. TRAINER
# =============================================================

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# --- TRAIN ---
train_result = trainer.train()

print()
print(SEP)
print("TRAINING COMPLETE")
print(SEP)
print(f"  Total steps      : {train_result.global_step}")
print(f"  Training loss    : {train_result.training_loss:.4f}")
print(f"  Train time (s)   : {train_result.metrics.get('train_runtime', 0):.1f}")

# =============================================================
# 10. VALIDATION — per-epoch history already logged by Trainer
#     Print the full per-epoch validation table here
# =============================================================
print()
print(SEP)
print("VALIDATION METRICS (per epoch)")
print(SEP)

history = [
    log for log in trainer.state.log_history
    if "eval_f1" in log
]

print(f"  {'Epoch':<8} {'Loss':<10} {'Accuracy':<12} {'F1':<10} {'Precision':<12} {'Recall':<10}")
print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")
for h in history:
    print(
        f"  {h.get('epoch', '?'):<8.1f} "
        f"{h.get('eval_loss', 0):<10.4f} "
        f"{h.get('eval_accuracy', 0):<12.4f} "
        f"{h.get('eval_f1', 0):<10.4f} "
        f"{h.get('eval_precision', 0):<12.4f} "
        f"{h.get('eval_recall', 0):<10.4f}"
    )

# =============================================================
# 11. TEST SET EVALUATION
# =============================================================
print()
print(SEP)
print("TEST SET EVALUATION")
print(SEP)

test_results = trainer.evaluate(eval_dataset=test_dataset)

print(f"  Test Loss      : {test_results.get('eval_loss', 0):.4f}")
print(f"  Test Accuracy  : {test_results.get('eval_accuracy', 0):.4f}")
print(f"  Test F1        : {test_results.get('eval_f1', 0):.4f}")
print(f"  Test Precision : {test_results.get('eval_precision', 0):.4f}")
print(f"  Test Recall    : {test_results.get('eval_recall', 0):.4f}")

# Full classification report
preds_out = trainer.predict(test_dataset)
preds     = np.argmax(preds_out.predictions, axis=-1)
labels    = preds_out.label_ids

print()
print("  Full Classification Report:")
print()
report = classification_report(
    labels, preds,
    target_names=["safe (0)", "injection (1)"],
    digits=4,
)
for line in report.split("\n"):
    print(f"    {line}")

# =============================================================
# 12. SAVE BEST MODEL
# =============================================================
print()
print(SEP)
print("SAVING BEST MODEL")
print(SEP)

trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

# Save training metadata
metadata = {
    "model_name"    : MODEL_NAME,
    "max_len"       : MAX_LEN,
    "num_epochs"    : NUM_EPOCHS,
    "learning_rate" : LEARNING_RATE,
    "batch_size"    : BATCH_SIZE,
    "class_weights" : {"safe": weight_benign, "injection": weight_injection},
    "val_history"   : history,
    "test_results"  : {k: v for k, v in test_results.items()},
}

meta_path = os.path.join(MODEL_DIR, "training_metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  Model saved to  : {MODEL_DIR}")
print(f"  Metadata saved  : {meta_path}")

print()
print(SEP)
print("DONE — best model is at:  models/best_model/")
print(SEP)
