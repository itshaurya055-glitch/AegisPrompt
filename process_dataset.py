"""
Prompt-Injection Dataset Pipeline
==================================
Steps:
  1. Load both parquet splits and merge them.
  2. Explore  - label distribution, prompt lengths, sample examples.
  3. Clean    - drop nulls, normalize whitespace, lowercase text.
  4. Split    - 80 / 10 / 10  stratified by label.
  5. Save     - train.csv, val.csv, test.csv in the project root.
"""

import sys
import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Fix Windows cp1252 encoding issue
sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────
# 0.  Paths
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR   = os.path.join(BASE_DIR, "data", "processed")      # save CSVs in data/processed

TRAIN_PQ  = os.path.join(DATA_DIR, "train-00000-of-00001-9564e8b05b4757ab.parquet")
TEST_PQ   = os.path.join(DATA_DIR, "test-00000-of-00001-701d16158af87368.parquet")

SEP = "-" * 60

# ─────────────────────────────────────────────
# 1.  Load
# ─────────────────────────────────────────────
print(SEP)
print("LOADING DATA")
print(SEP)

df_train_raw = pd.read_parquet(TRAIN_PQ)
df_test_raw  = pd.read_parquet(TEST_PQ)

print(f"  Raw train split : {len(df_train_raw):>5} rows")
print(f"  Raw test split  : {len(df_test_raw):>5} rows")

df = pd.concat([df_train_raw, df_test_raw], ignore_index=True)
print(f"  Combined total  : {len(df):>5} rows")
print(f"  Columns         : {list(df.columns)}")

# ─────────────────────────────────────────────
# 2.  Explore (BEFORE cleaning)
# ─────────────────────────────────────────────
print()
print(SEP)
print("EXPLORATION  (raw data)")
print(SEP)

# 2a. Label distribution
label_map = {0: "benign", 1: "injection"}
print("\n  Label distribution:")
dist = df["label"].value_counts().sort_index()
for lbl, cnt in dist.items():
    pct = cnt / len(df) * 100
    name = label_map.get(lbl, str(lbl))
    print(f"    label={lbl} ({name:>9s}) : {cnt:>4d}  ({pct:5.1f}%)")

# 2b. Missing values
print("\n  Missing values per column:")
nulls = df.isnull().sum()
for col, n in nulls.items():
    print(f"    {col:<10}: {n}")

# 2c. Prompt-length statistics (characters)
df["_len"] = df["text"].dropna().apply(len)
print("\n  Prompt length (characters) — raw text:")
stats = df["_len"].describe(percentiles=[0.25, 0.5, 0.75, 0.95])
for stat, val in stats.items():
    print(f"    {stat:<10}: {val:>8.1f}")

# 2d. Sample examples
print("\n  Sample examples (5 random rows):")
sample = df.sample(min(5, len(df)), random_state=42)[["label", "text"]]
for i, row in sample.iterrows():
    tag = label_map.get(row["label"], str(row["label"]))
    preview = str(row["text"])[:120].replace("\n", " ")
    print(f"\n    [{tag}]")
    print(f"    {preview}{'...' if len(str(row['text'])) > 120 else ''}")

df.drop(columns=["_len"], inplace=True)

# ─────────────────────────────────────────────
# 3.  Clean
# ─────────────────────────────────────────────
print()
print(SEP)
print("CLEANING")
print(SEP)

original_len = len(df)

# 3a. Drop rows where text or label is null
df.dropna(subset=["text", "label"], inplace=True)
after_null = len(df)
print(f"  Dropped {original_len - after_null} null rows  →  {after_null} remaining")

# 3b. Normalize whitespace (collapse runs of spaces/tabs/newlines to single space)
def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

# 3c. Lowercase
def clean_text(s: str) -> str:
    s = normalize_whitespace(s)
    s = s.lower()
    return s

df["text"] = df["text"].apply(clean_text)

# 3d. Drop any accidental empty strings after cleaning
before_empty = len(df)
df = df[df["text"].str.len() > 0]
print(f"  Dropped {before_empty - len(df)} empty-text rows after cleaning")

# 3e. Ensure label is integer
df["label"] = df["label"].astype(int)

# 3f. Reset index
df.reset_index(drop=True, inplace=True)
print(f"  Final clean size : {len(df)} rows")

# Show post-clean length stats
df["_len"] = df["text"].apply(len)
print("\n  Prompt length (characters) — after cleaning:")
stats = df["_len"].describe(percentiles=[0.25, 0.5, 0.75, 0.95])
for stat, val in stats.items():
    print(f"    {stat:<10}: {val:>8.1f}")
df.drop(columns=["_len"], inplace=True)

# ─────────────────────────────────────────────
# 4.  Stratified 80 / 10 / 10 split
# ─────────────────────────────────────────────
print()
print(SEP)
print("SPLITTING  (80 / 10 / 10  stratified)")
print(SEP)

# First: 80% train, 20% temp
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# Then split the 20% temp into 50/50 → val 10% / test 10%
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
    dist_str = " | ".join(
        f"label={l}:{c}" for l, c in split["label"].value_counts().sort_index().items()
    )
    print(f"  {name:<6}: {len(split):>4} rows   [{dist_str}]")

# ─────────────────────────────────────────────
# 5.  Save
# ─────────────────────────────────────────────
print()
print(SEP)
print("SAVING  CSVs")
print(SEP)

os.makedirs(OUT_DIR, exist_ok=True)
for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    path = os.path.join(OUT_DIR, f"{split_name}.csv")
    split_df.to_csv(path, index=False, encoding="utf-8")
    print(f"  Saved {split_name}.csv  →  {path}")

# ─────────────────────────────────────────────
# 6.  Sanity check – report anything missing
# ─────────────────────────────────────────────
print()
print(SEP)
print("SANITY CHECK")
print(SEP)

issues = []

total_out = len(train_df) + len(val_df) + len(test_df)
if total_out != len(df):
    issues.append(f"Row count mismatch: input={len(df)}, output splits sum={total_out}")

for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    path = os.path.join(OUT_DIR, f"{split_name}.csv")
    if not os.path.exists(path):
        issues.append(f"File not found: {path}")
    else:
        reloaded = pd.read_csv(path)
        if len(reloaded) != len(split_df):
            issues.append(f"{split_name}.csv row count wrong: expected {len(split_df)}, got {len(reloaded)}")
        if list(reloaded.columns) != ["text", "label"]:
            issues.append(f"{split_name}.csv columns unexpected: {list(reloaded.columns)}")

# Check stratification ratio tolerance (each class within 2% of 80/10/10)
for lbl in df["label"].unique():
    total_lbl = (df["label"] == lbl).sum()
    train_lbl = (train_df["label"] == lbl).sum()
    ratio = train_lbl / total_lbl
    if not (0.75 <= ratio <= 0.85):
        issues.append(f"Stratification off for label={lbl}: train ratio={ratio:.2f}")

if issues:
    print("  ⚠  Issues found:")
    for iss in issues:
        print(f"    - {iss}")
else:
    print("  ✓ All checks passed — no issues detected!")

print()
print(SEP)
print("DONE")
print(SEP)
