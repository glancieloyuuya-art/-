#!/usr/bin/env python3
# 02_train_logreg.py
# Group-wise file split (C / D / ND): 2/3 train files, 1/3 test files (per group)
# Then train a single Logistic Regression model and evaluate overall + per group.
#
# Usage:
#   (recommended) run inside ./out:
#     python 02_train_logreg.py
#
#   or specify directory:
#     python 02_train_logreg.py --in_dir /path/to/out
#
# Output:
#   - model:  <in_dir>/logreg_apnea_model.joblib
#   - result: <in_dir>/result
#   - (optional) test predictions CSV: <in_dir>/test_predictions.csv

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import math

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)
import joblib


WIN = 10  # baseline window (minutes)


# ----------------------------
# Feature engineering (per subject file)
# ----------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    *_merged.csv の列構成に合わせて学習用特徴量を作る（被験者ごとに計算することが重要）

    必須入力列:
      - label_1m
      - rr_mean_ms
      - spo2_mean_pct

    任意入力列（あれば使える）:
      - rr_std_ms, rr_min_ms, rr_max_ms
      - spo2_min_pct, spo2_max_pct
    """
    df = df.copy()

    # minute index（segment_indexがあればそれを使う）
    if "minute" not in df.columns:
        if "segment_index" in df.columns:
            df = df.rename(columns={"segment_index": "minute"})
        else:
            df.insert(0, "minute", np.arange(len(df), dtype=int))

    # 数値化
    must = ["label_1m", "rr_mean_ms", "spo2_mean_pct"]
    for c in must:
        if c not in df.columns:
            raise KeyError(f"必要列 '{c}' がCSVにありません。columns={df.columns.tolist()}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 時系列順に（minuteが連番前提でも安全のため）
    df = df.sort_values("minute").reset_index(drop=True)

    # HR（bpm）をRR平均(ms)から計算
    df["hr_mean_1m"] = 60000.0 / df["rr_mean_ms"]

    # baseline（直前10分中央値：current含めない）
    df["spo2_base10"] = df["spo2_mean_pct"].shift(1).rolling(WIN, min_periods=WIN).median()
    df["hr_base10"]   = df["hr_mean_1m"].shift(1).rolling(WIN, min_periods=WIN).median()

    # baseline差
    df["spo2_drop_from_baseline"] = df["spo2_base10"] - df["spo2_mean_pct"]
    df["hr_rise_from_baseline"]   = df["hr_mean_1m"] - df["hr_base10"]

    # 1分差分
    df["spo2_diff_1m"] = df["spo2_mean_pct"].diff(1)
    df["hr_diff_1m"]   = df["hr_mean_1m"].diff(1)

    # optional columns: numeric化（存在する場合のみ）
    optional = ["rr_std_ms", "rr_min_ms", "rr_max_ms", "spo2_min_pct", "spo2_max_pct"]
    for c in optional:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # inf除去
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


# ----------------------------
# Grouping / splitting
# ----------------------------
def subject_id_from_filename(p: Path) -> str:
    name = p.name
    if name.endswith("_merged.csv"):
        return name[:-len("_merged.csv")]
    return p.stem


def detect_group(subject_id: str) -> str | None:
    # ND is two letters, so check it first
    if subject_id.startswith("ND"):
        return "ND"
    if subject_id.startswith("C"):
        return "C"
    if subject_id.startswith("D"):
        return "D"
    return None


def split_files_groupwise(files_by_group: dict[str, list[Path]], seed: int) -> tuple[list[Path], list[Path], dict]:
    rng = np.random.default_rng(seed)
    train_files: list[Path] = []
    test_files: list[Path] = []
    summary = {}

    for g, files in files_by_group.items():
        files = sorted(files)
        n = len(files)
        if n == 0:
            continue
        # 1/3 test, 2/3 train (at least 1 test if possible)
        n_test = max(1, int(math.ceil(n / 3)))
        idx = rng.permutation(n)
        test_idx = set(idx[:n_test].tolist())
        g_test = [files[i] for i in range(n) if i in test_idx]
        g_train = [files[i] for i in range(n) if i not in test_idx]

        train_files.extend(g_train)
        test_files.extend(g_test)

        summary[g] = {
            "n_total": n,
            "n_train": len(g_train),
            "n_test": len(g_test),
            "train_subjects": [subject_id_from_filename(p) for p in g_train],
            "test_subjects": [subject_id_from_filename(p) for p in g_test],
        }

    return train_files, test_files, summary


# ----------------------------
# Logging helper (print + save)
# ----------------------------
class TeeLogger:
    def __init__(self):
        self.lines: list[str] = []

    def log(self, msg: str = ""):
        print(msg)
        self.lines.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default=".", help="*_merged.csv が置いてあるディレクトリ（デフォルト: カレント）")
    ap.add_argument("--pattern", type=str, default="*_merged.csv", help="読み込むファイルのglobパターン（デフォルト: *_merged.csv）")
    ap.add_argument("--label_col", type=str, default="label_1m", help="教師ラベル列（デフォルト: label_1m）")
    ap.add_argument("--seed", type=int, default=0, help="グループ内ファイル分割の乱数シード（デフォルト: 0）")
    ap.add_argument("--result_file", type=str, default="result", help="実行結果を書き出すファイル名（デフォルト: result）")
    ap.add_argument("--save_test_pred", action="store_true", help="test_predictions.csv を保存する")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    if not in_dir.exists():
        raise FileNotFoundError(f"in_dir not found: {in_dir}")

    log = TeeLogger()

    # gather files
    all_files = sorted(in_dir.glob(args.pattern))
    if not all_files:
        raise FileNotFoundError(f"No files matched: {in_dir}/{args.pattern}")

    # group them
    files_by_group: dict[str, list[Path]] = {"C": [], "D": [], "ND": []}
    skipped: list[Path] = []
    for p in all_files:
        sid = subject_id_from_filename(p)
        g = detect_group(sid)
        if g is None:
            skipped.append(p)
            continue
        files_by_group[g].append(p)

    log.log("============================================================")
    log.log(" Group-wise subject split (file-level) : 2/3 train, 1/3 test")
    log.log("============================================================")
    log.log(f"[INFO] in_dir      : {in_dir}")
    log.log(f"[INFO] pattern     : {args.pattern}")
    log.log(f"[INFO] seed        : {args.seed}")
    if skipped:
        log.log(f"[WARN] skipped (unknown group): {len(skipped)} files")

    train_files, test_files, split_summary = split_files_groupwise(files_by_group, seed=args.seed)

    for g in ["C", "D", "ND"]:
        s = split_summary.get(g)
        if not s:
            log.log(f"[WARN] group {g}: 0 files")
            continue
        log.log(f"[SPLIT] {g}: total={s['n_total']}  train={s['n_train']}  test={s['n_test']}")
        log.log(f"        train: {', '.join(s['train_subjects'][:20])}{' ...' if len(s['train_subjects'])>20 else ''}")
        log.log(f"        test : {', '.join(s['test_subjects'][:20])}{' ...' if len(s['test_subjects'])>20 else ''}")

    if not train_files or not test_files:
        raise RuntimeError("Train/Test files are empty. Check grouping and pattern.")

    # ----------------------------
    # Load each subject file, build features per file, then concat
    # ----------------------------
    def load_one(p: Path) -> pd.DataFrame:
        sid = subject_id_from_filename(p)
        g = detect_group(sid) or "UNK"
        df = pd.read_csv(p)
        df = add_features(df)
        df["subject_id"] = sid
        df["group"] = g
        return df

    log.log("")
    log.log("[INFO] loading train files ...")
    train_dfs = []
    for p in train_files:
        try:
            train_dfs.append(load_one(p))
        except Exception as e:
            log.log(f"[WARN] failed to load {p.name}: {e}")

    log.log("[INFO] loading test files ...")
    test_dfs = []
    for p in test_files:
        try:
            test_dfs.append(load_one(p))
        except Exception as e:
            log.log(f"[WARN] failed to load {p.name}: {e}")

    if not train_dfs or not test_dfs:
        raise RuntimeError("No data loaded. Please check CSV contents/columns.")

    df_train_all = pd.concat(train_dfs, ignore_index=True)
    df_test_all = pd.concat(test_dfs, ignore_index=True)

    # ----------------------------
    # Feature set (common columns across all loaded files)
    # ----------------------------
    base_features = [
        "spo2_mean_pct",
        "hr_mean_1m",
        "spo2_drop_from_baseline",
        "hr_rise_from_baseline",
        "spo2_diff_1m",
        "hr_diff_1m",
    ]
    optional_features = ["rr_std_ms", "spo2_min_pct", "spo2_max_pct", "rr_min_ms", "rr_max_ms"]
    candidates = base_features + optional_features

    # columns common to both train/test dataframes
    common_cols = set(df_train_all.columns) & set(df_test_all.columns)
    features = [c for c in candidates if c in common_cols]

    missing_base = [c for c in base_features if c not in features]
    if missing_base:
        raise KeyError(f"Base features missing after load: {missing_base}")

    label_col = args.label_col
    if label_col not in common_cols:
        raise KeyError(f"label_col '{label_col}' not found. columns={sorted(common_cols)}")

    # numeric conversion (safety)
    for c in features + [label_col]:
        df_train_all[c] = pd.to_numeric(df_train_all[c], errors="coerce")
        df_test_all[c] = pd.to_numeric(df_test_all[c], errors="coerce")

    # Drop NaN rows
    train_data = df_train_all.dropna(subset=features + [label_col]).copy()
    test_data  = df_test_all.dropna(subset=features + [label_col]).copy()

    X_train = train_data[features]
    y_train = train_data[label_col].astype(int)

    X_test = test_data[features]
    y_test = test_data[label_col].astype(int)

    log.log("")
    log.log("============================================================")
    log.log(" Data summary")
    log.log("============================================================")
    log.log(f"[INFO] features used ({len(features)}): {features}")
    log.log(f"[INFO] train rows(after dropna): {len(train_data)}   label_counts={y_train.value_counts().to_dict()}")
    log.log(f"[INFO] test  rows(after dropna): {len(test_data)}    label_counts={y_test.value_counts().to_dict()}")

    # ----------------------------
    # Train
    # ----------------------------
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=3000))
    ])
    model.fit(X_train, y_train)

    # Predict
    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, pred)
    correct = int((y_test.values == pred).sum())
    total = int(len(y_test))

    log.log("")
    log.log("============================================================")
    log.log(" Overall result")
    log.log("============================================================")
    log.log(f"✅ 正答率(Accuracy): {acc*100:.2f}%  ({correct}/{total})")

    try:
        auc = roc_auc_score(y_test, p)
        log.log(f"📈 AUC(ROC): {auc:.4f}")
    except Exception as e:
        log.log(f"[WARN] AUC could not be computed: {e}")

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    log.log("")
    log.log("Confusion Matrix (rows=true, cols=pred)")
    log.log("          pred0   pred1")
    log.log(f"true0     {tn:5d}  {fp:5d}")
    log.log(f"true1     {fn:5d}  {tp:5d}")

    log.log("")
    log.log(classification_report(y_test, pred, digits=4))

    # ----------------------------
    # Per-group accuracy on test set
    # ----------------------------
    log.log("============================================================")
    log.log(" Group-wise accuracy on TEST (for quick check)")
    log.log("============================================================")
    test_data = test_data.copy()
    test_data["pred"] = pred
    for g in ["C", "D", "ND"]:
        sub = test_data[test_data["group"] == g]
        if len(sub) == 0:
            log.log(f"[GROUP {g}] (no rows)")
            continue
        g_acc = accuracy_score(sub[label_col].astype(int), sub["pred"].astype(int))
        log.log(f"[GROUP {g}] rows={len(sub):5d}  accuracy={g_acc*100:6.2f}%   label_counts={sub[label_col].astype(int).value_counts().to_dict()}")

    # ----------------------------
    # Save artifacts
    # ----------------------------
    out_model = in_dir / "logreg_apnea_model.joblib"
    joblib.dump(model, out_model)
    log.log("")
    log.log(f"[OK] saved model: {out_model}")

    if args.save_test_pred:
        out_pred = in_dir / "test_predictions.csv"
        cols_to_save = ["subject_id", "group", "minute", label_col] + features + ["pred"]
        cols_to_save = [c for c in cols_to_save if c in test_data.columns]
        test_data[cols_to_save].to_csv(out_pred, index=False)
        log.log(f"[OK] saved test predictions: {out_pred}")

    # Save result log
    result_path = Path(args.result_file)
    if not result_path.is_absolute():
        result_path = in_dir / result_path
    result_path.write_text("\n".join(log.lines) + "\n", encoding="utf-8")
    log.log(f"[OK] saved result log: {result_path}")

    log.log("============================================================")
    log.log("Done.")
    log.log("============================================================")


if __name__ == "__main__":
    main()
