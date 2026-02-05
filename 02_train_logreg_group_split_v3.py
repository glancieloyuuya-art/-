#!/usr/bin/env python3
# 02_train_logreg_group_split_v3.py
# ------------------------------------------------------------
# Group-wise file split (C / D / ND): 2/3 train files, 1/3 test files (per group)
# Train Logistic Regression, evaluate, and export TEST judgments per file into result/ as CSVs.
#
# Output in <in_dir>/result/ (or fallback if not writable):
#   - overall_result.txt
#   - test_summary_by_file.csv
#   - <SUBJECT>_test_pred.csv  (row-wise predictions per test subject)
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import math
from pathlib import Path
from datetime import datetime
from typing import Dict

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

WIN10 = 10  # baseline window 10 min
WIN5  = 5
WIN20 = 20


def subject_id_from_filename(p: Path) -> str:
    name = p.name
    return name[:-len("_merged.csv")] if name.endswith("_merged.csv") else p.stem


def detect_group(subject_id: str) -> str | None:
    if subject_id.startswith("ND"):
        return "ND"
    if subject_id.startswith("C"):
        return "C"
    if subject_id.startswith("D"):
        return "D"
    return None


def split_files_groupwise(files_by_group: dict[str, list[Path]], seed: int):
    rng = np.random.default_rng(seed)
    train_files, test_files = [], []
    summary = {}

    for g, files in files_by_group.items():
        files = sorted(files)
        n = len(files)
        if n == 0:
            continue

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


class TeeLogger:
    def __init__(self):
        self.lines: list[str] = []

    def log(self, msg: str = ""):
        print(msg)
        self.lines.append(msg)


def ensure_writable_dir(target: Path, log: TeeLogger) -> Path:
    """
    Make sure target is a writable directory.
    If not writable (e.g., owned by root due to sudo), fallback to result_YYYYMMDD_HHMMSS
    """
    if target.exists() and target.is_file():
        raise FileExistsError(f"'{target}' exists as a FILE. Please rename/remove it.")

    target.mkdir(parents=True, exist_ok=True)

    # write test
    try:
        testfile = target / ".write_test"
        testfile.write_text("ok", encoding="utf-8")
        testfile.unlink(missing_ok=True)
        return target
    except PermissionError:
        fallback = target.parent / f"{target.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        fallback.mkdir(parents=True, exist_ok=True)
        log.log(f"[WARN] '{target}' is not writable. Fallback -> {fallback}")
        return fallback


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    強化版特徴量（OSA向け）
    - SpO2 drop は mean と min を両方使う
    - desaturation event / ODIっぽい頻度
    - lag（遅れ）特徴
    - RR/SpO2 range, CV
    """
    df = df.copy()

    if "minute" not in df.columns:
        if "segment_index" in df.columns:
            df = df.rename(columns={"segment_index": "minute"})
        else:
            df.insert(0, "minute", np.arange(len(df), dtype=int))

    must = ["label_1m", "rr_mean_ms", "spo2_mean_pct"]
    for c in must:
        if c not in df.columns:
            raise KeyError(f"必要列 '{c}' がCSVにありません。columns={df.columns.tolist()}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # optional numeric
    opt = ["rr_std_ms", "rr_min_ms", "rr_max_ms", "spo2_min_pct", "spo2_max_pct"]
    for c in opt:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("minute").reset_index(drop=True)

    # HR mean from RR mean
    df["hr_mean_1m"] = 60000.0 / df["rr_mean_ms"]

    # Baselines (prev windows) - no leakage: shift(1)
    for w in [WIN5, WIN10, WIN20]:
        df[f"spo2_base{w}"] = df["spo2_mean_pct"].shift(1).rolling(w, min_periods=w).median()
        df[f"hr_base{w}"]   = df["hr_mean_1m"].shift(1).rolling(w, min_periods=w).median()

        df[f"spo2_drop{w}_mean"] = df[f"spo2_base{w}"] - df["spo2_mean_pct"]
        df[f"hr_rise{w}_mean"]   = df["hr_mean_1m"] - df[f"hr_base{w}"]

        if "spo2_min_pct" in df.columns:
            df[f"spo2_drop{w}_min"] = df[f"spo2_base{w}"] - df["spo2_min_pct"]

    # 1-min diffs
    df["spo2_diff_1m"] = df["spo2_mean_pct"].diff(1)
    df["hr_diff_1m"]   = df["hr_mean_1m"].diff(1)

    # ranges / CV (if possible)
    if "rr_std_ms" in df.columns:
        df["rr_cv"] = df["rr_std_ms"] / df["rr_mean_ms"]
    if "rr_min_ms" in df.columns and "rr_max_ms" in df.columns:
        df["rr_range"] = df["rr_max_ms"] - df["rr_min_ms"]
    if "spo2_min_pct" in df.columns and "spo2_max_pct" in df.columns:
        df["spo2_range"] = df["spo2_max_pct"] - df["spo2_min_pct"]

    # Desaturation event (3% drop) based on 10-min baseline and SpO2 min if available
    if "spo2_min_pct" in df.columns:
        df["desat3"] = (df["spo2_drop10_min"] >= 3.0).astype(int)
    else:
        df["desat3"] = (df["spo2_drop10_mean"] >= 3.0).astype(int)

    # ODI-like counts (past window, no leakage)
    df["desat3_count_10"] = df["desat3"].shift(1).rolling(10, min_periods=10).sum()
    df["desat3_count_30"] = df["desat3"].shift(1).rolling(30, min_periods=30).sum()

    # Lags (no future)
    for k in [1, 2]:
        df[f"spo2_drop10_mean_lag{k}"] = df["spo2_drop10_mean"].shift(k)
        df[f"hr_rise10_mean_lag{k}"]   = df["hr_rise10_mean"].shift(k)
        if "spo2_min_pct" in df.columns:
            df[f"spo2_drop10_min_lag{k}"] = df["spo2_drop10_min"].shift(k)

    # Interaction terms
    if "spo2_min_pct" in df.columns:
        df["drop_x_rise"] = df["spo2_drop10_min"] * df["hr_rise10_mean"]
    else:
        df["drop_x_rise"] = df["spo2_drop10_mean"] * df["hr_rise10_mean"]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default=".", help="*_merged.csv が置いてあるディレクトリ")
    ap.add_argument("--pattern", type=str, default="*_merged.csv")
    ap.add_argument("--label_col", type=str, default="label_1m")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--result_dir", type=str, default="result")
    ap.add_argument("--C", type=float, default=1.0, help="LogReg regularization strength (C)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    log = TeeLogger()

    # result dir (writable check)
    result_dir = ensure_writable_dir(in_dir / args.result_dir, log)

    # gather files
    all_files = sorted(in_dir.glob(args.pattern))
    if not all_files:
        raise FileNotFoundError(f"No files matched: {in_dir}/{args.pattern}")

    files_by_group: dict[str, list[Path]] = {"C": [], "D": [], "ND": []}
    for p in all_files:
        sid = subject_id_from_filename(p)
        g = detect_group(sid)
        if g:
            files_by_group[g].append(p)

    log.log("============================================================")
    log.log(" Group-wise subject split (file-level) : 2/3 train, 1/3 test")
    log.log("============================================================")
    log.log(f"[INFO] in_dir      : {in_dir}")
    log.log(f"[INFO] pattern     : {args.pattern}")
    log.log(f"[INFO] seed        : {args.seed}")
    log.log(f"[INFO] threshold   : {args.threshold}")
    log.log(f"[INFO] result_dir  : {result_dir}")
    log.log(f"[INFO] LogReg C     : {args.C}")

    train_files, test_files, split_summary = split_files_groupwise(files_by_group, seed=args.seed)

    for g in ["C", "D", "ND"]:
        s = split_summary.get(g)
        if not s:
            log.log(f"[WARN] group {g}: 0 files")
            continue
        log.log(f"[SPLIT] {g}: total={s['n_total']}  train={s['n_train']}  test={s['n_test']}")
        log.log(f"        train: {', '.join(s['train_subjects'][:30])}{' ...' if len(s['train_subjects'])>30 else ''}")
        log.log(f"        test : {', '.join(s['test_subjects'][:30])}{' ...' if len(s['test_subjects'])>30 else ''}")

    if not train_files or not test_files:
        raise RuntimeError("Train/Test files are empty. Check grouping/pattern.")

    def load_one(p: Path) -> pd.DataFrame:
        sid = subject_id_from_filename(p)
        g = detect_group(sid) or "UNK"
        df = pd.read_csv(p)
        df = add_features(df)
        df["subject_id"] = sid
        df["group"] = g
        df["source_file"] = p.name
        return df

    log.log("")
    log.log("[INFO] loading train files ...")
    train_dfs = [load_one(p) for p in train_files]
    log.log("[INFO] loading test files ...")
    test_by_subject: Dict[str, pd.DataFrame] = {load_one(p)["subject_id"].iloc[0]: load_one(p) for p in test_files}

    df_train_all = pd.concat(train_dfs, ignore_index=True)
    df_test_all  = pd.concat(list(test_by_subject.values()), ignore_index=True)

    # Feature candidates (new)
    candidates = [
        # base
        "spo2_mean_pct", "hr_mean_1m", "spo2_diff_1m", "hr_diff_1m",
        "rr_std_ms", "rr_min_ms", "rr_max_ms", "spo2_min_pct", "spo2_max_pct",
        "rr_cv", "rr_range", "spo2_range",

        # baseline windows (10 is key)
        "spo2_drop10_mean", "hr_rise10_mean",
        "spo2_drop10_min",
        "spo2_drop5_mean", "hr_rise5_mean",
        "spo2_drop20_mean", "hr_rise20_mean",

        # event / ODI-like
        "desat3", "desat3_count_10", "desat3_count_30",

        # lags
        "spo2_drop10_mean_lag1", "spo2_drop10_mean_lag2",
        "hr_rise10_mean_lag1", "hr_rise10_mean_lag2",
        "spo2_drop10_min_lag1", "spo2_drop10_min_lag2",

        # interaction
        "drop_x_rise",
    ]

    common_cols = set(df_train_all.columns) & set(df_test_all.columns)
    features = [c for c in candidates if c in common_cols]

    label_col = args.label_col
    if label_col not in common_cols:
        raise KeyError(f"label_col '{label_col}' not found.")

    # numeric safety
    for c in features + [label_col]:
        df_train_all[c] = pd.to_numeric(df_train_all[c], errors="coerce")
        df_test_all[c]  = pd.to_numeric(df_test_all[c], errors="coerce")

    train_data = df_train_all.dropna(subset=features + [label_col]).copy()
    test_data_all = df_test_all.dropna(subset=features + [label_col]).copy()

    X_train = train_data[features]
    y_train = train_data[label_col].astype(int)

    X_test = test_data_all[features]
    y_test = test_data_all[label_col].astype(int)

    log.log("")
    log.log("============================================================")
    log.log(" Data summary")
    log.log("============================================================")
    log.log(f"[INFO] features used ({len(features)}): {features}")
    log.log(f"[INFO] train rows(after dropna): {len(train_data)}   label_counts={y_train.value_counts().to_dict()}")
    log.log(f"[INFO] test  rows(after dropna): {len(test_data_all)}    label_counts={y_test.value_counts().to_dict()}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=4000, C=args.C))
    ])
    model.fit(X_train, y_train)

    prob_all = model.predict_proba(X_test)[:, 1]
    pred_all = (prob_all >= args.threshold).astype(int)

    acc_all = accuracy_score(y_test, pred_all)
    correct_all = int((y_test.values == pred_all).sum())
    total_all = int(len(y_test))

    log.log("")
    log.log("============================================================")
    log.log(" Overall result (ALL TEST files merged)")
    log.log("============================================================")
    log.log(f"✅ 正答率(Accuracy): {acc_all*100:.2f}%  ({correct_all}/{total_all})")

    auc_all = safe_auc(y_test.values, prob_all)
    log.log(f"📈 AUC(ROC): {auc_all:.4f}" if auc_all is not None else "[WARN] AUC could not be computed.")

    cm = confusion_matrix(y_test, pred_all, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    log.log("")
    log.log("Confusion Matrix (rows=true, cols=pred)")
    log.log("          pred0   pred1")
    log.log(f"true0     {tn:5d}  {fp:5d}")
    log.log(f"true1     {fn:5d}  {tp:5d}")
    log.log("")
    log.log(classification_report(y_test, pred_all, digits=4))

    # Export per-file predictions
    log.log("============================================================")
    log.log(" Export TEST judgments per file -> result/*.csv")
    log.log("============================================================")

    per_file_rows = []
    for sid, df_raw in sorted(test_by_subject.items()):
        g = df_raw["group"].iloc[0]
        src = df_raw["source_file"].iloc[0]

        df = df_raw.copy()
        for c in features + [label_col]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=features + [label_col]).copy()
        if len(df) == 0:
            log.log(f"[WARN] {sid} ({g}) -> 0 rows after dropna, skipped.")
            continue

        X = df[features]
        y = df[label_col].astype(int).to_numpy()

        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= args.threshold).astype(int)

        acc = float(accuracy_score(y, pred))
        auc = safe_auc(y, prob)
        cm2 = confusion_matrix(y, pred, labels=[0, 1])
        tn2, fp2, fn2, tp2 = cm2.ravel()

        out_csv = result_dir / f"{sid}_test_pred.csv"
        export_cols = ["minute", "subject_id", "group", "source_file", label_col] + features
        export_cols = [c for c in export_cols if c in df.columns]
        out_df = df[export_cols].copy()
        out_df["prob_1"] = prob
        out_df["pred"] = pred
        out_df.to_csv(out_csv, index=False)

        per_file_rows.append({
            "subject_id": sid,
            "group": g,
            "source_file": src,
            "rows": int(len(df)),
            "label_1_count": int((y == 1).sum()),
            "label_0_count": int((y == 0).sum()),
            "accuracy": acc,
            "auc": auc if auc is not None else "",
            "tn": int(tn2), "fp": int(fp2), "fn": int(fn2), "tp": int(tp2),
            "threshold": args.threshold,
        })

        log.log(f"[OK] {sid:>4s} ({g}) rows={len(df):4d}  acc={acc*100:6.2f}%  auc={'-' if auc is None else f'{auc:.4f}'}  -> {out_csv.name}")

    summary_path = result_dir / "test_summary_by_file.csv"
    if per_file_rows:
        pd.DataFrame(per_file_rows).sort_values(["group", "subject_id"]).to_csv(summary_path, index=False)
        log.log(f"[OK] saved per-file summary: {summary_path}")

    # Save model + overall log
    out_model = in_dir / "logreg_apnea_model.joblib"
    joblib.dump(model, out_model)
    log.log(f"[OK] saved model: {out_model}")

    overall_log_path = result_dir / "overall_result.txt"
    overall_log_path.write_text("\n".join(log.lines) + "\n", encoding="utf-8")
    log.log(f"[OK] saved overall log: {overall_log_path}")

    log.log("============================================================")
    log.log("Done.")
    log.log("============================================================")


if __name__ == "__main__":
    main()
