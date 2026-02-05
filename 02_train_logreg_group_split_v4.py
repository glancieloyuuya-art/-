#!/usr/bin/env python3
# 02_train_logreg_group_split_v4_fixed_threshold.py
# ------------------------------------------------------------
# Goal: Improve OSA classification accuracy with stronger, OSA-aware features
# while keeping Logistic Regression (linear model) as the classifier.
#
# Split rule:
#   - Group-wise (C / D / ND), file-level split: 2/3 train files, 1/3 test files per group
#
# Outputs (in <in_dir>/result/ or fallback if not writable):
#   - overall_result.txt
#   - test_summary_by_file.csv
#   - <SUBJECT>_test_pred.csv  (row-wise predictions for each test subject)
#
# Optional:
#   - --auto_C         : choose regularization strength C on a validation split of TRAIN files
#   # 判定しきい値は FIXED_THRESHOLD (=0.5) に固定
#
# Usage:
#   cd /path/to/out
#   python 02_train_logreg_group_split_v4_fixed_threshold.py
#   python 02_train_logreg_group_split_v4_fixed_threshold.py  --auto_C
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

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


FIXED_THRESHOLD = 0.5  # 判定しきい値（固定）

# Windows (minutes) used for rolling features
WINS = [5, 10, 20, 30]
BASE_W = 10  # primary baseline window for OSA-like desaturation


# ----------------------------
# File helpers
# ----------------------------
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
    """2/3 train files, 1/3 test files per group (file-level, no leakage)."""
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


def split_train_into_train_val(train_files: List[Path], seed: int, val_frac: float) -> Tuple[List[Path], List[Path]]:
    """File-level split of TRAIN files into subtrain/val (for tuning threshold/C)."""
    rng = np.random.default_rng(seed + 12345)
    n = len(train_files)
    if n < 5:
        return train_files, []
    n_val = max(1, int(round(n * val_frac)))
    idx = rng.permutation(n)
    val_idx = set(idx[:n_val].tolist())
    subtrain = [train_files[i] for i in range(n) if i not in val_idx]
    val = [train_files[i] for i in range(n) if i in val_idx]
    return subtrain, val


# ----------------------------
# Logging helper
# ----------------------------
class TeeLogger:
    def __init__(self):
        self.lines: list[str] = []

    def log(self, msg: str = ""):
        print(msg)
        self.lines.append(msg)


def ensure_writable_dir(target: Path, log: TeeLogger) -> Path:
    """
    Ensure target is a writable directory.
    If not writable (e.g., root-owned from sudo), fallback to result_YYYYMMDD_HHMMSS.
    """
    if target.exists() and target.is_file():
        raise FileExistsError(f"'{target}' exists as a FILE. Please rename/remove it.")

    target.mkdir(parents=True, exist_ok=True)

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


# ----------------------------
# Feature engineering (OSA-aware, no future leakage)
# ----------------------------
def _run_length_prev(flag: pd.Series) -> pd.Series:
    """
    flag: 0/1 Series
    Return consecutive-1 run length that ends at PREVIOUS minute (leak-free):
      runlen_prev[t] is the run length of flag at t-1 (consecutive ones up to t-1).
    """
    s = flag.shift(1).fillna(0).astype(int)
    grp = (s == 0).cumsum()
    runlen = s.groupby(grp).cumcount() + 1
    runlen[s == 0] = 0
    return runlen


def _minutes_since_last_true_prev(flag: pd.Series) -> pd.Series:
    """
    Minutes since the last 1 in flag, based on PAST only (use flag.shift(1)).
    If none yet, returns NaN.
    """
    s = flag.shift(1).fillna(0).astype(int)
    idx = np.arange(len(s), dtype=float)
    last = np.where(s.to_numpy() == 1, idx, np.nan)
    last = pd.Series(last).ffill().to_numpy()
    out = idx - last
    out[np.isnan(last)] = np.nan
    return pd.Series(out, index=flag.index)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stronger features tailored for sleep apnea:
    - Desaturation severity using MIN SpO2 (more sensitive than mean)
    - Event frequency & duration (ODI-like counts, run-length, time-since)
    - Rolling context (min/max/mean/std) for SpO2 and HR over past windows
    - Z-scores (normalize per subject based on past window)
    - Lagged responses (capture delayed HR response after SpO2 drop)
    - Non-linear transforms (squared terms) + interactions (drop x rise)
    All rolling features are computed with shift(1) to avoid using current minute in "past" stats.
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

    opt = ["rr_std_ms", "rr_min_ms", "rr_max_ms", "spo2_min_pct", "spo2_max_pct"]
    for c in opt:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("minute").reset_index(drop=True)

    # HR (bpm) from RR mean
    df["hr_mean_1m"] = 60000.0 / df["rr_mean_ms"]

    # Basic diffs (current - prev)
    df["spo2_diff_1m"] = df["spo2_mean_pct"].diff(1)
    df["hr_diff_1m"]   = df["hr_mean_1m"].diff(1)

    # "Acceleration" (diff of diff)
    df["spo2_accel"] = df["spo2_diff_1m"] - df["spo2_diff_1m"].shift(1)
    df["hr_accel"]   = df["hr_diff_1m"] - df["hr_diff_1m"].shift(1)

    # RR variability proxies
    if "rr_std_ms" in df.columns:
        df["rr_cv"] = df["rr_std_ms"] / df["rr_mean_ms"]
    if "rr_min_ms" in df.columns and "rr_max_ms" in df.columns:
        df["rr_range"] = df["rr_max_ms"] - df["rr_min_ms"]

    if "spo2_min_pct" in df.columns and "spo2_max_pct" in df.columns:
        df["spo2_range"] = df["spo2_max_pct"] - df["spo2_min_pct"]

    # Rolling context for SpO2 and HR (past only -> shift(1))
    for w in WINS:
        spo2_prev = df["spo2_mean_pct"].shift(1)
        hr_prev   = df["hr_mean_1m"].shift(1)

        df[f"spo2_roll_mean_{w}"] = spo2_prev.rolling(w, min_periods=w).mean()
        df[f"spo2_roll_std_{w}"]  = spo2_prev.rolling(w, min_periods=w).std()
        df[f"spo2_roll_min_{w}"]  = spo2_prev.rolling(w, min_periods=w).min()
        df[f"spo2_roll_max_{w}"]  = spo2_prev.rolling(w, min_periods=w).max()

        df[f"hr_roll_mean_{w}"] = hr_prev.rolling(w, min_periods=w).mean()
        df[f"hr_roll_std_{w}"]  = hr_prev.rolling(w, min_periods=w).std()
        df[f"hr_roll_min_{w}"]  = hr_prev.rolling(w, min_periods=w).min()
        df[f"hr_roll_max_{w}"]  = hr_prev.rolling(w, min_periods=w).max()

        # Z-scores (normalize by past window stats)
        df[f"spo2_z_{w}"] = (df["spo2_mean_pct"] - df[f"spo2_roll_mean_{w}"]) / df[f"spo2_roll_std_{w}"]
        df[f"hr_z_{w}"]   = (df["hr_mean_1m"]   - df[f"hr_roll_mean_{w}"])   / df[f"hr_roll_std_{w}"]

    # Median baseline (robust)
    for w in [5, 10, 20, 30]:
        df[f"spo2_base_med_{w}"] = df["spo2_mean_pct"].shift(1).rolling(w, min_periods=w).median()
        df[f"hr_base_med_{w}"]   = df["hr_mean_1m"].shift(1).rolling(w, min_periods=w).median()

    # Desaturation magnitude (mean-based and min-based)
    df["spo2_drop10_mean"] = df["spo2_base_med_10"] - df["spo2_mean_pct"]
    df["hr_rise10_mean"]   = df["hr_mean_1m"] - df["hr_base_med_10"]

    if "spo2_min_pct" in df.columns:
        df["spo2_drop10_min"] = df["spo2_base_med_10"] - df["spo2_min_pct"]
        drop_for_event = df["spo2_drop10_min"]
    else:
        drop_for_event = df["spo2_drop10_mean"]

    # Clip negative drops (we only care about "drop", not rise)
    df["spo2_drop10_pos"] = drop_for_event.clip(lower=0.0)
    df["spo2_drop10_pos_sq"] = df["spo2_drop10_pos"] ** 2

    # Hypoxemia flags (use min if available; else mean)
    spo2_min_like = df["spo2_min_pct"] if "spo2_min_pct" in df.columns else df["spo2_mean_pct"]
    df["spo2_le_90"] = (spo2_min_like <= 90.0).astype(int)
    df["spo2_le_92"] = (spo2_min_like <= 92.0).astype(int)

    # Desaturation event flags (3%/4% drop from baseline)
    df["desat3"] = (df["spo2_drop10_pos"] >= 3.0).astype(int)
    df["desat4"] = (df["spo2_drop10_pos"] >= 4.0).astype(int)

    # ODI-like frequency (past only)
    df["desat3_count_10"] = df["desat3"].shift(1).rolling(10, min_periods=10).sum()
    df["desat3_count_30"] = df["desat3"].shift(1).rolling(30, min_periods=30).sum()
    df["desat4_count_30"] = df["desat4"].shift(1).rolling(30, min_periods=30).sum()

    # "Area" of desaturation in past window (severity x duration proxy)
    df["desat_area_10"] = df["spo2_drop10_pos"].shift(1).rolling(10, min_periods=10).sum()
    df["desat_area_30"] = df["spo2_drop10_pos"].shift(1).rolling(30, min_periods=30).sum()

    # Duration / recency features (past only)
    df["desat3_runlen_prev"] = _run_length_prev(df["desat3"])
    df["mins_since_desat3_prev"] = _minutes_since_last_true_prev(df["desat3"])

    # HR response lags: add lags (past only)
    for k in [1, 2, 3]:
        df[f"spo2_drop10_pos_lag{k}"] = df["spo2_drop10_pos"].shift(k)
        df[f"hr_rise10_mean_lag{k}"]  = df["hr_rise10_mean"].shift(k)
        df[f"hr_diff_1m_lag{k}"]      = df["hr_diff_1m"].shift(k)

    # Interactions (linear model needs help capturing "AND" logic)
    df["drop_x_rise"] = df["spo2_drop10_pos"] * df["hr_rise10_mean"]
    df["desat3_x_rise"] = df["desat3"] * df["hr_rise10_mean"]
    df["spo2le90_x_rise"] = df["spo2_le_90"] * df["hr_rise10_mean"]

    # Shape features (simple)
    df["spo2_fall_then_rise"] = ((df["spo2_diff_1m"].shift(1) < 0) & (df["spo2_diff_1m"] > 0)).astype(int)
    df["hr_rise_then_fall"]   = ((df["hr_diff_1m"].shift(1) > 0) & (df["hr_diff_1m"] < 0)).astype(int)

    # Clean
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


# ----------------------------
# Build dataset from a list of files
# ----------------------------
def load_one(p: Path) -> pd.DataFrame:
    sid = subject_id_from_filename(p)
    g = detect_group(sid) or "UNK"
    df = pd.read_csv(p)
    df = add_features(df)
    df["subject_id"] = sid
    df["group"] = g
    df["source_file"] = p.name
    return df


def build_concat(files: List[Path], log: TeeLogger, tag: str) -> pd.DataFrame:
    dfs = []
    for p in files:
        try:
            dfs.append(load_one(p))
        except Exception as e:
            log.log(f"[WARN] failed to load {tag} {p.name}: {e}")
    if not dfs:
        raise RuntimeError(f"No {tag} data loaded.")
    return pd.concat(dfs, ignore_index=True)


# ----------------------------
# Train + Evaluate helpers
# ----------------------------
def make_model(C: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=6000,
            C=C,
            solver="lbfgs",
        )),
    ])



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default=".", help="*_merged.csv が置いてあるディレクトリ")
    ap.add_argument("--pattern", type=str, default="*_merged.csv")
    ap.add_argument("--label_col", type=str, default="label_1m")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--result_dir", type=str, default="result")
    ap.add_argument("--C", type=float, default=1.0, help="LogReg regularization strength (C)")
    ap.add_argument("--auto_C", action="store_true", help="TRAIN内の検証でCを自動選択（候補グリッド）")
    ap.add_argument("--val_frac", type=float, default=0.20, help="auto_* 用の検証ファイル割合（train filesのうち）")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    log = TeeLogger()
    result_dir = ensure_writable_dir(in_dir / args.result_dir, log)

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
    log.log(f"[INFO] in_dir         : {in_dir}")
    log.log(f"[INFO] pattern        : {args.pattern}")
    log.log(f"[INFO] seed           : {args.seed}")
    log.log(f"[INFO] result_dir     : {result_dir}")
    log.log(f"[INFO] fixed threshold: {FIXED_THRESHOLD}")
    log.log(f"[INFO] LogReg C       : {args.C}")
    log.log(f"[INFO] threshold_mode : fixed")
    log.log(f"[INFO] auto_C         : {args.auto_C}")
    log.log(f"[INFO] val_frac       : {args.val_frac}")

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

    log.log("")
    log.log("[INFO] loading train files ...")
    df_train_all = build_concat(train_files, log, "train")
    log.log("[INFO] loading test files ...")
    df_test_all = build_concat(test_files, log, "test")

    test_by_subject: Dict[str, pd.DataFrame] = {}
    for p in test_files:
        df = load_one(p)
        test_by_subject[df["subject_id"].iloc[0]] = df

    # Candidate features
    candidates = [
        "spo2_mean_pct", "hr_mean_1m", "spo2_min_pct", "spo2_max_pct", "spo2_range",
        "rr_mean_ms", "rr_std_ms", "rr_min_ms", "rr_max_ms", "rr_cv", "rr_range",
        "spo2_diff_1m", "hr_diff_1m", "spo2_accel", "hr_accel",
        "spo2_drop10_mean", "hr_rise10_mean", "spo2_drop10_pos", "spo2_drop10_pos_sq",
        "spo2_le_90", "spo2_le_92",
        "desat3", "desat4",
        "desat3_count_10", "desat3_count_30", "desat4_count_30",
        "desat_area_10", "desat_area_30",
        "desat3_runlen_prev", "mins_since_desat3_prev",
        "spo2_drop10_pos_lag1", "spo2_drop10_pos_lag2", "spo2_drop10_pos_lag3",
        "hr_rise10_mean_lag1", "hr_rise10_mean_lag2", "hr_rise10_mean_lag3",
        "hr_diff_1m_lag1", "hr_diff_1m_lag2", "hr_diff_1m_lag3",
        "drop_x_rise", "desat3_x_rise", "spo2le90_x_rise",
        "spo2_fall_then_rise", "hr_rise_then_fall",
    ]
    for w in [5, 10, 20, 30]:
        candidates += [
            f"spo2_roll_mean_{w}", f"spo2_roll_std_{w}", f"spo2_roll_min_{w}", f"spo2_roll_max_{w}",
            f"hr_roll_mean_{w}", f"hr_roll_std_{w}", f"hr_roll_min_{w}", f"hr_roll_max_{w}",
            f"spo2_z_{w}", f"hr_z_{w}",
        ]

    common_cols = set(df_train_all.columns) & set(df_test_all.columns)
    features = [c for c in candidates if c in common_cols]

    label_col = args.label_col
    if label_col not in common_cols:
        raise KeyError(f"label_col '{label_col}' not found.")

    for c in features + [label_col]:
        df_train_all[c] = pd.to_numeric(df_train_all[c], errors="coerce")
        df_test_all[c] = pd.to_numeric(df_test_all[c], errors="coerce")

    train_data = df_train_all.dropna(subset=features + [label_col]).copy()
    test_data_all = df_test_all.dropna(subset=features + [label_col]).copy()

    X_train = train_data[features]
    y_train = train_data[label_col].astype(int).to_numpy()

    X_test = test_data_all[features]
    y_test = test_data_all[label_col].astype(int).to_numpy()

    log.log("")
    log.log("============================================================")
    log.log(" Data summary")
    log.log("============================================================")
    log.log(f"[INFO] features used ({len(features)}): {features}")
    log.log(f"[INFO] train rows(after dropna): {len(train_data)}   label_counts={pd.Series(y_train).value_counts().to_dict()}")
    log.log(f"[INFO] test  rows(after dropna): {len(test_data_all)}    label_counts={pd.Series(y_test).value_counts().to_dict()}")

    chosen_C = args.C
    chosen_threshold = FIXED_THRESHOLD

    if False or args.auto_C:
        log.log("")
        log.log("============================================================")
        log.log(" Optional tuning on TRAIN-only validation split")
        log.log("============================================================")

        subtrain_files: List[Path] = []
        val_files: List[Path] = []
        for g in ["C", "D", "ND"]:
            g_train = [p for p in train_files if detect_group(subject_id_from_filename(p)) == g]
            st, va = split_train_into_train_val(g_train, seed=args.seed, val_frac=args.val_frac)
            subtrain_files += st
            val_files += va

        if len(val_files) == 0:
            log.log("[WARN] validation set is empty; skip tuning.")
        else:
            df_subtrain = build_concat(subtrain_files, log, "subtrain")
            df_val = build_concat(val_files, log, "val")

            for c in features + [label_col]:
                df_subtrain[c] = pd.to_numeric(df_subtrain[c], errors="coerce")
                df_val[c] = pd.to_numeric(df_val[c], errors="coerce")

            subtrain = df_subtrain.dropna(subset=features + [label_col]).copy()
            val = df_val.dropna(subset=features + [label_col]).copy()

            X_st = subtrain[features]
            y_st = subtrain[label_col].astype(int).to_numpy()
            X_va = val[features]
            y_va = val[label_col].astype(int).to_numpy()

            log.log(f"[TUNE] subtrain files: {len(subtrain_files)}  val files: {len(val_files)}")
            log.log(f"[TUNE] subtrain rows: {len(subtrain)}  val rows: {len(val)}")

            C_grid = [0.1, 0.3, 1.0, 3.0, 10.0] if args.auto_C else [args.C]
            best = {"acc": -1.0, "C": args.C, "thr": FIXED_THRESHOLD}

            for Ccand in C_grid:
                m = make_model(Ccand)
                m.fit(X_st, y_st)
                prob_va = m.predict_proba(X_va)[:, 1]

                thr = FIXED_THRESHOLD
                acc = accuracy_score(y_va, (prob_va >= thr).astype(int))

                if acc > best["acc"]:
                    best = {"acc": float(acc), "C": float(Ccand), "thr": float(thr)}

            chosen_C = best["C"]
            chosen_threshold = best["thr"]
            log.log(f"[TUNE] best val-accuracy: {best['acc']*100:.2f}%  C={chosen_C}  threshold={chosen_threshold:.3f}")

    model = make_model(chosen_C)
    model.fit(X_train, y_train)

    prob_all = model.predict_proba(X_test)[:, 1]
    pred_all = (prob_all >= chosen_threshold).astype(int)

    acc_all = accuracy_score(y_test, pred_all)
    correct_all = int((y_test == pred_all).sum())
    total_all = int(len(y_test))

    log.log("")
    log.log("============================================================")
    log.log(" Overall result (ALL TEST files merged)")
    log.log("============================================================")
    log.log(f"✅ 正答率(Accuracy): {acc_all*100:.2f}%  ({correct_all}/{total_all})")
    log.log(f"✅ threshold used   : {chosen_threshold:.3f}")
    log.log(f"✅ C used           : {chosen_C}")

    auc_all = safe_auc(y_test, prob_all)
    if auc_all is not None:
        log.log(f"📈 AUC(ROC): {auc_all:.4f}")
    else:
        log.log("[WARN] AUC could not be computed (single class in y_test).")

    cm = confusion_matrix(y_test, pred_all, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    log.log("")
    log.log("Confusion Matrix (rows=true, cols=pred)")
    log.log("          pred0   pred1")
    log.log(f"true0     {tn:5d}  {fp:5d}")
    log.log(f"true1     {fn:5d}  {tp:5d}")
    log.log("")
    log.log(classification_report(y_test, pred_all, digits=4))

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
        pred = (prob >= chosen_threshold).astype(int)

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
        out_df["threshold_used"] = chosen_threshold
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
            "threshold_used": chosen_threshold,
            "C_used": chosen_C,
        })

        log.log(f"[OK] {sid:>4s} ({g}) rows={len(df):4d}  acc={acc*100:6.2f}%  auc={'-' if auc is None else f'{auc:.4f}'}  -> {out_csv.name}")

    summary_path = result_dir / "test_summary_by_file.csv"
    if per_file_rows:
        pd.DataFrame(per_file_rows).sort_values(["group", "subject_id"]).to_csv(summary_path, index=False)
        log.log(f"[OK] saved per-file summary: {summary_path}")

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