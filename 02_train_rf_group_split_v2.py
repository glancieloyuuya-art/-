#!/usr/bin/env python3
# 02_train_rf_group_split_v2.py
# ------------------------------------------------------------
# Group-wise file split (C / D / ND): 2/3 train files, 1/3 test files (per group)
# Train a single RandomForestClassifier model, evaluate, and export TEST judgments
# per test file into a result/ folder as CSVs.
#
# ✅ What you get in <in_dir>/result/ :
#   - overall_result.txt                 : overall metrics + split info
#   - test_summary_by_file.csv           : metrics per test file (accuracy, AUC, confusion matrix)
#   - feature_importances.csv            : feature importances (Random Forest)
#   - <SUBJECT>_test_pred.csv            : row-wise predictions for each test file
#
# Usage:
#   cd /path/to/out
#   python 02_train_rf_group_split_v2.py
#
# Optional:
#   python 02_train_rf_group_split_v2.py --seed 42 --threshold 0.5 --save_test_pred \
#       --n_estimators 500 --max_depth 10 --min_samples_leaf 2 --max_features sqrt
# ------------------------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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

    # 数値化（必須）
    must = ["label_1m", "rr_mean_ms", "spo2_mean_pct"]
    for c in must:
        if c not in df.columns:
            raise KeyError(f"必要列 '{c}' がCSVにありません。columns={df.columns.tolist()}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 時系列順
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


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    # AUC needs both classes present
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default=".", help="*_merged.csv が置いてあるディレクトリ（デフォルト: カレント）")
    ap.add_argument("--pattern", type=str, default="*_merged.csv", help="読み込むファイルのglobパターン（デフォルト: *_merged.csv）")
    ap.add_argument("--label_col", type=str, default="label_1m", help="教師ラベル列（デフォルト: label_1m）")
    ap.add_argument("--seed", type=int, default=0, help="グループ内ファイル分割の乱数シード（デフォルト: 0）")
    ap.add_argument("--threshold", type=float, default=0.5, help="判定しきい値（デフォルト: 0.5）")
    ap.add_argument("--result_dir", type=str, default="result", help="結果を保存するフォルダ名（デフォルト: result）")
    ap.add_argument("--save_test_pred", action="store_true", help="各テスト被験者の予測CSVを保存（通常ON推奨）")

    # Random Forest hyperparams
    ap.add_argument("--n_estimators", type=int, default=500, help="木の本数（デフォルト: 500）")
    ap.add_argument("--max_depth", type=int, default=0, help="木の最大深さ。0なら制限なし（デフォルト: 0）")
    ap.add_argument("--min_samples_split", type=int, default=2, help="分割に必要な最小サンプル数（デフォルト: 2）")
    ap.add_argument("--min_samples_leaf", type=int, default=1, help="葉に必要な最小サンプル数（デフォルト: 1）")
    ap.add_argument("--max_features", type=str, default="sqrt", help="各分割で見る特徴量数（例: sqrt, log2, 0.5, 10）（デフォルト: sqrt）")
    ap.add_argument("--bootstrap", action="store_true", help="ブートストラップを使う（デフォルト: False）")
    ap.add_argument("--oob_score", action="store_true", help="OOBスコアを計算（bootstrap=Trueが必要）")
    ap.add_argument("--n_jobs", type=int, default=-1, help="並列数（デフォルト: -1）")

    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    if not in_dir.exists():
        raise FileNotFoundError(f"in_dir not found: {in_dir}")

    result_dir = ensure_dir(in_dir / args.result_dir)

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
    log.log(f"[INFO] threshold   : {args.threshold}")
    log.log(f"[INFO] result_dir  : {result_dir}")
    log.log(f"[INFO] RF params   : n_estimators={args.n_estimators}, max_depth={'None' if args.max_depth<=0 else args.max_depth}, "
            f"min_samples_split={args.min_samples_split}, min_samples_leaf={args.min_samples_leaf}, max_features={args.max_features}, "
            f"class_weight=balanced, n_jobs={args.n_jobs}, bootstrap={args.bootstrap}, oob_score={args.oob_score}")
    if skipped:
        log.log(f"[WARN] skipped (unknown group): {len(skipped)} files")

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
        df["source_file"] = p.name
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
    # keep per-subject DF for per-file export later
    test_by_subject: Dict[str, pd.DataFrame] = {}
    for p in test_files:
        try:
            df = load_one(p)
            sid = df["subject_id"].iloc[0]
            test_by_subject[sid] = df
        except Exception as e:
            log.log(f"[WARN] failed to load {p.name}: {e}")

    if not train_dfs or not test_by_subject:
        raise RuntimeError("No data loaded. Please check CSV contents/columns.")

    df_train_all = pd.concat(train_dfs, ignore_index=True)
    df_test_all = pd.concat(list(test_by_subject.values()), ignore_index=True)

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
    test_data_all  = df_test_all.dropna(subset=features + [label_col]).copy()

    X_train = train_data[features]
    y_train = train_data[label_col].astype(int)

    X_test_all = test_data_all[features]
    y_test_all = test_data_all[label_col].astype(int)

    log.log("")
    log.log("============================================================")
    log.log(" Data summary")
    log.log("============================================================")
    log.log(f"[INFO] features used ({len(features)}): {features}")
    log.log(f"[INFO] train rows(after dropna): {len(train_data)}   label_counts={y_train.value_counts().to_dict()}")
    log.log(f"[INFO] test  rows(after dropna): {len(test_data_all)}    label_counts={y_test_all.value_counts().to_dict()}")

    # ----------------------------
    # Train (Random Forest)
    # ----------------------------
    max_depth = None if args.max_depth <= 0 else int(args.max_depth)

    # RandomForestClassifier supports predict_proba and class_weight. :contentReference[oaicite:1]{index=1}
    model = RandomForestClassifier(
        n_estimators=int(args.n_estimators),
        random_state=int(args.seed),
        class_weight="balanced",
        n_jobs=int(args.n_jobs),
        max_depth=max_depth,
        min_samples_split=int(args.min_samples_split),
        min_samples_leaf=int(args.min_samples_leaf),
        max_features=args.max_features,
        bootstrap=bool(args.bootstrap),
        oob_score=bool(args.oob_score),
    )
    model.fit(X_train, y_train)

    if args.oob_score and args.bootstrap:
        try:
            log.log(f"[INFO] OOB score: {model.oob_score_:.4f}")
        except Exception:
            log.log("[WARN] OOB score requested but not available (check bootstrap=True and classification settings).")

    # Predict overall test
    prob_all = model.predict_proba(X_test_all)[:, 1]
    pred_all = (prob_all >= args.threshold).astype(int)

    # Metrics
    acc_all = accuracy_score(y_test_all, pred_all)
    correct_all = int((y_test_all.values == pred_all).sum())
    total_all = int(len(y_test_all))

    log.log("")
    log.log("============================================================")
    log.log(" Overall result (ALL TEST files merged)")
    log.log("============================================================")
    log.log(f"✅ 正答率(Accuracy): {acc_all*100:.2f}%  ({correct_all}/{total_all})")

    auc_all = safe_auc(y_test_all.values, prob_all)
    if auc_all is not None:
        log.log(f"📈 AUC(ROC): {auc_all:.4f}")
    else:
        log.log("[WARN] AUC could not be computed (single class in y_test).")

    cm_all = confusion_matrix(y_test_all, pred_all, labels=[0, 1])
    tn, fp, fn, tp = cm_all.ravel()
    log.log("")
    log.log("Confusion Matrix (rows=true, cols=pred)")
    log.log("          pred0   pred1")
    log.log(f"true0     {tn:5d}  {fp:5d}")
    log.log(f"true1     {fn:5d}  {tp:5d}")
    log.log("")
    log.log(classification_report(y_test_all, pred_all, digits=4))

    # ----------------------------
    # Export feature importances
    # ----------------------------
    try:
        fi = getattr(model, "feature_importances_", None)
        if fi is not None:
            fi_path = result_dir / "feature_importances.csv"
            pd.DataFrame({"feature": features, "importance": fi}).sort_values("importance", ascending=False).to_csv(fi_path, index=False)
            log.log(f"[OK] saved feature importances: {fi_path}")
    except Exception as e:
        log.log(f"[WARN] failed to export feature importances: {e}")

    # ----------------------------
    # Export per-test-file predictions CSV
    # ----------------------------
    log.log("============================================================")
    log.log(" Export TEST judgments per file -> result/*.csv")
    log.log("============================================================")

    per_file_rows = []

    for sid, df_raw in sorted(test_by_subject.items()):
        g = df_raw["group"].iloc[0]
        src = df_raw["source_file"].iloc[0]

        # numeric safety
        df_raw = df_raw.copy()
        for c in features + [label_col]:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

        # drop rows with NaN in required cols
        df = df_raw.dropna(subset=features + [label_col]).copy()
        if len(df) == 0:
            log.log(f"[WARN] {sid} ({g}) -> 0 rows after dropna, skipped export.")
            continue

        X = df[features]
        y = df[label_col].astype(int).to_numpy()

        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= args.threshold).astype(int)

        acc = float(accuracy_score(y, pred))
        auc = safe_auc(y, prob)

        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # save CSV per subject (row-wise)
        if args.save_test_pred or True:  # keep same behavior as your original script (always save)
            out_csv = result_dir / f"{sid}_test_pred.csv"
            # nice column order
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
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "threshold": args.threshold,
        })

        log.log(f"[OK] {sid:>4s} ({g}) rows={len(df):4d}  acc={acc*100:6.2f}%  auc={'-' if auc is None else f'{auc:.4f}'}  -> {sid}_test_pred.csv")

    # summary CSV
    summary_path = result_dir / "test_summary_by_file.csv"
    if per_file_rows:
        pd.DataFrame(per_file_rows).sort_values(["group", "subject_id"]).to_csv(summary_path, index=False)
        log.log(f"[OK] saved per-file summary: {summary_path}")
    else:
        log.log("[WARN] no per-file rows exported (maybe all files became empty after dropna).")

    # ----------------------------
    # Per-group accuracy on test set (quick)
    # ----------------------------
    log.log("")
    log.log("============================================================")
    log.log(" Group-wise accuracy on TEST (ALL test rows merged)")
    log.log("============================================================")
    test_data_all = test_data_all.copy()
    test_data_all["prob_1"] = prob_all
    test_data_all["pred"] = pred_all

    for g in ["C", "D", "ND"]:
        sub = test_data_all[test_data_all["group"] == g]
        if len(sub) == 0:
            log.log(f"[GROUP {g}] (no rows)")
            continue
        g_acc = accuracy_score(sub[label_col].astype(int), sub["pred"].astype(int))
        g_auc = safe_auc(sub[label_col].astype(int).to_numpy(), sub["prob_1"].to_numpy())
        log.log(f"[GROUP {g}] rows={len(sub):5d}  acc={g_acc*100:6.2f}%  auc={'-' if g_auc is None else f'{g_auc:.4f}'}  label_counts={sub[label_col].astype(int).value_counts().to_dict()}")

    # ----------------------------
    # Save model + overall log
    # ----------------------------
    out_model = in_dir / "rf_apnea_model.joblib"
    joblib.dump(model, out_model)
    log.log("")
    log.log(f"[OK] saved model: {out_model}")

    overall_log_path = result_dir / "overall_result.txt"
    overall_log_path.write_text("\n".join(log.lines) + "\n", encoding="utf-8")
    log.log(f"[OK] saved overall log: {overall_log_path}")

    log.log("============================================================")
    log.log("Done.")
    log.log("============================================================")


if __name__ == "__main__":
    main()
