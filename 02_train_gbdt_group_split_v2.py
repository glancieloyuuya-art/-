#!/usr/bin/env python3
# 02_train_gbdt_group_split_v2.py
# ------------------------------------------------------------
# Group-wise file split (C / D / ND): 2/3 train files, 1/3 test files (per group)
# Train a single Gradient Boosting Decision Tree model
# (LightGBM / XGBoost / CatBoost), evaluate, and export TEST judgments
# per test file into a result/ folder as CSVs.
#
# ✅ What you get in <in_dir>/result/ :
#   - overall_result.txt                 : overall metrics + split info
#   - test_summary_by_file.csv           : metrics per test file (accuracy, AUC, confusion matrix)
#   - <SUBJECT>_test_pred.csv            : row-wise predictions for each test file
#   - (optional) feature_importance.csv  : feature importances (if supported)
#
# Usage:
#   cd /path/to/out
#   python 02_train_gbdt_group_split_v2.py --algo lgbm
#
# Optional:
#   python 02_train_gbdt_group_split_v2.py --algo xgb --seed 42 --threshold 0.5
#   python 02_train_gbdt_group_split_v2.py --algo cat --no_balanced_weight
#
# ------------------------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

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


def compute_balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    """
    sklearnの 'balanced' 相当:
      weight_c = n_samples / (n_classes * count_c)
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    w_map = {c: n / (n_classes * cnt) for c, cnt in zip(classes, counts)}
    return np.array([w_map[int(v)] for v in y], dtype=float)


# ----------------------------
# Model builder
# ----------------------------
def build_model(algo: str, seed: int, args) -> object:
    algo = algo.lower()

    if algo == "lgbm":
        try:
            from lightgbm import LGBMClassifier
        except Exception as e:
            raise ImportError("LightGBM が未インストールです: pip install -U lightgbm") from e

        # LightGBM default-ish params (調整はCLIで)
        return LGBMClassifier(
            objective="binary",
            n_estimators=args.n_estimators if args.n_estimators is not None else 800,
            learning_rate=args.learning_rate if args.learning_rate is not None else 0.05,
            num_leaves=args.num_leaves if args.num_leaves is not None else 31,
            max_depth=args.max_depth if args.max_depth is not None else -1,
            subsample=args.subsample if args.subsample is not None else 0.8,
            colsample_bytree=args.colsample_bytree if args.colsample_bytree is not None else 0.8,
            reg_alpha=args.reg_alpha if args.reg_alpha is not None else 0.0,
            reg_lambda=args.reg_lambda if args.reg_lambda is not None else 0.0,
            random_state=seed,
            n_jobs=args.n_jobs,
        )

    if algo == "xgb":
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise ImportError("XGBoost が未インストールです: pip install -U xgboost") from e

        return XGBClassifier(
            objective="binary:logistic",
            n_estimators=args.n_estimators if args.n_estimators is not None else 1000,
            learning_rate=args.learning_rate if args.learning_rate is not None else 0.03,
            max_depth=args.max_depth if args.max_depth is not None else 4,
            subsample=args.subsample if args.subsample is not None else 0.8,
            colsample_bytree=args.colsample_bytree if args.colsample_bytree is not None else 0.8,
            reg_alpha=args.reg_alpha if args.reg_alpha is not None else 0.0,
            reg_lambda=args.reg_lambda if args.reg_lambda is not None else 1.0,
            min_child_weight=args.min_child_weight if args.min_child_weight is not None else 1.0,
            gamma=args.gamma if args.gamma is not None else 0.0,
            tree_method=args.tree_method,
            random_state=seed,
            n_jobs=args.n_jobs,
            eval_metric="logloss",
        )

    if algo == "cat":
        try:
            from catboost import CatBoostClassifier
        except Exception as e:
            raise ImportError("CatBoost が未インストールです: pip install -U catboost") from e

        # CatBoost は iterations/depth/l2_leaf_reg など名称が違う
        return CatBoostClassifier(
            loss_function="Logloss",
            iterations=args.n_estimators if args.n_estimators is not None else 2000,
            learning_rate=args.learning_rate if args.learning_rate is not None else 0.03,
            depth=args.max_depth if args.max_depth is not None else 6,
            l2_leaf_reg=args.l2_leaf_reg if args.l2_leaf_reg is not None else 3.0,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=args.n_jobs if args.n_jobs is not None else -1,
        )

    raise ValueError(f"Unknown algo: {algo} (use lgbm/xgb/cat)")


def get_feature_importance(model: object, feature_names: list[str]) -> Optional[pd.DataFrame]:
    """
    可能なら feature importance を DataFrame で返す。
    """
    try:
        # LightGBM / XGBoost sklearn wrapper: feature_importances_
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=float)
            return pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)

        # CatBoost: get_feature_importance()
        if hasattr(model, "get_feature_importance"):
            imp = np.asarray(model.get_feature_importance(), dtype=float)
            return pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)
    except Exception:
        return None

    return None


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", type=str, default="lgbm", choices=["lgbm", "xgb", "cat"],
                    help="学習アルゴリズム: lgbm (LightGBM) / xgb (XGBoost) / cat (CatBoost)")
    ap.add_argument("--in_dir", type=str, default=".", help="*_merged.csv が置いてあるディレクトリ（デフォルト: カレント）")
    ap.add_argument("--pattern", type=str, default="*_merged.csv", help="読み込むファイルのglobパターン（デフォルト: *_merged.csv）")
    ap.add_argument("--label_col", type=str, default="label_1m", help="教師ラベル列（デフォルト: label_1m）")
    ap.add_argument("--seed", type=int, default=0, help="グループ内ファイル分割の乱数シード（デフォルト: 0）")
    ap.add_argument("--threshold", type=float, default=0.5, help="判定しきい値（デフォルト: 0.5）")
    ap.add_argument("--result_dir", type=str, default="result", help="結果を保存するフォルダ名（デフォルト: result）")

    # 予測CSVは基本保存（元スクリプトが実質 always-save だったため）
    ap.add_argument("--no_save_test_pred", action="store_true", help="各テスト被験者の予測CSV保存を無効化")

    # 不均衡対策（3手法で揃う sample_weight をデフォルトON）
    ap.add_argument("--no_balanced_weight", action="store_true", help="balanced sample_weight を使わない")

    # Common-ish hyperparams (任意)
    ap.add_argument("--n_estimators", type=int, default=None, help="木の本数/iterations（未指定ならalgo別デフォルト）")
    ap.add_argument("--learning_rate", type=float, default=None, help="学習率（未指定ならalgo別デフォルト）")
    ap.add_argument("--max_depth", type=int, default=None, help="木の深さ（XGB/CatBoost中心。LGBMはmax_depth）")
    ap.add_argument("--num_leaves", type=int, default=None, help="(LightGBM) num_leaves")
    ap.add_argument("--subsample", type=float, default=None, help="subsample")
    ap.add_argument("--colsample_bytree", type=float, default=None, help="colsample_bytree")
    ap.add_argument("--reg_alpha", type=float, default=None, help="L1正則化")
    ap.add_argument("--reg_lambda", type=float, default=None, help="L2正則化")

    # XGBoost extras
    ap.add_argument("--min_child_weight", type=float, default=None, help="(XGB) min_child_weight")
    ap.add_argument("--gamma", type=float, default=None, help="(XGB) gamma")
    ap.add_argument("--tree_method", type=str, default="hist", help="(XGB) tree_method (default: hist)")

    # CatBoost extras
    ap.add_argument("--l2_leaf_reg", type=float, default=None, help="(CatBoost) l2_leaf_reg")

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
    log.log("  + GBDT model (LightGBM / XGBoost / CatBoost)")
    log.log("============================================================")
    log.log(f"[INFO] algo        : {args.algo}")
    log.log(f"[INFO] in_dir      : {in_dir}")
    log.log(f"[INFO] pattern     : {args.pattern}")
    log.log(f"[INFO] seed        : {args.seed}")
    log.log(f"[INFO] threshold   : {args.threshold}")
    log.log(f"[INFO] result_dir  : {result_dir}")
    log.log(f"[INFO] balanced_weight : {not args.no_balanced_weight}")
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

    # Drop NaN rows（元スクリプト踏襲）
    train_data = df_train_all.dropna(subset=features + [label_col]).copy()
    test_data_all = df_test_all.dropna(subset=features + [label_col]).copy()

    X_train = train_data[features]
    y_train = train_data[label_col].astype(int).to_numpy()

    X_test_all = test_data_all[features]
    y_test_all = test_data_all[label_col].astype(int).to_numpy()

    log.log("")
    log.log("============================================================")
    log.log(" Data summary")
    log.log("============================================================")
    log.log(f"[INFO] features used ({len(features)}): {features}")
    log.log(f"[INFO] train rows(after dropna): {len(train_data)}   label_counts={pd.Series(y_train).value_counts().to_dict()}")
    log.log(f"[INFO] test  rows(after dropna): {len(test_data_all)}    label_counts={pd.Series(y_test_all).value_counts().to_dict()}")

    # ----------------------------
    # Train (GBDT)
    # ----------------------------
    model = build_model(args.algo, seed=args.seed, args=args)

    sample_weight = None
    if not args.no_balanced_weight:
        sample_weight = compute_balanced_sample_weight(y_train)

    # sklearn互換 fit を基本にする（3方式で揃える）
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    # Predict overall test
    prob_all = model.predict_proba(X_test_all)[:, 1]
    pred_all = (prob_all >= args.threshold).astype(int)

    # Metrics
    acc_all = accuracy_score(y_test_all, pred_all)
    correct_all = int((y_test_all == pred_all).sum())
    total_all = int(len(y_test_all))

    log.log("")
    log.log("============================================================")
    log.log(" Overall result (ALL TEST files merged)")
    log.log("============================================================")
    log.log(f"✅ 正答率(Accuracy): {acc_all*100:.2f}%  ({correct_all}/{total_all})")

    auc_all = safe_auc(y_test_all, prob_all)
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
    # Export per-test-file predictions CSV
    # ----------------------------
    log.log("============================================================")
    log.log(" Export TEST judgments per file -> result/*.csv")
    log.log("============================================================")

    per_file_rows = []
    save_test_pred = not args.no_save_test_pred

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
        if save_test_pred:
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
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "threshold": args.threshold,
            "algo": args.algo,
        })

        log.log(f"[OK] {sid:>4s} ({g}) rows={len(df):4d}  acc={acc*100:6.2f}%  auc={'-' if auc is None else f'{auc:.4f}'}"
                f"  -> {sid}_test_pred.csv" if save_test_pred else "")

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
    tmp = test_data_all.copy()
    tmp["prob_1"] = prob_all
    tmp["pred"] = pred_all

    for g in ["C", "D", "ND"]:
        sub = tmp[tmp["group"] == g]
        if len(sub) == 0:
            log.log(f"[GROUP {g}] (no rows)")
            continue
        g_acc = accuracy_score(sub[label_col].astype(int), sub["pred"].astype(int))
        g_auc = safe_auc(sub[label_col].astype(int).to_numpy(), sub["prob_1"].to_numpy())
        log.log(f"[GROUP {g}] rows={len(sub):5d}  acc={g_acc*100:6.2f}%  auc={'-' if g_auc is None else f'{g_auc:.4f}'}"
                f"  label_counts={sub[label_col].astype(int).value_counts().to_dict()}")

    # ----------------------------
    # Save model + feature importance + overall log
    # ----------------------------
    model_out = in_dir / f"gbdt_apnea_model_{args.algo}.joblib"
    joblib.dump(model, model_out)
    log.log("")
    log.log(f"[OK] saved model: {model_out}")

    # feature importance
    fi = get_feature_importance(model, features)
    if fi is not None:
        fi_path = result_dir / "feature_importance.csv"
        fi.to_csv(fi_path, index=False)
        log.log(f"[OK] saved feature importances: {fi_path}")
    else:
        log.log("[INFO] feature importance not available for this model/config.")

    overall_log_path = result_dir / "overall_result.txt"
    overall_log_path.write_text("\n".join(log.lines) + "\n", encoding="utf-8")
    log.log(f"[OK] saved overall log: {overall_log_path}")

    log.log("============================================================")
    log.log("Done.")
    log.log("============================================================")


if __name__ == "__main__":
    main()
