# merge_to_csv_C1.py (fixed)
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_mat_numeric_1d(mat_path: Path, preferred_keys=None) -> np.ndarray:
    """
    .mat から「1次元の数値配列」をなるべく自動で取り出す（セル/struct対応）
    - preferred_keys があればその順で探す
    - なければ '__' 以外のキーを走査して最長の数値配列を採用
    """
    d = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in d.keys() if not k.startswith("__")]

    # scipy の mat_struct を環境差吸収して import
    try:
        from scipy.io.matlab import mat_struct  # type: ignore
    except Exception:
        from scipy.io.matlab._mio5_params import mat_struct  # type: ignore

    def to_1d_numeric(a):
        a = np.asarray(a)
        if a.dtype.kind not in "iuf":  # int/uint/float 以外は除外
            return None
        a = a.reshape(-1).astype(float)
        a = a[np.isfinite(a)]
        return a if a.size else None

    def collect_numeric(obj):
        out = []
        if obj is None:
            return out

        # 直接数値配列
        arr = to_1d_numeric(obj)
        if arr is not None:
            out.append(arr)
            return out

        # MATLABセル相当（object ndarray）
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            for item in obj.ravel():
                out.extend(collect_numeric(item))
            return out

        # dict
        if isinstance(obj, dict):
            for v in obj.values():
                out.extend(collect_numeric(v))
            return out

        # MATLAB struct
        if isinstance(obj, mat_struct):
            if hasattr(obj, "_fieldnames") and obj._fieldnames:
                for f in obj._fieldnames:
                    out.extend(collect_numeric(getattr(obj, f)))
            return out

        return out

    def pick_from_key(keyname):
        if keyname not in d:
            return None
        arrays = collect_numeric(d[keyname])
        if not arrays:
            return None
        arrays.sort(key=lambda x: x.size, reverse=True)
        print(f"[INFO] {mat_path.name}: picked key = '{keyname}', len = {arrays[0].size}")
        return arrays[0]

    # 1) preferred_keys を優先
    if preferred_keys:
        for k in preferred_keys:
            arr = pick_from_key(k)
            if arr is not None:
                return arr

    # 2) 全キーから探索して最長を採用
    candidates = []
    for k in keys:
        arr = pick_from_key(k)
        if arr is not None:
            candidates.append((k, arr))

    if not candidates:
        raise RuntimeError(f"{mat_path.name} に数値配列が見つかりませんでした。 keys={keys}")

    candidates.sort(key=lambda x: x[1].size, reverse=True)
    chosen_key, chosen_arr = candidates[0]
    print(f"[INFO] {mat_path.name}: auto-picked key = '{chosen_key}', len = {chosen_arr.size}")
    return chosen_arr


def load_required_key_1d(mat_path: Path, key: str) -> np.ndarray:
    """LABELSみたいにキーが確定してるものを安全に読み出す"""
    d = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if key not in d:
        keys = [k for k in d.keys() if not k.startswith("__")]
        raise KeyError(f"{mat_path.name}: '{key}' not found. keys={keys}")
    a = np.asarray(d[key]).reshape(-1)
    if a.dtype.kind not in "iuf":
        raise TypeError(f"{mat_path.name}: '{key}' is not numeric (dtype={a.dtype})")
    a = a.astype(float)
    a = a[np.isfinite(a)]
    return a


def series_to_per_min_mean(x: np.ndarray, n_minutes: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size < n_minutes:
        return np.full(n_minutes, np.nan)
    chunks = np.array_split(x, n_minutes)
    return np.array([np.nanmean(c) if c.size else np.nan for c in chunks], dtype=float)


def rr_to_hr_per_min(rr_ms: np.ndarray, n_minutes: int) -> np.ndarray:
    rr_ms = np.asarray(rr_ms, dtype=float)
    med = np.nanmedian(rr_ms)
    if med < 10:  # 秒っぽいならmsへ
        rr_ms = rr_ms * 1000.0

    rr_ms = rr_ms[(rr_ms > 200) & (rr_ms < 3000)]
    if rr_ms.size < 10:
        return np.full(n_minutes, np.nan)

    t_sec = np.cumsum(rr_ms) / 1000.0
    hr_inst = 60000.0 / rr_ms
    minute_idx = (t_sec // 60).astype(int)

    df = pd.DataFrame({"minute": minute_idx, "hr": hr_inst})
    hr_min = df.groupby("minute")["hr"].mean()

    out = np.full(n_minutes, np.nan)
    for m, v in hr_min.items():
        m = int(m)  # ★Pylance警告対策＋安全化
        if 0 <= m < n_minutes:
            out[m] = float(v)
    return out


# ==== ここだけ自分のファイル名に合わせる ====
LABELS = Path("LABELS/D1.mat")
SAT    = Path("SAT/D1.mat")
RR     = Path("RR/D1.mat")
OUTCSV = Path("D1_merged_1min.csv")

# 1分ラベル（確定）
label_1m = load_required_key_1d(LABELS, "salida_man_1m").astype(int)
n_minutes = len(label_1m)

# 30秒ラベル
label_30s = load_required_key_1d(LABELS, "salida_man").astype(int)
label_30s = label_30s[: n_minutes * 2]
label_30s_first  = label_30s[0::2]
label_30s_second = label_30s[1::2]

# SpO2（SAT）→ 1分平均（SATキーは優先して探す）
sat_raw = load_mat_numeric_1d(SAT, preferred_keys=["SAT", "sat", "SpO2", "SPO2", "spo2"])
spo2_mean_1m = series_to_per_min_mean(sat_raw, n_minutes)

# RR → HR（RRキーは候補で探す：あなたの例を最優先）
rr_raw = load_mat_numeric_1d(RR, preferred_keys=["RR_notch_abs_pr_ada", "RR", "rr", "RRI", "rri", "rr_ms"])
hr_mean_1m = rr_to_hr_per_min(rr_raw, n_minutes)

# 1つのテーブルに結合
df = pd.DataFrame({
    "minute": np.arange(n_minutes, dtype=int),
    "label_1m": label_1m,
    "label_30s_first": label_30s_first,
    "label_30s_second": label_30s_second,
    "spo2_mean_1m": spo2_mean_1m,
    "hr_mean_1m": hr_mean_1m,
})

df.to_csv(OUTCSV, index=False)
print("saved:", OUTCSV.resolve())
print(df.head())
