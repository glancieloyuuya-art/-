# merge_mat_to_csv.py
# 使い方例:
# python merge_mat_to_csv.py --labels LABELS_C1.mat --mat1 RR_C1.mat --mat2 SAT_C1.mat --out merged.csv --mode 1hz
#
# mode:
#   1hz   : 1行=1秒（おすすめ。RRとSpO2を同じ時間軸にできる）
#   minute: 1行=1分（特徴量だけ出す。軽い）

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


def _primary_var(mat: dict) -> tuple[str, np.ndarray]:
    keys = [k for k in mat.keys() if not k.startswith("__")]
    if not keys:
        raise ValueError("MAT内に変数が見つかりませんでした。")
    if len(keys) > 1:
        # このデータは基本1変数なので、迷ったら先頭を使う
        # 必要ならここを変えてください
        key = keys[0]
    else:
        key = keys[0]
    return key, np.array(mat[key])


def _cell_to_list(cell_arr: np.ndarray) -> list[np.ndarray]:
    cell_arr = np.array(cell_arr)
    if cell_arr.ndim == 2:
        if cell_arr.shape[0] == 1:
            items = [cell_arr[0, i] for i in range(cell_arr.shape[1])]
        elif cell_arr.shape[1] == 1:
            items = [cell_arr[i, 0] for i in range(cell_arr.shape[0])]
        else:
            raise ValueError(f"想定外のcell shape: {cell_arr.shape}")
    elif cell_arr.ndim == 1:
        items = list(cell_arr)
    else:
        raise ValueError(f"想定外のcell ndim: {cell_arr.ndim}")

    out = []
    for x in items:
        out.append(np.array(x, dtype=float).ravel())
    return out


def _detect_type_by_len(series_list: list[np.ndarray]) -> str:
    lens = [a.size for a in series_list if a.size > 0]
    if not lens:
        return "unknown"
    med = float(np.median(lens))
    # このデータだと SpO2 は 15000/分 (=250Hz) でかなり大きい
    return "spo2_like" if med >= 10000 else "rr_like"


def _bin_to_60s_mean(x: np.ndarray) -> np.ndarray:
    """
    1分ぶんの系列 x を 0..59秒に平均ビン詰めして 60個にする
    時間軸は「1分間に均等に並んでいる」と仮定（RRの補間系列向け）
    """
    x = x.ravel()
    L = x.size
    if L == 0:
        return np.full(60, np.nan)

    t = np.linspace(0, 60, L, endpoint=False)
    sec_idx = np.floor(t).astype(int)
    out = np.full(60, np.nan)
    for sec in range(60):
        m = sec_idx == sec
        if m.any():
            out[sec] = float(np.mean(x[m]))
    return out


def _spo2_to_60s_mean(spo2: np.ndarray) -> np.ndarray:
    """
    SpO2が 250Hz(=15000/分) なら 250サンプル=1秒で平均
    それ以外でも「サンプル数/60」を使って秒ごと平均
    """
    spo2 = spo2.ravel()
    if spo2.size == 0:
        return np.full(60, np.nan)

    if spo2.size == 15000:
        return spo2.reshape(60, 250).mean(axis=1)

    fs = spo2.size / 60.0
    out = []
    for sec in range(60):
        s = int(sec * fs)
        e = int((sec + 1) * fs)
        out.append(float(np.mean(spo2[s:e])) if e > s else np.nan)
    return np.array(out, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="LABELS_*.mat のパス")
    ap.add_argument("--mat1", required=True, help="RR or SAT の .mat（どっちでもOK）")
    ap.add_argument("--mat2", required=True, help="RR or SAT の .mat（どっちでもOK）")
    ap.add_argument("--out", required=True, help="出力CSVパス")
    ap.add_argument("--mode", choices=["1hz", "minute"], default="1hz")
    args = ap.parse_args()

    labels = loadmat(args.labels)
    if "salida_man_1m" not in labels or "salida_man" not in labels:
        raise ValueError("LABELS mat に salida_man_1m / salida_man が見つかりませんでした。")

    lab_1m_all = labels["salida_man_1m"].ravel().astype(int)
    lab_30s_all = labels["salida_man"].ravel().astype(int)

    m1 = loadmat(args.mat1)
    m2 = loadmat(args.mat2)

    name1, var1 = _primary_var(m1)
    name2, var2 = _primary_var(m2)

    list1 = _cell_to_list(var1)
    list2 = _cell_to_list(var2)

    t1 = _detect_type_by_len(list1)
    t2 = _detect_type_by_len(list2)

    if t1 == t2:
        raise ValueError(
            f"mat1({name1}) と mat2({name2}) の判定が同じでした: {t1}. "
            "中身が想定と違うかもしれません。"
        )

    if t1 == "spo2_like":
        spo2_list, rr_list = list1, list2
    else:
        rr_list, spo2_list = list1, list2

    Nseg = min(len(rr_list), len(spo2_list))  # ふつう316
    rr_list = rr_list[:Nseg]
    spo2_list = spo2_list[:Nseg]

    # ラベルをセグメント数に合わせる（多い分は切る）
    lab_1m = lab_1m_all[:Nseg]
    lab_30s = lab_30s_all[: 2 * Nseg]

    # 両方が空じゃないセグメントだけ使う（あなたのC1だと先頭2+末尾2が空）
    valid = [i for i in range(Nseg) if (rr_list[i].size > 0 and spo2_list[i].size > 0)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "minute":
        rows = []
        for seg_idx in valid:
            rr = rr_list[seg_idx]
            spo2 = spo2_list[seg_idx]
            rows.append(
                {
                    "segment_index": seg_idx,
                    "label_1m": int(lab_1m[seg_idx]),
                    "label_30s_first": int(lab_30s[2 * seg_idx + 0]),
                    "label_30s_second": int(lab_30s[2 * seg_idx + 1]),
                    "rr_mean_ms": float(np.mean(rr)),
                    "rr_std_ms": float(np.std(rr)),
                    "rr_min_ms": float(np.min(rr)),
                    "rr_max_ms": float(np.max(rr)),
                    "spo2_mean_pct": float(np.mean(spo2)),
                    "spo2_min_pct": float(np.min(spo2)),
                    "spo2_max_pct": float(np.max(spo2)),
                    "rr_samples_in_minute": int(rr.size),
                    "spo2_samples_in_minute": int(spo2.size),
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        print(f"[OK] minute CSV saved: {out_path} (rows={len(df)})")
        return

    # mode == 1hz（1行=1秒）
    rows = []
    minute_from_start = 0
    for seg_idx in valid:
        rr_sec = _bin_to_60s_mean(rr_list[seg_idx])
        spo2_sec = _spo2_to_60s_mean(spo2_list[seg_idx])

        label_1m = int(lab_1m[seg_idx])
        label_30s_pair = lab_30s[2 * seg_idx : 2 * seg_idx + 2]

        for sec in range(60):
            label_30s = int(label_30s_pair[0] if sec < 30 else label_30s_pair[1])
            rows.append(
                {
                    "minute_from_start": minute_from_start,
                    "segment_index": seg_idx,
                    "second": sec,
                    "t_sec": minute_from_start * 60 + sec,
                    "rr_ms": float(rr_sec[sec]) if np.isfinite(rr_sec[sec]) else np.nan,
                    "spo2_pct": float(spo2_sec[sec]) if np.isfinite(spo2_sec[sec]) else np.nan,
                    "label_1m": label_1m,
                    "label_30s": label_30s,
                    "rr_samples_in_minute": int(rr_list[seg_idx].size),
                    "spo2_samples_in_minute": int(spo2_list[seg_idx].size),
                }
            )
        minute_from_start += 1

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[OK] 1Hz CSV saved: {out_path} (rows={len(df)})")


if __name__ == "__main__":
    main()
