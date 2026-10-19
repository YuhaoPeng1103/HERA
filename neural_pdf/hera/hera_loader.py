# -*- coding: utf-8 -*-
"""
HERA combined NC e+p reduced cross sections (sqrt(s)=318 GeV)
=============================================================
数据：HEPData ins1377206, Table 1 (EPJC 75 (2015) 580, arXiv:1506.06042)
列：Q2, x, sigma_r, stat+- , uncor+-, cor+-, rel+-, gamma p+-, had+-,
    1+-, 2+-, 3+-, 4+-  （不确定性均为相对百分比）

协方差：
    C_ij = delta_ij (stat_i^2 + uncor_i^2) + sum_k d_i^k d_j^k
其中 k = cor, rel, gamma p, had, 1..4 共 8 个关联源；
d_i^k 取 CSV 中 "+" 列的带符号绝对值（百分比 × sigma_i）。

LO 演示近似（论文中必须明确标注）：
    sigma_r,NC ≈ F2   （忽略 FL 与 gamma-Z 干涉；只在低 Q2 子集使用）
"""
import argparse
import csv
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.abspath(os.path.join(
    _HERE, "..", "data", "hera_nc_eplus_318.csv"))

CORR_NAMES = ["cor", "rel", "gammap", "had", "1", "2", "3", "4"]
COLS = {"cor": 7, "rel": 9, "gammap": 11, "had": 13,
        "1": 15, "2": 17, "3": 19, "4": 21}


def pct(s):
    s = s.strip().replace("%", "")
    return float(s) / 100.0 if s else 0.0


def load(path=DEFAULT_CSV, q2max=10.0, q2min=0.0, xmin=1e-4, xmax=0.65):
    rows = []
    with open(path, encoding="utf-8") as f:
        lines = [l for l in f if l.strip() and not l.startswith("#")]
    reader = csv.reader(lines)
    header = next(reader)
    for row in reader:
        if len(row) < 23:
            continue
        try:
            q2 = float(row[0]); x = float(row[1]); y = float(row[2])
        except ValueError:
            continue
        if not (q2min <= q2 <= q2max and xmin <= x <= xmax):
            continue
        stat = pct(row[3]) * y
        uncor = pct(row[5]) * y
        shifts = {k: pct(row[c]) * y for k, c in COLS.items()}
        rows.append((q2, x, y, stat, uncor, shifts))

    q2 = np.array([r[0] for r in rows])
    x = np.array([r[1] for r in rows])
    y = np.array([r[2] for r in rows])
    stat = np.array([r[3] for r in rows])
    uncor = np.array([r[4] for r in rows])
    D = np.stack([np.array([r[5][k] for r in rows]) for k in CORR_NAMES], axis=1)
    C = np.diag(stat ** 2 + uncor ** 2) + D @ D.T
    return dict(q2=q2, x=x, y=y, stat=stat, uncor=uncor, D=D,
                corr_names=CORR_NAMES, C=C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--q2max", type=float, default=10.0)
    ap.add_argument("--q2min", type=float, default=0.0)
    ap.add_argument("--xmin", type=float, default=1e-4)
    ap.add_argument("--xmax", type=float, default=0.65)
    ap.add_argument("--out", default=os.path.join(
        os.getcwd(), "outputs", "hera_selected.npz"))
    args = ap.parse_args()
    d = load(args.csv, args.q2max, args.q2min, args.xmin, args.xmax)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **d)
    sig = np.sqrt(np.diag(d["C"]))
    print(f"选取 {len(d['y'])} 个点：Q2∈[{d['q2'].min():g},{d['q2'].max():g}], "
          f"x∈[{d['x'].min():g},{d['x'].max():g}]")
    print(f"相对总误差中位数 = {100*np.median(sig/d['y']):.1f}%  "
          f"范围 {100*np.min(sig/d['y']):.1f}–{100*np.max(sig/d['y']):.1f}%")
    print(f"sigma_r 范围 {d['y'].min():.3f}–{d['y'].max():.3f}")
    print(f"协方差条件数 = {np.linalg.cond(d['C']):.2e}")
    print(f"已保存：{args.out}")


if __name__ == "__main__":
    main()
