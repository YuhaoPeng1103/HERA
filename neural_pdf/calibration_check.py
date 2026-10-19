# -*- coding: utf-8 -*-
"""
误差带校准检查（M0.5）
======================
重要：closure 覆盖率必须区分两个对象

* **单个 replica** 相对 ensemble 带：检验误差模型本身是否校准
  （这是"误差棒是否可信"的正确检验）；
* **ensemble 均值** 相对 ensemble 带：均值是 R 个拟合的平均，误差天然更小，
  覆盖率会偏高（保守），不能用来判定误差带过宽。

本脚本用保存的 f0_replicas 重新演化到各 Q2，逐 replica 计算数据点覆盖率。
"""
import argparse
import json
import os
import sys

import numpy as np

from . import model as nmod
from . import multidata as mdata
from . import pseudodata as pdata


def coverage(z, k=1.0):
    return float(np.mean(np.abs(z) <= k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(
        os.getcwd(), "outputs", "replicas_m05.npz"))
    ap.add_argument("--out", default=os.path.join(
        os.getcwd(), "outputs", "calibration_check.json"))
    args = ap.parse_args()

    x, w, wx, mats, pdf0 = pdata.build()
    datasets, _ = mdata.build_datasets(x, w, mats, pdf0)
    q2_keys = sorted(set(float(q) for d in datasets.values() for q in d["q2"]))
    U = nmod.evolution_operators(x, w, mats, q2_keys)

    d = np.load(args.npz)
    f0 = d["f0_replicas"]                 # (R,4,N)
    truth0 = d["truth0"]
    R = f0.shape[0]
    xmask = (x >= mdata.X_BINS.min()) & (x <= mdata.X_BINS.max())

    # ---- Q0 PDF ----
    s0 = f0.std(axis=0)
    z_rep0 = (f0 - truth0[None]) / np.maximum(s0[None], 1e-30)
    z_mean0 = (f0.mean(axis=0) - truth0) / np.maximum(s0, 1e-30)
    flavors = ["Vu", "Vd", "S", "g"]
    q0 = {
        "single_replica": {
            "coverage_1sigma": coverage(z_rep0[:, :, xmask]),
            "coverage_90": coverage(z_rep0[:, :, xmask], 1.6449),
            "rms_z": float(np.sqrt(np.mean(z_rep0[:, :, xmask] ** 2))),
            "per_flavor_1sigma": {
                flavors[k]: coverage(z_rep0[:, k, xmask]) for k in range(4)},
        },
        "ensemble_mean": {
            "coverage_1sigma": coverage(z_mean0[:, xmask]),
            "coverage_90": coverage(z_mean0[:, xmask], 1.6449),
        },
    }

    # ---- 数据点：逐 replica 重新演化并插值 ----
    data = {}
    for name, dd in datasets.items():
        preds = np.zeros((R, len(dd["y"])))
        for r in range(R):
            Vu, Vd, S, g = f0[r]
            state0 = np.concatenate([Vu, Vd, Vu + Vd + 4.0 * S, g])
            for q in dd["q2"]:
                st = (U[float(q)] @ state0).reshape(4, len(x))
                Vu2, Vd2, Sig2, g2 = st
                S2 = (Sig2 - Vu2 - Vd2) / 4.0
                obs = mdata.observables_np(
                    {"Vu": Vu2, "Vd": Vd2, "S": S2, "g": g2}, x)[name]
                m = dd["q2"] == q
                preds[r, m] = np.interp(dd["x"][m], x, obs)
        s = preds.std(axis=0)
        zr = (preds - dd["t_true"][None]) / np.maximum(s[None], 1e-30)
        zm = (preds.mean(axis=0) - dd["t_true"]) / np.maximum(s, 1e-30)
        data[name] = {
            "single_replica_1sigma": coverage(zr),
            "single_replica_90": coverage(zr, 1.6449),
            "single_replica_rms_z": float(np.sqrt(np.mean(zr ** 2))),
            "ensemble_mean_1sigma": coverage(zm),
        }

    out = dict(n_rep=int(R), q0_pdf=q0, data_points=data)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"== 校准检查（{R} replicas）==")
    print("Q0 PDF 单 replica：1σ=%.1f%%  90%%=%.1f%%  rms z=%.2f" % (
        100 * q0["single_replica"]["coverage_1sigma"],
        100 * q0["single_replica"]["coverage_90"],
        q0["single_replica"]["rms_z"]))
    print("  逐味 1σ：" + "  ".join(
        f"{k}={100*v:.0f}%" for k, v in
        q0["single_replica"]["per_flavor_1sigma"].items()))
    print("Q0 PDF ensemble 均值：1σ=%.1f%%（保守，均值有 R 倍平均）" % (
        100 * q0["ensemble_mean"]["coverage_1sigma"]))
    print("数据点单 replica：")
    for n, v in data.items():
        print(f"  {n:7s} 1σ={100*v['single_replica_1sigma']:.1f}%  "
              f"90%={100*v['single_replica_90']:.1f}%  "
              f"rms z={v['single_replica_rms_z']:.2f}  "
              f"（均值 1σ={100*v['ensemble_mean_1sigma']:.1f}%）")
    print(f"已保存：{args.out}")


if __name__ == "__main__":
    main()
