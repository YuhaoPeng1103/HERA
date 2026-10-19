# -*- coding: utf-8 -*-
"""M0.5 数值自检：观测量一致性、profile chi2、replica 统计。"""
import os
import sys

import numpy as np
import torch

from . import dglap
from . import multidata as mdata
from . import pseudodata as pdata
from . import train_m05 as T


def main():
    torch.set_default_dtype(torch.float64)
    x, w, wx, mats, pdf0 = pdata.build()
    datasets, evolved = mdata.build_datasets(x, w, mats, pdf0)

    # 1) numpy 与 torch 观测量逐点一致
    x_t = torch.tensor(x, dtype=torch.float64)
    for q2 in (2.0, 16.0, 120.0):
        np_obs = mdata.observables_np(evolved[q2], x)
        torch_obs = T.obs_torch(
            {k: torch.tensor(evolved[q2][k]) for k in ("Vu", "Vd", "S", "g")},
            x_t)
        for name in np_obs:
            diff = np.max(np.abs(np_obs[name] - torch_obs[name].numpy()))
            assert diff < 1e-10, (q2, name, diff)
    print("1) numpy/torch 观测量一致：PASS")

    # 2) 无噪声理论数据代入 profile chi2 应为 0
    for name, d in datasets.items():
        pred = torch.tensor(d["t_true"])
        B = torch.tensor(d["B"])
        c2, lam = T.profile_chi2(torch.zeros_like(pred), pred, B,
                                 torch.tensor(d["sigma"]),
                                 torch.tensor(d["sigma_beta"]))
        assert abs(float(c2)) < 1e-8 and np.max(np.abs(lam.numpy())) < 1e-10
    print("2) 理论数据 profile chi2 = 0：PASS")

    # 3) replica 统计：均值→真值，标准差→ sqrt(sigma^2 + syst^2)
    rng = np.random.default_rng(123)
    name = "F2p"
    d = datasets[name]
    draws = np.stack([mdata.draw_replica(d, rng) for _ in range(4000)])
    mean_bias = np.mean((draws - d["y"]) / d["sigma"])
    syst_var = np.sum((d["A"] * d["sigma_beta"][None, :]) ** 2, axis=1)
    expected = np.sqrt(d["sigma"] ** 2 + syst_var)
    ratio = np.std(draws, axis=0) / expected
    assert abs(mean_bias) < 0.02 and np.max(np.abs(ratio - 1)) < 0.1
    print(f"3) replica 统计（以中心数据为基准）：归一化均值偏差 {mean_bias:+.2e}，"
          f"标准差比 {ratio.min():.3f}–{ratio.max():.3f}：PASS")

    # 4) profile chi2 的无偏性：MC 期望应约为 1
    def profiled(t, y, sig, B, sb):
        wgt = 1.0 / sig ** 2
        A = t[:, None] * B
        M = A.T @ (wgt[:, None] * A) + np.diag(1.0 / sb ** 2)
        r = y - t
        v = A.T @ (wgt * r)
        return np.sum(wgt * r ** 2) - v @ np.linalg.solve(M, v)

    d0 = datasets["F2p"]
    vals = []
    for _ in range(500):
        lam0 = rng.normal(0.0, d0["sigma_beta"])
        y = d0["t_true"] + d0["A"] @ lam0 + rng.normal(0.0, d0["sigma"])
        vals.append(profiled(d0["t_true"], y, d0["sigma"],
                             d0["B"], d0["sigma_beta"]) / len(y))
    mc_mean = float(np.mean(vals))
    assert abs(mc_mean - 1.0) < 0.03
    print(f"4) profiled chi2/N 的 MC 期望 = {mc_mean:.3f}（目标 1）：PASS")

    # 5) 数据集规模
    n = sum(len(v["y"]) for v in datasets.values())
    print(f"5) 数据点总数 = {n}（预期 336）：PASS" if n == 336 else f"5) 数据点 {n}")
    print("\n全部自检通过。")


if __name__ == "__main__":
    main()
