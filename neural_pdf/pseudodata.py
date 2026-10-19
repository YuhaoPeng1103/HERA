# -*- coding: utf-8 -*-
"""
DIS 伪数据生成（闭环测试）
==========================
用真值 PDF + LO DGLAP 生成质子结构函数 F2(x,Q^2) 的"伪实验数据"，
在每个 (x,Q^2) 点加入相对误差的高斯噪声，保存（x, Q2, F2, err）。
"""
import numpy as np

from . import dglap, truth


def build(n_x=400, seed=42):
    """构建网格、矩阵、真值演化与伪数据。"""
    x, w, wx, edges = dglap.make_grid(n=n_x)
    Mqq, Mqg, Mgq, Mgg = dglap.build_matrices(x, w, edges)
    pdf0 = truth.truth_pdfs(x, wx)
    return x, w, wx, (Mqq, Mqg, Mgq, Mgg), pdf0


def make_data(x, w, mats, pdf0, q2_list=(2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 120.0),
              rel_err=0.02, seed=2026):
    """在 (x, Q2) 网格上生成 F2 伪数据（运动学截断：x<0.7）。"""
    Mqq, Mqg, Mgq, Mgg = mats
    evolved = dglap.evolve(pdf0, x, w, Mqq, Mqg, Mgq, Mgg, q2_list)

    rng = np.random.default_rng(seed)
    x_bins = np.array([
        0.0016, 0.0025, 0.004, 0.0063, 0.01, 0.016, 0.025, 0.04,
        0.063, 0.1, 0.16, 0.25, 0.4, 0.65,
    ])

    rows = []
    for q2 in q2_list:
        pdf = evolved[q2]
        # 在数据点 x 上插值 F2
        f2_grid = dglap.f2_proton(pdf, x)
        for xd in x_bins:
            if xd > 0.7:
                continue
            f2_true = float(np.interp(xd, x, f2_grid))
            err = rel_err * f2_true
            f2_obs = f2_true + rng.normal(0.0, err)
            rows.append((xd, q2, f2_obs, err))

    data = np.array(rows, dtype=[("x", "f8"), ("q2", "f8"), ("f2", "f8"), ("err", "f8")])
    return data, evolved


if __name__ == "__main__":
    x, w, wx, mats, pdf0 = build()
    Mqq, Mqg, Mgq, Mgg = mats
    dglap.check_moments(x, w, wx, Mqq, Mqg, Mgq, Mgg)
    truth.check_sum_rules(pdf0, x, wx)
    data, evolved = make_data(x, w, mats, pdf0)
    print(f"伪数据点数：{len(data)}")
    print(data[:5])
    # 动量守恒随演化检验
    for q2 in sorted(evolved):
        p = evolved[q2]
        mom = np.sum(wx * x * (p["Vu"] + p["Vd"] + 4 * p["S"] + p["g"]))
        print(f"  Q^2={q2:6.1f}  动量和={mom:.6f}")
