# -*- coding: utf-8 -*-
"""
M0.5 多观测量伪数据与关联系统误差
==================================
在 M0（仅 F2^p、仅非关联误差）基础上的升级：

观测量
------
* F2^p（光子，质子）
* F2^d（光子，氘核；同位旋标量平均，忽略核修正）
* xF3^{nu p}（中微子 CC）  = 2 x V_d
* xF3^{nubar p}（反中微子 CC）= 2 x V_u

在 n_f=2、海夸克对称（ubar = dbar = S，无 s/c）假设下，中微子
结构函数只依赖价夸克，因此直接提供 V_u、V_d 的味分离约束；
F2^d = (5/18) x (Vu+Vd+4S) 提供总夸克单态组合的独立约束。

关联系统误差
------------
每个数据集有 K=3 个 nuisance 参数 lambda_k，对应 ln x 上的
Legendre 基 b_k(x) in {1, z, (3z^2-1)/2}：
    y_i = t_i (1 + sum_k lambda_k b_{k,i}) + eps_i,
    lambda_k ~ N(0, sigma_k^2),  eps_i ~ N(0, sigma_i^2).
拟合时对 lambda 做解析 profile（KKT 线性方程组），得到
chi2_prof = r^T W r - v^T (M^-1 v)，
其中 M = A^T W A + Lambda, v = A^T W r, A_ik = t_i b_{k,i}。
该式对理论值可微，可直接反传到神经 PDF。
"""
import numpy as np

from . import dglap

# ---------------------------------------------------------------------------
# 运动学网格
# ---------------------------------------------------------------------------
Q2_GRID = np.array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 120.0])
Q2_NEUTRINO = np.array([2.0, 4.0, 8.0, 16.0, 32.0])
X_BINS = np.array([
    0.0016, 0.0025, 0.004, 0.0063, 0.01, 0.016, 0.025, 0.04,
    0.063, 0.1, 0.16, 0.25, 0.4, 0.65,
])

# 每个数据集：(Q2 列表, 非关联相对误差, 三个关联系统误差先验宽度)
SPECS = {
    "F2p":    dict(q2=Q2_GRID,     rel=0.020, syst=(0.015, 0.010, 0.008)),
    "F2d":    dict(q2=Q2_GRID,     rel=0.030, syst=(0.020, 0.012, 0.010)),
    "xF3vp":  dict(q2=Q2_NEUTRINO, rel=0.050, syst=(0.030, 0.020, 0.015)),
    "xF3vbp": dict(q2=Q2_NEUTRINO, rel=0.050, syst=(0.030, 0.020, 0.015)),
}


# ---------------------------------------------------------------------------
# 观测量（numpy 版，用于生成数据）
# ---------------------------------------------------------------------------
def observables_np(pdfs, x):
    """pdfs: dict(Vu,Vd,S,g)，均为 len(x) 数组。返回各观测量的网格值。"""
    Vu, Vd, S = pdfs["Vu"], pdfs["Vd"], pdfs["S"]
    f2p = x * (4.0 / 9.0 * (Vu + 2.0 * S) + 1.0 / 9.0 * (Vd + 2.0 * S))
    f2n = x * (4.0 / 9.0 * (Vd + 2.0 * S) + 1.0 / 9.0 * (Vu + 2.0 * S))
    return {
        "F2p": f2p,
        "F2d": 0.5 * (f2p + f2n),
        # 中微子 CC：n_f=2、对称海下 F3 只含价夸克
        "xF3vp": 2.0 * x * Vd,
        "xF3vbp": 2.0 * x * Vu,
    }


def syst_basis(xd):
    """ln x 上的 Legendre 基 b_k(x)：常数、线性、二次。"""
    z = 2.0 * (np.log(xd) - np.log(X_BINS[0])) / (
        np.log(X_BINS[-1]) - np.log(X_BINS[0])) - 1.0
    return np.stack([np.ones_like(z), z, 0.5 * (3.0 * z ** 2 - 1.0)], axis=1)


# ---------------------------------------------------------------------------
# 数据集构建
# ---------------------------------------------------------------------------
def build_datasets(x, w, mats, pdf0, seed=2026):
    """由真值 PDF 生成四个数据集的中心值（lambda=0 + 一次统计涨落）。

    返回 datasets[name] = dict(x, q2, t_true, y, sigma, B, A, sigma_beta)。
    A = t_true[:,None] * B 是系统误差对数据的线性响应矩阵。
    """
    evolved = dglap.evolve(pdf0, x, w, *mats, Q2_GRID)
    rng = np.random.default_rng(seed)
    datasets = {}
    for name, spec in SPECS.items():
        xs, q2s, tvals = [], [], []
        for q2 in spec["q2"]:
            obs_grid = observables_np(evolved[q2], x)[name]
            for xd in X_BINS:
                xs.append(xd)
                q2s.append(q2)
                tvals.append(float(np.interp(xd, x, obs_grid)))
        xd = np.asarray(xs)
        q2d = np.asarray(q2s)
        t_true = np.asarray(tvals)
        sigma = spec["rel"] * np.abs(t_true)
        B = syst_basis(xd)
        A = t_true[:, None] * B
        # 中心数据 = 真值 + 一次系统偏移 lambda0 + 一次统计涨落。
        # lambda0 必须抽样，否则中心数据异常贴近真值，覆盖率会系统性高估。
        lam0 = rng.normal(0.0, np.asarray(spec["syst"], dtype=float))
        y = t_true + A @ lam0 + rng.normal(0.0, sigma)
        datasets[name] = dict(
            x=xd, q2=q2d, t_true=t_true, y=y, sigma=sigma,
            B=B, A=A, sigma_beta=np.asarray(spec["syst"], dtype=float),
            lam0=lam0,
        )
    return datasets, evolved


def draw_replica(d, rng):
    """NNPDF 风格 replica：在**中心数据**上独立重抽统计与关联系统误差。

    注意必须以中心数据 d["y"] 为基准（而不是无噪声真值），否则集合均值
    会退化为真值，覆盖率检验将系统性高估（100%）。
    """
    eps = rng.normal(0.0, d["sigma"])
    lam = rng.normal(0.0, d["sigma_beta"])
    return d["y"] + eps + d["A"] @ lam


def dataset_summary(datasets):
    return [(name, len(d["y"]), d["sigma_beta"].copy())
            for name, d in datasets.items()]
