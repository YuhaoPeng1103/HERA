# -*- coding: utf-8 -*-
"""
神经 PDF 参数化（PyTorch）
==========================
- 输入 x，输出 Q0 标度下 4 个味分布 (Vu, Vd, S, g)；
- 软正定性：softplus 输出 × 固定端点包络（保证可积、支持 x→0/1 行为）；
- 求和规则（价夸克数、动量）通过 loss 软约束实现；
- LO DGLAP 演化是线性的，演化算符 U(Q^2) 预先在 numpy 侧算好，
  训练时只需 U·f_Q0，反传极快。
"""
import numpy as np
import torch
import torch.nn as nn

from . import dglap


# ---------------------------------------------------------------------------
# 演化算符（numpy 预计算）
# ---------------------------------------------------------------------------
def evolution_operators(x, w, mats, q2_list):
    """解析演化算符 U(Q2)=exp(η K)，与 dglap.evolve 完全一致（无条件稳定）。"""
    from scipy.linalg import expm
    Mqq, Mqg, Mgq, Mgg = mats
    K = dglap.build_K(Mqq, Mqg, Mgq, Mgg)
    return {float(q2): expm(dglap.eta_q2(q2) * K) for q2 in sorted(q2_list)}


# ---------------------------------------------------------------------------
# 神经网络 PDF
# ---------------------------------------------------------------------------
ENVELOPE_POW = {
    # 基础小 x 指数（再加一个可学习的 δ 指数）；大 x 指数取真值量级。
    "Vu": (0.5, 3.2),
    "Vd": (0.3, 4.5),
    "S":  (-1.2, 8.0),
    "g":  (-1.0, 4.0),
}


class NeuralPDF(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 4),
        )
        # 小幅可学习整体尺度（初值为 1 附近）
        self.scale = nn.Parameter(torch.zeros(4))
        # 可学习的小 x 幂律修正指数 δ（x^{a0+δ}）
        self.exp_delta = nn.Parameter(torch.zeros(4))

    def shape_and_pdf(self, x):
        """返回形状因子 h(N,4) 与 PDF(4,N)；光滑正则应作用于 h（已剔除固定包络）。"""
        feat = torch.stack([torch.log(x), torch.log1p(-x), x], dim=1)
        h = torch.nn.functional.softplus(self.net(feat))  # (N,4)
        h = h * torch.exp(self.scale).unsqueeze(0)
        out = []
        for k, key in enumerate(("Vu", "Vd", "S", "g")):
            a, b = ENVELOPE_POW[key]
            env = torch.pow(x, a + self.exp_delta[k]) * torch.pow(1.0 - x, b)
            out.append(h[:, k] * env)
        return h, torch.stack(out, dim=0)

    def forward(self, x):
        """x: 1-D tensor。返回 (Vu, Vd, S, g)，形状 (4, N)。"""
        return self.shape_and_pdf(x)[1]


def state0_to_flavors(state0):
    """(Vu,Vd,Sigma,g) -> (Vu,Vd,S,g)。"""
    Vu, Vd, Sigma, g = state0
    S = (Sigma - Vu - Vd) / 4.0
    return torch.stack([Vu, Vd, S, g], dim=0)


def predict_all(model, x_t, U_t):
    """返回各 Q2 下的味分布 dict: {q2: (4,N) tensor}。"""
    f0 = model(x_t)  # (4,N)
    Vu, Vd, S, g = [f0[k] for k in range(4)]
    state0 = torch.stack([Vu, Vd, Vu + Vd + 4.0 * S, g], dim=0).reshape(-1)
    n = x_t.shape[0]
    res = {}
    for q2, U in U_t.items():
        st = U @ state0
        st = st.reshape(4, n)
        res[q2] = state0_to_flavors(st)
    return res


def f2_from_pdfs(pdfs, x_t):
    """pdfs: (4,N) 顺序 Vu,Vd,S,g。返回 F2 网格（未乘 x? 这里乘 x）。"""
    Vu, Vd, S, g = [pdfs[k] for k in range(4)]
    return x_t * (4.0 / 9.0 * (Vu + 2.0 * S) + 1.0 / 9.0 * (Vd + 2.0 * S))


def interp1d(x_grid, y_grid, x_new):
    """torch 版线性插值（要求 x_grid 单调递增）。"""
    idx = torch.searchsorted(x_grid, x_new).clamp(1, x_grid.numel() - 1)
    x0 = x_grid[idx - 1]
    x1 = x_grid[idx]
    y0 = y_grid[idx - 1]
    y1 = y_grid[idx]
    t = (x_new - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)
