# -*- coding: utf-8 -*-
"""
"真值"PDF 参数化（闭环测试用）
================================
在 Q0^2 = 1 GeV^2 给出一套满足求和规则的玩具真值：
    Vu(x) = A_u x^{a_u} (1-x)^{b_u}
    Vd(x) = A_d x^{a_d} (1-x)^{b_d}
    S(x)  = A_s x^{a_s} (1-x)^{b_s}
    g(x)  = A_g x^{a_g} (1-x)^{b_g}

归一化：
    ∫ Vu dx = 2, ∫ Vd dx = 1                      （价夸克数）
    ∫ x (Vu + Vd + 4S + g) dx = 1                 （动量求和）
"""
import numpy as np
from math import gamma

def B(a, b):
    """Beta 函数（避免 scipy 依赖）。"""
    return gamma(a) * gamma(b) / gamma(a + b)

# 形状参数（低标度下小 x 胶子/海夸克近似 x^-1 奇异性，
# 经 LO DGLAP 演化后 F2 在数据点上落在物理量级，已通过参数扫描校准）
SHAPE = {
    "Vu": dict(a=0.7, b=3.2),
    "Vd": dict(a=0.5, b=4.5),
    "S":  dict(a=-1.4, b=8.0),
    "g":  dict(a=-1.2, b=4.0),
}
SEA_MOMENTUM_FRACTION = 0.015  # 4S 携带的动量份额（Q0 处玩具设定）


def _shape(x, a, b):
    return np.power(x, a) * np.power(1.0 - x, b)


def truth_pdfs(x, wx):
    """返回归一化后的真值 PDF 字典（Vu, Vd, S, g）。wx 为线性权重 dx。"""
    raw = {k: _shape(x, p["a"], p["b"]) for k, p in SHAPE.items()}

    def integ(f):
        return float(np.sum(wx * f))

    def xinteg(f):
        return float(np.sum(wx * x * f))

    Au = 2.0 / B(SHAPE["Vu"]["a"] + 1, SHAPE["Vu"]["b"] + 1)
    Ad = 1.0 / B(SHAPE["Vd"]["a"] + 1, SHAPE["Vd"]["b"] + 1)
    Vu = Au * raw["Vu"]
    Vd = Ad * raw["Vd"]

    # 海：给定动量份额反推归一化
    mom_4s_target = SEA_MOMENTUM_FRACTION
    mom_raw_4s = xinteg(4.0 * raw["S"])
    As = mom_4s_target / mom_raw_4s
    S = As * raw["S"]

    # 胶子：动量求和补余
    mom_g = 1.0 - xinteg(Vu) - xinteg(Vd) - xinteg(4.0 * S)
    Ag = mom_g / xinteg(raw["g"])
    g = Ag * raw["g"]

    return {"Vu": Vu, "Vd": Vd, "S": S, "g": g}


def check_sum_rules(pdfs, x, wx, verbose=True):
    """求和规则诊断（线性权重 dx）。"""
    n_u = np.sum(wx * pdfs["Vu"])
    n_d = np.sum(wx * pdfs["Vd"])
    mom = np.sum(wx * x * (pdfs["Vu"] + pdfs["Vd"] + 4 * pdfs["S"] + pdfs["g"]))
    if verbose:
        print("== 求和规则 ==")
        print(f"  ∫Vu dx = {n_u:.6f}  (目标 2)")
        print(f"  ∫Vd dx = {n_d:.6f}  (目标 1)")
        print(f"  动量和  = {mom:.6f}  (目标 1)")
    return dict(n_u=n_u, n_d=n_d, mom=mom)
