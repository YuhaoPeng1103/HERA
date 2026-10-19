# -*- coding: utf-8 -*-
"""
LO DGLAP，n_f=3（u,d,s 对称海），用于 HERA 真实数据演示
=======================================================
味基：Vu, Vd, S(=ubar=dbar=sbar), g
单态：Sigma = Vu + Vd + 6S
F2p = x[4/9 Vu + 1/9 Vd + 4/3 S]
beta0 = 11 - 2/3*3 = 9
Pgg 的 beta0 项相对 n_f=2 有 -1/3 的对角修正。
"""
import numpy as np
from scipy.linalg import expm

from .. import dglap

NF = 3
BETA0 = 11.0 - 2.0 / 3.0 * NF   # = 9
Q02 = dglap.Q02
ALPHAS_Q0 = dglap.ALPHAS_Q0


def eta_q2(q2):
    t0 = np.log(Q02)
    tlam = t0 - 2.0 * np.pi / (BETA0 * ALPHAS_Q0)
    return (1.0 / BETA0) * np.log((np.log(q2) - tlam) / (t0 - tlam))


def build_K(mats):
    Mqq, Mqg, Mgq, Mgg = mats
    n = Mqq.shape[0]
    # n_f=2 的 Mgg 用了 beta0=29/3；换成 9，对角修正 0.5*(9-29/3)=-1/3
    Mgg3 = Mgg - (1.0 / 3.0) * np.eye(n)
    K = np.zeros((4 * n, 4 * n))
    for k in range(3):
        K[k * n:(k + 1) * n, k * n:(k + 1) * n] = Mqq
    K[2 * n:3 * n, 3 * n:4 * n] = 2.0 * NF * Mqg
    K[3 * n:4 * n, 2 * n:3 * n] = Mgq
    K[3 * n:4 * n, 3 * n:4 * n] = Mgg3
    return K


def evolution_operators(x, w, mats, q2_list):
    K = build_K(mats)
    return {float(q): expm(eta_q2(q) * K) for q in sorted(set(q2_list))}


def f2_proton(pdfs, x):
    return x * (4.0 / 9.0 * pdfs["Vu"] + 1.0 / 9.0 * pdfs["Vd"]
                + 4.0 / 3.0 * pdfs["S"])


def state0(vu, vd, s, g):
    return np.concatenate([vu, vd, vu + vd + 6.0 * s, g])


def flavors_from_state(state, n):
    vu, vd, sigma, g = [state[k * n:(k + 1) * n] for k in range(4)]
    s = (sigma - vu - vd) / 6.0
    return dict(Vu=vu, Vd=vd, S=s, g=g)
