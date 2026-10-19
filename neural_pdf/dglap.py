# -*- coding: utf-8 -*-
"""
LO DGLAP 演化器（x 空间，n_f=2 玩具模型，仅含 u/d 味与对称海）
================================================================

味基矢：
    V_u = u - ubar   （u 价夸克，非单态）
    V_d = d - dbar   （d 价夸克，非单态）
    S   = ubar = dbar（每味反夸克，对称海；s 味暂关闭）
    g   胶子
    单态组合 Sigma = u + ubar + d + dbar = V_u + V_d + 4 S

演化方程（t = ln Q^2，a(t) = alpha_s/(2pi)）：
    dV_u/dt = a (Pqq (x) V_u)
    dV_d/dt = a (Pqq (x) V_d)
    dSigma/dt = a (Pqq (x) Sigma + 2 nf Pqg (x) g)
    dg/dt     = a (Pgq (x) Sigma + Pgg (x) g)

分裂函数（LO）：
    Pqq(z) = CF [ (1+z^2)/(1-z)_+ + 3/2 delta(1-z) ]
    Pqg(z) = TR [ z^2 + (1-z)^2 ]
    Pgq(z) = CF [ 1 + (1-z)^2 ] / z
    Pgg(z) = 2 CA [ z/(1-z) + (1-z)/z + z(1-z) ] + beta0/2 delta(1-z)
    beta0  = 11/3 CA - 4/3 TR nf

所有卷积矩阵在对数 x 网格上预计算（梯形法），plus 处方用解析式处理。
"""
import numpy as np

CF = 4.0 / 3.0
CA = 3.0
TR = 1.0 / 2.0
NF = 2  # 活动味数（u, d）
BETA0 = 11.0 / 3.0 * CA - 4.0 / 3.0 * TR * NF  # = 29/3

Q02 = 1.0       # 初始标度 GeV^2
ALPHAS_Q0 = 0.5  # LO 玩具设定


def make_grid(n=300, xmin=1e-4, xmax=1.0 - 1e-6, p_map=2.0):
    """自洽的单元-中点网格：n 个格元、n 个中点节点。
    边在映射后的对数空间均匀分布（u=u_max(1-r)^p，在 x≈1 处加密）；
    节点 x_j = 格元几何中点。
    返回 x（节点）、w（格元对数权重 Δt，即 dy/y）、
    wx（格元线性权重 Δx，即 dx）、edges（格元边界）。
    """
    umax = -np.log(xmin)
    r = np.linspace(0.0, 1.0, n + 1)
    u = umax * (1.0 - r) ** p_map
    u = np.clip(u, -np.log(xmax), None)
    te = -u
    te[0] = np.log(xmin)
    te[-1] = np.log(xmax)
    edges = np.exp(te)
    t = 0.5 * (te[:-1] + te[1:])
    x = np.exp(t)
    w = te[1:] - te[:-1]          # 每格元 Δt
    wx = edges[1:] - edges[:-1]   # 每格元 Δx
    return x, w, wx, edges


def _kernels(z):
    """不含 delta 的正则分裂函数（plus 项单独处理）。"""
    safe = (z < 1.0)
    p_qq_reg = np.where(safe, (1.0 + z ** 2) / (1.0 - z), 0.0)
    p_qg = TR * (z ** 2 + (1.0 - z) ** 2)
    p_gq = CF * (1.0 + (1.0 - z) ** 2) / z
    p_gg_reg = np.where(safe,
                        2.0 * CA * (z / (1.0 - z) + (1.0 - z) / z + z * (1.0 - z)),
                        0.0)
    return p_qq_reg, p_qg, p_gq, p_gg_reg


def build_matrices(x, w, edges):
    """构造 4 个卷积矩阵 M，使 (P⊗f)_i = Σ_j M_ij f_j。

    自洽单元-中点离散：f 存于格元中点 x_j；
    - 完整格元 j>i：y=x_j，权重 Δt_j=w_j，无插值（z=x_i/x_j<1）；
    - 首格元 j=i：y∈[x_i,edge_{i+1}] 为半格，取半格中点 y_hi，
      权重 w_i/2，f 线性插值到 0.75 权重的 f_{i+1}。
    极点 1/(1-z) 用对角扣除 + 解析常数实现 plus 处方。
    """
    n = len(x)
    t = np.log(x)
    te = np.log(edges)
    Mqq = np.zeros((n, n))
    Mqg = np.zeros((n, n))
    Mgq = np.zeros((n, n))
    Mgg = np.zeros((n, n))

    def add(i, j, wj, z, theta):
        """向矩阵加入一个求积点（theta 为 f_{j+1} 的插值权重）。"""
        gqq = (1.0 + z ** 2) / (1.0 - z)
        cgg = (1.0 / (1.0 - z) - 1.0
               + (1.0 - z) / z + z * (1.0 - z))
        vqg = TR * (z ** 2 + (1.0 - z) ** 2)
        vgq = CF * (1.0 + (1.0 - z) ** 2) / z
        Mqq[i, j] += wj * CF * gqq * (1 - theta)
        Mqq[i, i] -= wj * CF * 2.0 * z / (1.0 - z)
        Mgg[i, j] += wj * 2.0 * CA * cgg * (1 - theta)
        Mgg[i, i] -= wj * 2.0 * CA * z / (1.0 - z)
        Mqg[i, j] += wj * vqg * (1 - theta)
        Mgq[i, j] += wj * vgq * (1 - theta)
        if theta > 0:
            Mqq[i, j + 1] += wj * CF * gqq * theta
            Mgg[i, j + 1] += wj * 2.0 * CA * cgg * theta
            Mqg[i, j + 1] += wj * vqg * theta
            Mgq[i, j + 1] += wj * vgq * theta

    for i in range(n):
        xi = x[i]
        # 首格元 j=i 的上半格
        if i < n - 1:
            y_hi = np.exp(0.5 * (t[i] + te[i + 1]))
            add(i, i, 0.5 * w[i], xi / y_hi, 0.75)
        # 完整格元 j>i
        for j in range(i + 1, n):
            z = xi / x[j]
            add(i, j, w[j], z, 0.0)
        Mqq[i, i] += CF * (1.5 + 2.0 * np.log(1.0 - xi))
        Mgg[i, i] += 2.0 * CA * np.log(1.0 - xi) + 0.5 * BETA0

    return Mqq, Mqg, Mgq, Mgg


def check_moments(x, w, wx, Mqq, Mqg, Mgq, Mgg, verbose=True):
    """矩检验：价夸克数守恒用 n=0 矩；动量守恒用 n=1 矩。
    K_n(P) = [Σ_i wx_i x_i^n (P⊗1)_i] / [Σ_j w_j x_j^{n+1}]。"""
    ones = np.ones_like(x)
    def Kn(M, n):
        return np.sum(wx * x ** n * (M @ ones)) / np.sum(w * x ** (n + 1))
    K0 = lambda M: Kn(M, 0)
    K1 = lambda M: Kn(M, 1)

    k0_qq = K0(Mqq)
    k0_qg = K0(Mqg)
    k1_qq, k1_qg, k1_gq, k1_gg = K1(Mqq), K1(Mqg), K1(Mgq), K1(Mgg)
    # n=0：价夸克数守恒；K0(Pqg)=2/3 TR = 1/3
    # n=1：动量守恒 K1(Pqq)+K1(Pgq)=0，K1(Pgg)+2nf K1(Pqg)=0
    checks = {
        "K0(Pqq)=0 (价夸克数守恒)": (k0_qq, 0.0),
        "K0(Pqg)=2/3 TR=1/3": (k0_qg, 2.0 / 3.0 * TR),
        "K1(Pqq)=-4/3 CF=-16/9": (k1_qq, -4.0 / 3.0 * CF),
        "K1(Pgq)=+4/3 CF=+16/9": (k1_gq, 4.0 / 3.0 * CF),
        "K1(Pqg)=TR/3=1/6": (k1_qg, TR / 3.0),
        "K1(Pgg)+2nf K1(Pqg)=0": (k1_gg + 2 * NF * k1_qg, 0.0),
    }
    if verbose:
        print("== DGLAP 卷积矩阵矩检验 ==")
        for name, (got, ref) in checks.items():
            print(f"  {name:32s} 数值={got:+.5f}  理论={ref:+.5f}  偏差={abs(got-ref):.2e}")
    return checks


def alphas_of_t(t):
    """LO 跑动耦合：alpha_s(t) = 2pi / (beta0 (t - t_Lambda))。"""
    t0 = np.log(Q02)
    t_lam = t0 - 2.0 * np.pi / (BETA0 * ALPHAS_Q0)
    return 2.0 * np.pi / (BETA0 * (t - t_lam))


def t_lambda():
    """t_Lambda = t0 - 2pi/(beta0 alpha_s(t0))。"""
    t0 = np.log(Q02)
    return t0 - 2.0 * np.pi / (BETA0 * ALPHAS_Q0)


def eta_q2(q2):
    """等效演化时间 η = ∫ alpha_s/(2pi) dlnQ² = (1/beta0) ln((t-tLam)/(t0-tLam))。"""
    t0 = np.log(Q02)
    tlam = t_lambda()
    return (1.0 / BETA0) * np.log((np.log(q2) - tlam) / (t0 - tlam))


def build_K(Mqq, Mqg, Mgq, Mgg):
    """4n×4n 线性演化矩阵 K（状态序 Vu,Vd,Σ,g；Σ=Vu+Vd+4S）。"""
    n = Mqq.shape[0]
    K = np.zeros((4 * n, 4 * n))
    for k in range(3):
        K[k * n:(k + 1) * n, k * n:(k + 1) * n] = Mqq
    K[2 * n:3 * n, 3 * n:4 * n] = 2.0 * NF * Mqg
    K[3 * n:4 * n, 2 * n:3 * n] = Mgq
    K[3 * n:4 * n, 3 * n:4 * n] = Mgg
    return K


def evolve(pdfs_q0, x, w, Mqq, Mqg, Mgq, Mgg, q2_grid):
    """
    从 Q0^2 演化到 q2_grid 中各标度（解析矩阵指数，无条件稳定）。
    pdfs_q0: dict(Vu, Vd, S, g)，均为长度 len(x) 的数组。
    返回 {q2: dict(Vu,Vd,S,g)}。
    """
    from scipy.linalg import expm
    n = len(x)
    Vu = pdfs_q0["Vu"].copy()
    Vd = pdfs_q0["Vd"].copy()
    S = pdfs_q0["S"].copy()
    g = pdfs_q0["g"].copy()
    Sigma = Vu + Vd + 4.0 * S
    state0 = np.concatenate([Vu, Vd, Sigma, g])
    K = build_K(Mqq, Mqg, Mgq, Mgg)

    out = {}
    for q2 in sorted(set(float(q) for q in q2_grid)):
        U = expm(eta_q2(q2) * K)
        state = U @ state0
        Vu2, Vd2, Sigma2, g2 = [state[k * n:(k + 1) * n] for k in range(4)]
        S2 = (Sigma2 - Vu2 - Vd2) / 4.0
        out[q2] = {"Vu": Vu2, "Vd": Vd2, "S": S2, "g": g2}
    return out


def f2_proton(pdfs, x_eval=None):
    """F2(x,Q2) = x[4/9 (Vu+2S) + 1/9 (Vd+2S)]，可在指定 x 点取值。"""
    f2 = x_eval * (4.0 / 9.0 * (pdfs["Vu"] + 2 * pdfs["S"])
                   + 1.0 / 9.0 * (pdfs["Vd"] + 2 * pdfs["S"]))
    return f2


def interp_pdfs(pdfs, x, xnew):
    return {k: np.interp(xnew, x, v) for k, v in pdfs.items()}
