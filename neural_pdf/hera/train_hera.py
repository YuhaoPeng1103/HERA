# -*- coding: utf-8 -*-
"""
HERA 真实数据上的神经 PDF 拟合（A 论文的真实数据演示）
=====================================================
* 数据：HERA combined NC e+p reduced cross sections, sqrt(s)=318 GeV,
  Q2 <= q2max, x >= 1e-4；完整协方差（stat+uncor+8 个关联源）。
* 理论：LO、n_f=2、sigma_r,NC ≈ F2（忽略 FL 与 gamma-Z 干涉）。
* 拟合：chi2 = r^T C^-1 r；sum rules + smoothness 软约束；
  replicas 从 N(y, C) 抽样。
* 明确局限：n_f=2、无 s/c、无 NLO、低 Q2 近似；这是方法演示，
  不是竞争性的 PDF 提取。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from .. import dglap
from .. import model as nmod
from .. import pseudodata as pdata
from . import dglap_nf3


def build_fit_inputs(npz_path, q2max, xmin):
    d = np.load(npz_path)
    q2, x, y, C = d["q2"], d["x"], d["y"], d["C"]
    m = (q2 <= q2max) & (x >= xmin)
    q2, x, y, C = q2[m], x[m], y[m], C[np.ix_(m, m)]
    q2_keys = sorted(set(float(q) for q in q2))
    q2i = np.array([q2_keys.index(float(q)) for q in q2])
    return dict(q2=q2, x=x, y=y, C=C, q2_keys=q2_keys, q2i=q2i)


def prediction(f0, x_t, U_t, q2_keys, q2i, xd):
    Vu, Vd, S, g = f0
    state0 = torch.stack([Vu, Vd, Vu + Vd + 6.0 * S, g], dim=0).reshape(-1)
    n = x_t.shape[0]
    pred = torch.zeros_like(xd)
    for qi, q in enumerate(q2_keys):
        mask = q2i == qi
        if not mask.any():
            continue
        st = (U_t[q] @ state0).reshape(4, n)
        Vu2, Vd2, Sig2, g2 = st
        S2 = (Sig2 - Vu2 - Vd2) / 6.0
        f2 = x_t * (4.0 / 9.0 * Vu2 + 1.0 / 9.0 * Vd2 + 4.0 / 3.0 * S2)
        pred[mask] = nmod.interp1d(x_t, f2, xd[mask])
    return pred


def fit_replica(x_t, wx_t, U_t, data, Linv_t, seed, epochs=1500, lr=0.02,
                w_sum=100.0, w_smooth=1e-3, w_exp=0.1, lbfgs_iters=400,
                width=24, verbose=False):
    torch.manual_seed(seed)
    net = nmod.NeuralPDF(width=width).double()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    xd, q2i, y, q2_keys = data["x"], data["q2i"], data["y"], data["q2_keys"]
    dl = torch.log(x_t)[1:] - torch.log(x_t)[:-1]
    n = len(y)

    def objective():
        h, f0 = net.shape_and_pdf(x_t)
        Vu, Vd, S, g = f0
        pred = prediction(f0, x_t, U_t, q2_keys, q2i, xd)
        r = y - pred
        chi2 = torch.sum((Linv_t @ r) ** 2) / n
        num_u = torch.sum(wx_t * Vu)
        num_d = torch.sum(wx_t * Vd)
        mom = torch.sum(wx_t * x_t * (Vu + Vd + 6.0 * S + g))
        loss_sum = (num_u - 2.0) ** 2 + (num_d - 1.0) ** 2 + (mom - 1.0) ** 2
        smooth = 0.0
        for k in range(4):
            hk = h[:, k]
            d2 = (hk[2:] - 2.0 * hk[1:-1] + hk[:-2]) / dl[1:] ** 2
            smooth = smooth + torch.mean((dl[:-1] * d2) ** 2)
        loss = chi2 + w_sum * loss_sum + w_smooth * smooth \
            + w_exp * torch.sum(net.exp_delta ** 2)
        return loss, chi2

    for ep in range(epochs):
        opt.zero_grad()
        loss, chi2 = objective()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and ep % 300 == 0:
            print(f"  ep {ep:5d} chi2/N={chi2.item():.3f}")
    if lbfgs_iters > 0:
        lb = torch.optim.LBFGS(net.parameters(), lr=0.5, max_iter=lbfgs_iters,
                               line_search_fn="strong_wolfe", history_size=50,
                               tolerance_grad=1e-9, tolerance_change=1e-11)

        def closure():
            lb.zero_grad()
            loss, _ = objective()
            loss.backward()
            return loss
        lb.step(closure)
        with torch.no_grad():
            _, chi2 = objective()
    return net, chi2.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(
        os.getcwd(), "outputs", "hera_selected.npz"))
    ap.add_argument("--q2max", type=float, default=10.0)
    ap.add_argument("--xmin", type=float, default=1e-4)
    ap.add_argument("--n-rep", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lbfgs", type=int, default=400)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--wsum", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=os.path.join(os.getcwd(), "outputs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(args.seed)

    x, w, wx, mats, _ = pdata.build()
    data = build_fit_inputs(args.npz, args.q2max, args.xmin)
    q2_keys = data["q2_keys"]
    U = dglap_nf3.evolution_operators(x, w, mats, q2_keys)
    U_t = {float(q): torch.tensor(Uq, dtype=torch.float64) for q, Uq in U.items()}
    x_t = torch.tensor(x, dtype=torch.float64)
    wx_t = torch.tensor(wx, dtype=torch.float64)
    xd = torch.tensor(data["x"], dtype=torch.float64)
    y = torch.tensor(data["y"], dtype=torch.float64)
    q2i = torch.tensor(data["q2i"], dtype=torch.long)
    C = data["C"]
    jitter = 1e-12 * np.trace(C) / len(C)
    L = np.linalg.cholesky(C + jitter * np.eye(len(C)))
    Linv = np.linalg.inv(L)
    Linv_t = torch.tensor(Linv, dtype=torch.float64)
    fit_data = dict(x=xd, q2i=q2i, y=y, q2_keys=q2_keys)
    n = len(y)
    print(f"HERA 拟合：{n} 点，{len(q2_keys)} 个 Q2 值，"
          f"Q2≤{args.q2max}，x≥{args.xmin}")

    f0_list, chi2_list = [], []
    t0 = time.time()
    for r in range(args.n_rep):
        # 从 N(y, C) 抽样：y_rep = y + L z
        z = torch.tensor(rng.normal(size=n), dtype=torch.float64)
        y_rep = y + torch.tensor(L, dtype=torch.float64) @ z
        fd = dict(fit_data); fd["y"] = y_rep
        net, chi2 = fit_replica(x_t, wx_t, U_t, fd, Linv_t,
                                seed=args.seed + 1000 + r,
                                epochs=args.epochs, lbfgs_iters=args.lbfgs,
                                width=args.width, w_sum=args.wsum)
        with torch.no_grad():
            _, f0 = net.shape_and_pdf(x_t)
        f0_list.append(f0.numpy())
        chi2_list.append(chi2)
        print(f"replica {r+1}/{args.n_rep}  chi2/N(noise)={chi2:.3f}  "
              f"累计 {time.time()-t0:.0f}s")

    f0s = np.stack(f0_list)
    f0_mean, f0_std = f0s.mean(0), f0s.std(0)
    # 集合平均 vs 中心数据的 chi2（用未加噪的 y）
    with torch.no_grad():
        net_f0 = torch.tensor(f0_mean)
        pred = prediction(net_f0, x_t, U_t, q2_keys, q2i, xd).numpy()
    r0 = data["y"] - pred
    chi2_central = float(r0 @ np.linalg.solve(C, r0) / len(r0))
    pulls = Linv @ r0
    out = dict(n_rep=args.n_rep, n_points=int(n), q2max=args.q2max,
               xmin=args.xmin,
               chi2_noise_mean=float(np.mean(chi2_list)),
               chi2_noise_std=float(np.std(chi2_list)),
               chi2_central=chi2_central,
               pull_mean=float(np.mean(pulls)),
               pull_std=float(np.std(pulls)),
               sumrules=dict(
                   num_u=float(np.sum(wx * f0_mean[0])),
                   num_d=float(np.sum(wx * f0_mean[1])),
                   mom=float(np.sum(wx * x * f0_mean.sum(0)))))
    np.savez(os.path.join(args.out, "hera_fit.npz"),
             x=x, wx=wx, f0_mean=f0_mean, f0_std=f0_std,
             f0_replicas=f0s, data_q2=data["q2"], data_x=data["x"],
             data_y=data["y"], data_pred=pred, pulls=pulls)
    with open(os.path.join(args.out, "hera_fit.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n== HERA 拟合结果 ==")
    print(f"chi2/N (ensemble vs central) = {chi2_central:.3f}")
    print(f"chi2/N (replicas vs own noisy data) = "
          f"{np.mean(chi2_list):.3f} ± {np.std(chi2_list):.3f}")
    print(f"pull 均值/标准差 = {np.mean(pulls):+.3f} / {np.std(pulls):.3f}")
    print(f"求和规则：∫Vu={out['sumrules']['num_u']:.4f}  "
          f"∫Vd={out['sumrules']['num_d']:.4f}  "
          f"动量={out['sumrules']['mom']:.4f}")
    print(f"结果保存至 {args.out}/hera_fit.npz, hera_fit.json")


if __name__ == "__main__":
    main()
