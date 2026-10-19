# -*- coding: utf-8 -*-
"""
M0.5 闭环训练：多观测量 + 关联系统误差 + 解析 profile chi2
=========================================================
在 M0 的神经 PDF + LO DGLAP + replicas 基础上：

* 四个数据集联合拟合：F2^p, F2^d, xF3^{nu p}, xF3^{nubar p}；
* 每个数据集带 3 个 Legendre 形状的关联系统误差 nuisance，
  在每次前向中解析 profile（3x3 线性求解），chi2 对神经参数可微；
* 用 replicas 集合给出中心值与 68% 带，并按数据集报告
  chi2/N、pulls、覆盖率、求和规则，以及与 M0 的 PDF 不确定度对比。

纯 CPU 运行：演化算符 U(Q2)=exp(eta K) 预计算，训练时只做矩阵乘法。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from . import dglap
from . import model as nmod
from . import multidata as mdata
from . import pseudodata as pdata


# ---------------------------------------------------------------------------
# 张量化的观测量与 profile chi2
# ---------------------------------------------------------------------------
def evolved_flavors(net, x_t, U_t):
    """返回 Q0 形状因子 h、Q0 四味 f0，以及各 Q2 的四味分布。"""
    h, f0 = net.shape_and_pdf(x_t)
    Vu, Vd, S, g = f0
    state0 = torch.stack([Vu, Vd, Vu + Vd + 4.0 * S, g], dim=0).reshape(-1)
    n = x_t.shape[0]
    flavs = {}
    for q, U in U_t.items():
        st = (U @ state0).reshape(4, n)
        Vu2, Vd2, Sig2, g2 = st
        S2 = (Sig2 - Vu2 - Vd2) / 4.0
        flavs[q] = {"Vu": Vu2, "Vd": Vd2, "S": S2, "g": g2}
    return h, f0, flavs


def obs_torch(flav, x_t):
    """与 multidata.observables_np 完全一致的 torch 版本。"""
    Vu, Vd, S = flav["Vu"], flav["Vd"], flav["S"]
    f2p = x_t * (4.0 / 9.0 * (Vu + 2.0 * S) + 1.0 / 9.0 * (Vd + 2.0 * S))
    f2n = x_t * (4.0 / 9.0 * (Vd + 2.0 * S) + 1.0 / 9.0 * (Vu + 2.0 * S))
    return {
        "F2p": f2p,
        "F2d": 0.5 * (f2p + f2n),
        "xF3vp": 2.0 * x_t * Vd,
        "xF3vbp": 2.0 * x_t * Vu,
    }


def dataset_prediction(flavs, x_t, q2_keys, d):
    pred = torch.zeros_like(d["y"])
    for qi, q in enumerate(q2_keys):
        mask = d["q2i"] == qi
        if mask.any():
            grid = obs_torch(flavs[q], x_t)[d["name"]]
            pred[mask] = nmod.interp1d(x_t, grid, d["x"][mask])
    return pred


def profile_chi2(resid, pred, B, sigma, sigma_beta):
    """解析 profile 关联 nuisance：返回 (chi2_prof, lambda_best)。"""
    w = 1.0 / sigma ** 2
    A = pred.unsqueeze(1) * B
    M = A.transpose(0, 1) @ (w.unsqueeze(1) * A) + torch.diag(1.0 / sigma_beta ** 2)
    v = A.transpose(0, 1) @ (w * resid)
    lam = torch.linalg.solve(M, v)
    chi2 = torch.sum(w * resid ** 2) - v @ lam
    return chi2, lam


def make_tensors(datasets, x_t, q2_keys):
    out = {}
    for name, d in datasets.items():
        q2_key_index = {q: i for i, q in enumerate(q2_keys)}
        out[name] = dict(
            name=name,
            x=torch.tensor(d["x"], dtype=torch.float64),
            q2i=torch.tensor([q2_key_index[float(q)] for q in d["q2"]],
                             dtype=torch.long),
            y=torch.tensor(d["y"], dtype=torch.float64),
            t_true=torch.tensor(d["t_true"], dtype=torch.float64),
            sigma=torch.tensor(d["sigma"], dtype=torch.float64),
            B=torch.tensor(d["B"], dtype=torch.float64),
            sigma_beta=torch.tensor(d["sigma_beta"], dtype=torch.float64),
        )
    return out


# ---------------------------------------------------------------------------
# 单个 replica
# ---------------------------------------------------------------------------
def fit_replica(x_t, wx_t, U_t, data_t, q2_keys, seed, epochs=1200, lr=0.02,
                w_sum=100.0, w_smooth=1e-3, w_exp=0.1,
                lbfgs_iters=300, width=24, temp=1.0, verbose=False):
    torch.manual_seed(seed)
    net = nmod.NeuralPDF(width=width).double()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n_total = sum(len(d["y"]) for d in data_t.values())
    x_t_log = torch.log(x_t)
    dl = x_t_log[1:] - x_t_log[:-1]

    def objective():
        h, f0, flavs = evolved_flavors(net, x_t, U_t)
        Vu, Vd, S, g = f0
        chi2_sum = 0.0
        for d in data_t.values():
            pred = dataset_prediction(flavs, x_t, q2_keys, d)
            c2, _ = profile_chi2(d["y"] - pred, pred, d["B"],
                                 d["sigma"], d["sigma_beta"])
            chi2_sum = chi2_sum + c2
        chi2 = chi2_sum / n_total

        num_u = torch.sum(wx_t * Vu)
        num_d = torch.sum(wx_t * Vd)
        mom = torch.sum(wx_t * x_t * (Vu + Vd + 4.0 * S + g))
        loss_sum = (num_u - 2.0) ** 2 + (num_d - 1.0) ** 2 + (mom - 1.0) ** 2
        smooth = 0.0
        for k in range(4):
            hk = h[:, k]
            d2 = (hk[2:] - 2.0 * hk[1:-1] + hk[:-2]) / dl[1:] ** 2
            smooth = smooth + torch.mean((dl[:-1] * d2) ** 2)
        # 似然温度化：chi2/T 等价于把数据方差放大 T 倍，展宽后验
        loss = chi2 / temp + w_sum * loss_sum + w_smooth * smooth \
            + w_exp * torch.sum(net.exp_delta ** 2)
        return loss, chi2

    for ep in range(epochs):
        opt.zero_grad()
        loss, chi2 = objective()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and ep % 200 == 0:
            print(f"  ep {ep:5d} chi2/N={chi2.item():.3f}")

    if lbfgs_iters > 0:
        lb = torch.optim.LBFGS(net.parameters(), lr=0.5, max_iter=lbfgs_iters,
                               line_search_fn="strong_wolfe",
                               history_size=50, tolerance_grad=1e-9,
                               tolerance_change=1e-11)

        def closure():
            lb.zero_grad()
            loss, _ = objective()
            loss.backward()
            return loss
        lb.step(closure)
        with torch.no_grad():
            _, chi2 = objective()
    return net, chi2.item()


def predict_replica(net, x_t, U_t, data_t, q2_keys):
    """返回 Q0 四味 (4,N) 与各数据集的预言，均为 numpy。"""
    with torch.no_grad():
        _, f0, flavs = evolved_flavors(net, x_t, U_t)
        preds = {name: dataset_prediction(flavs, x_t, q2_keys, d).cpu().numpy()
                 for name, d in data_t.items()}
        f0_np = f0.cpu().numpy()
    return f0_np, preds


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rep", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--lbfgs", type=int, default=300)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--smooth", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=os.path.join(os.getcwd(), "outputs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(args.seed)

    x, w, wx, mats, pdf0 = pdata.build()
    datasets, evolved_truth = mdata.build_datasets(x, w, mats, pdf0,
                                                  seed=args.seed)
    q2_list = sorted(set(float(q) for d in datasets.values() for q in d["q2"]))
    x_t = torch.tensor(x, dtype=torch.float64)
    wx_t = torch.tensor(wx, dtype=torch.float64)
    U = nmod.evolution_operators(x, w, mats, q2_list)
    U_t = {float(q): torch.tensor(Uq, dtype=torch.float64)
           for q, Uq in U.items()}
    q2_keys = sorted(U_t.keys())
    data_t = make_tensors(datasets, x_t, q2_keys)

    print("数据集：")
    for name, d in datasets.items():
        print(f"  {name:7s} {len(d['y']):4d} 点  "
              f"统计误差 {100*np.median(d['sigma']/np.abs(d['t_true'])):.1f}%  "
              f"关联系统 {100*d['sigma_beta']} %")

    f0_list, pred_lists = [], {name: [] for name in datasets}
    chi2_noise_list = []
    per_dataset_chi2 = {name: [] for name in datasets}
    t0 = time.time()
    for r in range(args.n_rep):
        rep_data = {name: dict(y=torch.tensor(mdata.draw_replica(d, rng),
                                              dtype=torch.float64),
                               **{k: v for k, v in data_t[name].items()
                                  if k != "y"})
                    for name, d in datasets.items()}
        # draw_replica 用 numpy 的 d，而 rep_data 保留其余张量字段
        net, chi2_noise = fit_replica(x_t, wx_t, U_t, rep_data, q2_keys,
                                      seed=args.seed + 1000 + r,
                                      epochs=args.epochs,
                                      lbfgs_iters=args.lbfgs,
                                      width=args.width,
                                      w_smooth=args.smooth)
        f0_np, preds = predict_replica(net, x_t, U_t, data_t, q2_keys)
        f0_list.append(f0_np)
        for name in datasets:
            pred_lists[name].append(preds[name])
        chi2_noise_list.append(chi2_noise)

        # 该 replica 对照中心数据的每数据集 chi2/N
        msg = []
        for name, d in data_t.items():
            pred = torch.tensor(preds[name], dtype=torch.float64)
            c2, _ = profile_chi2(d["y"] - pred, pred, d["B"],
                                 d["sigma"], d["sigma_beta"])
            c2n = float(c2) / len(d["y"])
            per_dataset_chi2[name].append(c2n)
            msg.append(f"{name}={c2n:.3f}")
        print(f"replica {r + 1:3d}/{args.n_rep}  chi2(noise)={chi2_noise:.3f}  "
              + "  ".join(msg) + f"  累计 {time.time() - t0:.0f}s")

        # 检查点：已完成的 replica 立即落盘，防止中途丢失
        np.savez(os.path.join(args.out, "checkpoint_m05.npz"),
                 f0=np.stack(f0_list),
                 **{f"pred_{name}": np.stack(v) for name, v in pred_lists.items()})

    # ---- 汇总 ----
    f0_stack = np.stack(f0_list)                 # (R,4,N)
    f0_mean = f0_stack.mean(axis=0)
    f0_std = f0_stack.std(axis=0)
    n_rep = len(f0_list)

    # 每数据集：集合预言均值/标准差、profile chi2、pulls、真值覆盖率
    metrics = {}
    save_arrays = {}
    all_pulls = []
    for name, d in datasets.items():
        stack = np.stack(pred_lists[name])       # (R,Nd)
        p_mean = stack.mean(axis=0)
        p_std = stack.std(axis=0)

        def prof_np(resid, pred):
            wgt = 1.0 / d["sigma"] ** 2
            A = pred[:, None] * d["B"]
            M = A.T @ (wgt[:, None] * A) + np.diag(1.0 / d["sigma_beta"] ** 2)
            v = A.T @ (wgt * resid)
            lam = np.linalg.solve(M, v)
            chi2 = np.sum(wgt * resid ** 2) - v @ lam
            return float(chi2), lam

        chi2_ens, lam = prof_np(d["y"] - p_mean, p_mean)
        p_shift = p_mean * (1.0 + d["B"] @ lam)
        pulls = (d["y"] - p_shift) / d["sigma"]
        cover_truth = float(np.mean(np.abs(p_mean - d["t_true"]) <= p_std))
        all_pulls.append(pulls)

        truth_chi2, _ = prof_np(d["y"] - d["t_true"], d["t_true"])
        truth_chi2_stat = float(np.mean(
            ((d["y"] - d["t_true"]) / d["sigma"]) ** 2))
        metrics[name] = dict(
            n_points=len(d["y"]),
            truth_chi2_per_point=truth_chi2 / len(d["y"]),
            truth_chi2_stat_per_point=truth_chi2_stat,
            chi2_per_point=chi2_ens / len(d["y"]),
            chi2_replica_mean=float(np.mean(per_dataset_chi2[name])),
            chi2_replica_std=float(np.std(per_dataset_chi2[name])),
            pull_mean=float(np.mean(pulls)),
            pull_std=float(np.std(pulls)),
            truth_coverage_1sigma=cover_truth,
            nuisance_best=lam.tolist(),
        )
        save_arrays[f"{name}_x"] = d["x"]
        save_arrays[f"{name}_q2"] = d["q2"]
        save_arrays[f"{name}_y"] = d["y"]
        save_arrays[f"{name}_sigma"] = d["sigma"]
        save_arrays[f"{name}_t_true"] = d["t_true"]
        save_arrays[f"{name}_pred_mean"] = p_mean
        save_arrays[f"{name}_pred_std"] = p_std
        save_arrays[f"{name}_pulls"] = pulls

    pulls_all = np.concatenate(all_pulls)

    # 求和规则
    sumrules = {
        "num_u": float(np.sum(wx * f0_mean[0])),
        "num_d": float(np.sum(wx * f0_mean[1])),
        "mom": float(np.sum(wx * x * f0_mean.sum(axis=0))),
    }

    # Q0 PDF 在数据区间的覆盖率
    truth0 = np.stack([pdf0["Vu"], pdf0["Vd"], pdf0["S"], pdf0["g"]])
    xmask = (x >= mdata.X_BINS.min()) & (x <= mdata.X_BINS.max())
    inside = np.abs(f0_mean - truth0) <= f0_std
    flav_names = ["Vu", "Vd", "S", "g"]
    q0_cov = {flav_names[k]: float(np.mean(inside[k, xmask])) for k in range(4)}
    q0_cov["all"] = float(np.mean(inside[:, xmask]))

    # 与 M0 的 PDF 不确定度对比（若 M0 输出存在）
    unc_ratio = None
    m0_npz = os.path.join(os.getcwd(), "outputs", "replicas.npz")
    if os.path.exists(m0_npz):
        m0 = np.load(m0_npz)
        m0_std = m0["f0_std"]
        ratio = np.median(f0_std[:, xmask] / np.maximum(m0_std[:, xmask], 1e-12),
                          axis=1)
        unc_ratio = {flav_names[k]: float(ratio[k]) for k in range(4)}

    # g-S 简并：在若干 x 处计算 replicas 的 Pearson 相关系数
    def x_index(target):
        return int(np.argmin(np.abs(np.log10(x) - np.log10(target))))
    corr = {}
    for xt in (1e-3, 1e-2, 0.1):
        k = x_index(xt)
        gk = f0_stack[:, 3, k]
        sk = f0_stack[:, 2, k]
        corr[f"{xt:g}"] = float(np.corrcoef(gk, sk)[0, 1])
    corr_matrix = {}
    for xt in (1e-2,):
        k = x_index(xt)
        C = np.corrcoef(f0_stack[:, :, k].T)
        corr_matrix[f"{xt:g}"] = C.tolist()

    # 集合平均的全局 chi2/N（按点平均）
    n_tot = sum(m["n_points"] for m in metrics.values())
    chi2_global = sum(m["chi2_per_point"] * m["n_points"]
                      for m in metrics.values()) / n_tot

    closure = dict(
        n_rep=n_rep,
        n_points_total=int(n_tot),
        chi2_global_per_point=float(chi2_global),
        chi2_noise_mean=float(np.mean(chi2_noise_list)),
        chi2_noise_std=float(np.std(chi2_noise_list)),
        per_dataset=metrics,
        pulls_all_mean=float(np.mean(pulls_all)),
        pulls_all_std=float(np.std(pulls_all)),
        sumrules=sumrules,
        q0_coverage_1sigma=q0_cov,
        uncertainty_ratio_vs_M0=unc_ratio,
        gS_correlation=corr,
        q0_corr_matrix_x1e2=corr_matrix,
    )

    np.savez(os.path.join(args.out, "replicas_m05.npz"),
             x=x, wx=wx, f0_mean=f0_mean, f0_std=f0_std, truth0=truth0,
             f0_replicas=f0_stack, pulls_all=pulls_all, **save_arrays)
    with open(os.path.join(args.out, "closure_m05.json"), "w",
              encoding="utf-8") as f:
        json.dump(closure, f, ensure_ascii=False, indent=2)
    try:
        os.remove(os.path.join(args.out, "checkpoint_m05.npz"))
    except OSError:
        pass

    # ---- 报告 ----
    print("\n== M0.5 闭环验收 ==")
    print(f"{'数据集':8s} {'N':>4s} {'真值chi2/N':>10s} {'chi2/N(ens)':>11s} "
          f"{'chi2/N(rep)':>11s} {'pull mean':>9s} {'pull std':>8s} "
          f"{'真值覆盖':>8s}")
    for name, m in metrics.items():
        print(f"{name:8s} {m['n_points']:4d} {m['truth_chi2_per_point']:10.3f} "
              f"{m['chi2_per_point']:11.3f} {m['chi2_replica_mean']:11.3f} "
              f"{m['pull_mean']:9.3f} {m['pull_std']:8.3f} "
              f"{m['truth_coverage_1sigma']*100:7.0f}%")
    print(f"全局 chi2/N={chi2_global:.3f}；pull(全部) 均值={np.mean(pulls_all):.3f} "
          f"标准差={np.std(pulls_all):.3f}")
    print(f"求和规则：∫Vu={sumrules['num_u']:.4f}(2)  "
          f"∫Vd={sumrules['num_d']:.4f}(1)  动量={sumrules['mom']:.4f}(1)")
    print("Q0 PDF 1σ 覆盖率：" + "  ".join(
        f"{k}={100*v:.0f}%" for k, v in q0_cov.items()))
    if unc_ratio:
        print("PDF 不确定度 / M0：" + "  ".join(
            f"{k}={v:.2f}" for k, v in unc_ratio.items()))
    print("g-S 相关系数：" + "  ".join(
        f"x={k}: r={v:+.2f}" for k, v in corr.items()))
    print(f"结果已保存至 {args.out}/replicas_m05.npz, closure_m05.json")


if __name__ == "__main__":
    main()
