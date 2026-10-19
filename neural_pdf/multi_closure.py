# -*- coding: utf-8 -*-
"""
多伪实验 closure（Phase 1）
===========================
每个伪实验生成一套独立的中心伪数据（不同随机种子），训练一套 replicas，
统计 **单 replica** 与 **ensemble 均值** 的覆盖率。

用法（并行时用 --exp-start/--exp-end 切分互不重叠的伪实验编号）：
    python multi_closure.py --exp-start 0 --exp-end 5 --n-rep 12 --threads 4
输出：outputs/multi_closure/exp_{i}.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from . import model as nmod
from . import multidata as mdata
from . import pseudodata as pdata
from . import train_m05 as T


def cov(z, k=1.0):
    return float(np.mean(np.abs(z) <= k))


def z_quantiles(z):
    a = np.abs(np.asarray(z)).ravel()
    return {f"q{int(q*100)}": float(np.quantile(a, q))
            for q in (0.68, 0.90, 0.95)}


def z_profile(z, probs=None):
    """细网格经验分位数：用于拟合 k 依赖的带宽膨胀因子。"""
    a = np.abs(np.asarray(z)).ravel()
    if probs is None:
        probs = np.arange(0.50, 0.991, 0.02)
    return {"p": [round(float(x), 4) for x in probs],
            "q": [float(np.quantile(a, x)) for x in probs]}


def prof_chi2_np(d, pred):
    w = 1.0 / d["sigma"] ** 2
    A = pred[:, None] * d["B"]
    M = A.T @ (w[:, None] * A) + np.diag(1.0 / d["sigma_beta"] ** 2)
    r = d["y"] - pred
    v = A.T @ (w * r)
    return float(np.sum(w * r ** 2) - v @ np.linalg.solve(M, v))


def run_experiment(exp_id, exp_seed, args, x, w, wx, mats, pdf0,
                   U_t, q2_keys, x_t, wx_t):
    t_start = time.time()
    datasets, _ = mdata.build_datasets(x, w, mats, pdf0, seed=exp_seed)
    data_t = T.make_tensors(datasets, x_t, q2_keys)
    rng = np.random.default_rng(exp_seed * 100 + 17)

    f0_list, chi2_list = [], []
    preds = {name: [] for name in datasets}
    for r in range(args.n_rep):
        rep_data = {}
        for name, d in datasets.items():
            y = mdata.draw_replica(d, rng)
            rep_data[name] = dict(
                y=torch.tensor(y, dtype=torch.float64),
                **{k: v for k, v in data_t[name].items() if k != "y"})
        net, chi2 = T.fit_replica(
            x_t, wx_t, U_t, rep_data, q2_keys,
            seed=exp_seed * 1000 + r,
            epochs=args.epochs, lbfgs_iters=args.lbfgs,
            width=args.width, w_smooth=args.smooth, temp=args.temp)
        f0, pred = T.predict_replica(net, x_t, U_t, data_t, q2_keys)
        f0_list.append(f0)
        chi2_list.append(chi2)
        for name in datasets:
            preds[name].append(pred[name])

    f0s = np.stack(f0_list)                      # (R,4,N)
    truth0 = np.stack([pdf0["Vu"], pdf0["Vd"], pdf0["S"], pdf0["g"]])
    xmask = (x >= mdata.X_BINS.min()) & (x <= mdata.X_BINS.max())
    s0 = f0s.std(axis=0)
    zr0 = (f0s - truth0[None]) / np.maximum(s0[None], 1e-30)
    zm0 = (f0s.mean(axis=0) - truth0) / np.maximum(s0, 1e-30)

    q0 = dict(
        single_1sigma=cov(zr0[:, :, xmask]), single_90=cov(zr0[:, :, xmask], 1.6449),
        single_rms=float(np.sqrt(np.mean(zr0[:, :, xmask] ** 2))),
        mean_1sigma=cov(zm0[:, xmask]), mean_90=cov(zm0[:, xmask], 1.6449),
        z_quantiles=z_quantiles(zr0[:, :, xmask]),
        z_profile=z_profile(zr0[:, :, xmask]),
        per_flavor_1sigma={
            n: cov(zr0[:, k, xmask]) for k, n in enumerate(["Vu", "Vd", "S", "g"])},
    )
    data = {}
    for name, d in datasets.items():
        stack = np.stack(preds[name])             # (R,Nd)
        s = stack.std(axis=0)
        zr = (stack - d["t_true"][None]) / np.maximum(s[None], 1e-30)
        zm = (stack.mean(axis=0) - d["t_true"]) / np.maximum(s, 1e-30)
        data[name] = dict(
            single_1sigma=cov(zr), single_90=cov(zr, 1.6449),
            single_rms=float(np.sqrt(np.mean(zr ** 2))),
            mean_1sigma=cov(zm), mean_90=cov(zm, 1.6449),
            z_profile=z_profile(zr),
            truth_chi2=prof_chi2_np(d, d["t_true"]) / len(d["y"]),
            fit_chi2=prof_chi2_np(d, stack.mean(axis=0)) / len(d["y"]),
        )

    n_tot = sum(len(d["y"]) for d in datasets.values())
    z_all = np.concatenate([
        ((np.stack(preds[n]) - datasets[n]["t_true"][None])
         / np.maximum(np.stack(preds[n]).std(axis=0)[None], 1e-30)).ravel()
        for n in datasets])
    z_all_mean = np.concatenate([
        ((np.stack(preds[n]).mean(axis=0) - datasets[n]["t_true"])
         / np.maximum(np.stack(preds[n]).std(axis=0), 1e-30)).ravel()
        for n in datasets])
    out = dict(
        exp_id=exp_id, exp_seed=exp_seed, n_rep=args.n_rep,
        q0=q0,
        data=data,
        data_all_single_1sigma=cov(z_all),
        data_all_single_90=cov(z_all, 1.6449),
        data_all_mean_1sigma=cov(z_all_mean),
        data_all_mean_90=cov(z_all_mean, 1.6449),
        data_all_z_quantiles=z_quantiles(z_all),
        data_all_z_profile=z_profile(z_all),
        chi2_noise_mean=float(np.mean(chi2_list)),
        chi2_noise_std=float(np.std(chi2_list)),
        elapsed=time.time() - t_start,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-start", type=int, default=0)
    ap.add_argument("--exp-end", type=int, default=2)
    ap.add_argument("--base-seed", type=int, default=10000)
    ap.add_argument("--n-rep", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--lbfgs", type=int, default=300)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--smooth", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(
        os.getcwd(), "outputs", "multi_closure"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)

    x, w, wx, mats, pdf0 = pdata.build()
    q2_list = mdata.Q2_GRID.tolist()
    q2_keys = sorted(set(float(q) for q in q2_list))
    U = nmod.evolution_operators(x, w, mats, q2_keys)
    U_t = {float(q): torch.tensor(Uq, dtype=torch.float64)
           for q, Uq in U.items()}
    x_t = torch.tensor(x, dtype=torch.float64)
    wx_t = torch.tensor(wx, dtype=torch.float64)

    for exp_id in range(args.exp_start, args.exp_end):
        exp_seed = args.base_seed + exp_id
        out = run_experiment(exp_id, exp_seed, args, x, w, wx, mats, pdf0,
                             U_t, q2_keys, x_t, wx_t)
        path = os.path.join(args.out, f"exp_{exp_id:03d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[exp {exp_id:03d}] Q0 single={100*out['q0']['single_1sigma']:.1f}% "
              f"mean={100*out['q0']['mean_1sigma']:.1f}% "
              f"data single={100*out['data_all_single_1sigma']:.1f}% "
              f"({out['elapsed']:.0f}s) -> {path}", flush=True)


if __name__ == "__main__":
    main()
