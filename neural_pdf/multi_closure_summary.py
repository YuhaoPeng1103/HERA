# -*- coding: utf-8 -*-
"""汇总多伪实验 closure：覆盖率分布与论文级图。"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NPTS = {"F2p": 98, "F2d": 98, "xF3vp": 70, "xF3vbp": 70}


def stats(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return None
    return dict(n=int(v.size), mean=float(v.mean()), std=float(v.std(ddof=1)),
                median=float(np.median(v)), q16=float(np.quantile(v, 0.16)),
                q84=float(np.quantile(v, 0.84)), min=float(v.min()),
                max=float(v.max()))


def weighted_mean(e, key):
    num = den = 0.0
    for n, k in NPTS.items():
        if key in e["data"].get(n, {}):
            num += e["data"][n][key] * k
            den += k
    return num / den if den else None


def collect(ddir):
    files = sorted(glob.glob(os.path.join(ddir, "exp_*.json")))
    exps = [json.load(open(f, encoding="utf-8")) for f in files]
    q0_s1 = [e["q0"]["single_1sigma"] for e in exps]
    q0_m1 = [e["q0"]["mean_1sigma"] for e in exps]
    q0_s90 = [e["q0"]["single_90"] for e in exps]
    q0_m90 = [e["q0"]["mean_90"] for e in exps]
    d_s1 = [e["data_all_single_1sigma"] for e in exps]
    d_m1 = [e.get("data_all_mean_1sigma", weighted_mean(e, "mean_1sigma"))
            for e in exps]
    d_s90 = [e["data_all_single_90"] for e in exps]
    d_m90 = [e.get("data_all_mean_90", weighted_mean(e, "mean_90"))
             for e in exps]
    d_m1 = [v for v in d_m1 if v is not None]
    d_m90 = [v for v in d_m90 if v is not None]
    chi2 = [e["chi2_noise_mean"] for e in exps]
    return dict(
        n_exp=len(exps),
        n_rep=int(exps[0]["n_rep"]) if exps else 0,
        q0_single_1sigma=stats(q0_s1), q0_mean_1sigma=stats(q0_m1),
        q0_single_90=stats(q0_s90), q0_mean_90=stats(q0_m90),
        data_single_1sigma=stats(d_s1), data_mean_1sigma=stats(d_m1),
        data_single_90=stats(d_s90), data_mean_90=stats(d_m90),
        chi2_noise=stats(chi2),
    ), (q0_s1, q0_m1, q0_s90, q0_m90, d_s1, d_m1, d_s90, d_m90)


def hist_panel(ax, s, m, nominal, title, tag):
    bins = np.linspace(0.30, 1.00, 29)
    ax.hist(s, bins=bins, alpha=0.55, color="#1f77b4", edgecolor="white",
            label="single replica")
    if m is not None and len(m):
        ax.hist(m, bins=bins, alpha=0.35, color="#d62728", edgecolor="white",
                label="ensemble mean")
    ax.axvline(nominal, color="k", ls="--", lw=1.2,
               label=f"nominal {nominal:.0%}")
    txt = f"single: {np.mean(s):.3f} ± {np.std(s, ddof=1):.3f}"
    if m is not None and len(m):
        txt += f"\nmean: {np.mean(m):.3f} ± {np.std(m, ddof=1):.3f}"
    ax.text(0.03, 0.96, txt, transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xlabel("per-experiment coverage")
    ax.set_ylabel("number of pseudo-experiments")
    ax.set_title(f"{tag} {title}", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(
        os.getcwd(), "outputs", "multi_closure"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    outdir = args.out or args.dir
    summary, vals = collect(args.dir)
    q0_s1, q0_m1, q0_s90, q0_m90, d_s1, d_m1, d_s90, d_m90 = vals

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    hist_panel(axes[0, 0], q0_s1, q0_m1, 0.68, "Q0 PDF, 1σ", "(a)")
    hist_panel(axes[0, 1], d_s1, d_m1, 0.68, "data points, 1σ", "(b)")
    hist_panel(axes[1, 0], q0_s90, q0_m90, 0.90, "Q0 PDF, 90%", "(c)")
    hist_panel(axes[1, 1], d_s90, d_m90, 0.90, "data points, 90%", "(d)")
    fig.suptitle(f"Closure coverage across {summary['n_exp']} pseudo-experiments "
                 f"({summary['n_rep']} replicas each)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(outdir, "multi_closure_summary.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)

    with open(os.path.join(outdir, "multi_closure_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"== 多伪实验 closure（{summary['n_exp']} 个伪实验）==")
    for key, label, nom in [
            ("q0_single_1sigma", "Q0 PDF 单 replica 1σ", 0.68),
            ("q0_mean_1sigma", "Q0 PDF ensemble 均值 1σ", 0.68),
            ("q0_single_90", "Q0 PDF 单 replica 90%", 0.90),
            ("q0_mean_90", "Q0 PDF ensemble 均值 90%", 0.90),
            ("data_single_1sigma", "数据点 单 replica 1σ", 0.68),
            ("data_mean_1sigma", "数据点 ensemble 均值 1σ", 0.68),
            ("data_single_90", "数据点 单 replica 90%", 0.90),
            ("data_mean_90", "数据点 ensemble 均值 90%", 0.90)]:
        st = summary[key]
        if st:
            print(f"  {label:28s} mean={st['mean']:.3f} ± {st['std']:.3f}  "
                  f"[{st['q16']:.3f}, {st['q84']:.3f}]  (nominal {nom:.2f})")
    print(f"已保存：{path} 与 multi_closure_summary.json")


if __name__ == "__main__":
    main()
