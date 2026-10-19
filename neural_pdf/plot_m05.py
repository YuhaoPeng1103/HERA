# -*- coding: utf-8 -*-
"""
M0.5 结果可视化
================
1. Q0 四味 PDF：M0.5（多观测+系统误差）vs M0（仅 F2p）vs 真值；
2. 四个数据集的拟合带、伪数据与真值（F2^p, F2^d, xF3^{nu p}, xF3^{nubar p}）；
3. 各数据集 pulls 分布；
4. g-S 简并：replicas 在固定 x 处的散点与相关系数。
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _cand in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
              "Arial Unicode MS"):
    if any(_cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_cand, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

OBS_LABEL = {
    "F2p": r"$F_2^p$",
    "F2d": r"$F_2^d$",
    "xF3vp": r"$xF_3^{\nu p}$",
    "xF3vbp": r"$xF_3^{\bar\nu p}$",
}
FLAV = [r"$V_u$", r"$V_d$", r"$S$", r"$g$"]
COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]


def plot_pdfs(d, m0, out):
    x = d["x"]
    has_m0 = m0 is not None
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for k, ax in enumerate(axes.ravel()):
        m, s = d["f0_mean"][k], d["f0_std"][k]
        ax.fill_between(x, m - s, m + s, color=COLORS[k], alpha=0.20,
                        label="M0.5 68% 带")
        ax.plot(x, m, color=COLORS[k], lw=1.6, label="M0.5 中心")
        if has_m0:
            m0m, m0s = m0["f0_mean"][k], m0["f0_std"][k]
            ax.fill_between(x, m0m - m0s, m0m + m0s, color="gray", alpha=0.18,
                            label="M0 68% 带（仅 F2p）")
            ax.plot(x, m0m, color="gray", lw=1.0, ls=":")
        ax.plot(x, d["truth0"][k], "k--", lw=1.2, label="真值")
        ax.axvspan(x.min(), 1.6e-3, color="gray", alpha=0.08, lw=0)
        ax.axvline(1.6e-3, color="gray", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(x.min(), 1.0)
        ax.set_ylabel(FLAV[k] + r"$(x,Q_0^2)$")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=7)
            ax.text(1.8e-4, 0.97, "无数据外推区", transform=ax.get_xaxis_transform(),
                    fontsize=8, color="gray", va="top")
    for ax in axes[-1]:
        ax.set_xlabel("x")
    fig.suptitle("Q0 处 PDF：多观测量 + 关联系统误差 vs M0")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pdf_q0_m05.png"), dpi=300)
    plt.close(fig)


def plot_observables(d, out, q2_show=4.0):
    names = ["F2p", "F2d", "xF3vp", "xF3vbp"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, name in zip(axes.ravel(), names):
        m = d[f"{name}_q2"] == q2_show
        ax.errorbar(d[f"{name}_x"][m], d[f"{name}_y"][m],
                    yerr=d[f"{name}_sigma"][m], fmt="o", ms=3,
                    color="red", alpha=0.75, label="伪数据")
        pm = d[f"{name}_pred_mean"][m]
        ps = d[f"{name}_pred_std"][m]
        tt = d[f"{name}_t_true"][m]
        ax.fill_between(d[f"{name}_x"][m], pm - ps, pm + ps,
                        color="#1f77b4", alpha=0.25, label="M0.5 68% 带")
        ax.plot(d[f"{name}_x"][m], pm, color="#1f77b4", lw=1.5, label="拟合")
        ax.plot(d[f"{name}_x"][m], tt, "k--", lw=1.1, label="真值")
        ax.set_xscale("log")
        ax.set_title(OBS_LABEL[name] + rf"  ($Q^2={q2_show:g}$ GeV$^2$)",
                     fontsize=10)
        ax.set_ylabel(OBS_LABEL[name])
        ax.grid(alpha=0.3)
        if name == "F2p":
            ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("x")
    fig.suptitle("多观测量联合拟合：伪数据、神经拟合带与真值")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "observables_m05.png"), dpi=300)
    plt.close(fig)


def plot_pulls(d, out):
    names = ["F2p", "F2d", "xF3vp", "xF3vbp"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    z = np.linspace(-4, 4, 200)
    for ax, name in zip(axes.ravel(), names):
        p = d[f"{name}_pulls"]
        ax.hist(p, bins=np.linspace(-4, 4, 33), color="#1f77b4",
                alpha=0.75, edgecolor="white", density=True)
        ax.plot(z, np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi), "r--",
                label="N(0,1)")
        ax.set_title(OBS_LABEL[name] + f"：均值 {p.mean():+.2f}，"
                     f"σ {p.std():.2f}", fontsize=10)
        ax.set_xlabel("pull")
        ax.set_ylabel("概率密度")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("各数据集 pulls（profile 掉关联系统误差后）")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pulls_m05.png"), dpi=300)
    plt.close(fig)


def plot_gS(d, npz_path, m0, out, x_target=1e-2):
    x = d["x"]
    k = int(np.argmin(np.abs(np.log10(x) - np.log10(x_target))))
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    # M0.5 replicas 未直接保存，用均值/标准差无法画散点；
    # 因此从 checkpoint 或 replicas 文件中读取单 replica 值（若存在）。
    if os.path.exists(npz_path):
        raw = np.load(npz_path)
        key = "f0_replicas" if "f0_replicas" in raw.files else "f0"
        if key in raw.files:
            f0 = raw[key]
            g0, s0 = d["truth0"][3, k], d["truth0"][2, k]
            ax.scatter((f0[:, 2, k] - s0) / s0, (f0[:, 3, k] - g0) / g0,
                       s=22, alpha=0.75, color="#1f77b4", label="M0.5")
            if f0.shape[0] > 2:
                r = np.corrcoef(f0[:, 2, k], f0[:, 3, k])[0, 1]
                ax.text(0.03, 0.95, f"M0.5 ({f0.shape[0]} replicas): r = {r:+.2f}",
                        transform=ax.transAxes, fontsize=10, color="#1f77b4")
    if m0 is not None:
        g0, s0 = d["truth0"][3, k], d["truth0"][2, k]
        ax.scatter((m0["f0_mean"][2, k] - s0) / s0,
                   (m0["f0_mean"][3, k] - g0) / g0, marker="x", s=40,
                   color="gray", label="M0 中心")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$(S-S_{\rm true})/S_{\rm true}$")
    ax.set_ylabel(r"$(g-g_{\rm true})/g_{\rm true}$")
    ax.set_title(rf"$x={x_target:g}$ 处 g–S 简并")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gS_correlation_m05.png"), dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(
        os.getcwd(), "outputs", "replicas_m05.npz"))
    ap.add_argument("--m0", default=os.path.join(
        os.getcwd(), "outputs", "replicas.npz"))
    ap.add_argument("--checkpoint", default=os.path.join(
        os.getcwd(), "outputs", "checkpoint_m05.npz"))
    ap.add_argument("--out", default=os.path.join(os.getcwd(), "outputs"))
    args = ap.parse_args()
    d = np.load(args.npz)
    m0 = np.load(args.m0) if os.path.exists(args.m0) else None
    plot_pdfs(d, m0, args.out)
    plot_observables(d, args.out)
    plot_pulls(d, args.out)
    plot_gS(d, args.npz, m0, args.out)
    print(f"已生成：{args.out}/pdf_q0_m05.png, observables_m05.png, "
          f"pulls_m05.png, gS_correlation_m05.png")


if __name__ == "__main__":
    main()
