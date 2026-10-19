# -*- coding: utf-8 -*-
"""HERA stress-test 出图：数据/拟合、pulls。"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=os.path.join(
        os.getcwd(), "outputs", "restricted_nf3", "hera_fit.npz"))
    ap.add_argument("--data", default=os.path.join(
        os.getcwd(), "outputs", "hera_selected_restricted.npz"))
    ap.add_argument("--json", default=None)
    ap.add_argument("--out", default=os.path.join(
        os.getcwd(), "outputs", "hera_stress_test.png"))
    args = ap.parse_args()
    d = np.load(args.fit)
    dd = np.load(args.data)
    q2, x, y, pred = d["data_q2"], d["data_x"], d["data_y"], d["data_pred"]
    pulls = d["pulls"]
    err = np.sqrt(np.diag(dd["C"]))
    jf = args.json or os.path.join(os.path.dirname(args.fit), "hera_fit.json")
    meta = json.load(open(jf)) if os.path.exists(jf) else {}

    fig = plt.figure(figsize=(13, 4.6))
    ax1 = fig.add_subplot(1, 3, (1, 2))
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, 4))
    qs = [2.0, 3.5, 6.5, 10.0]
    for c, q in zip(colors, qs):
        m = np.isclose(q2, q)
        if not m.any():
            continue
        ax1.errorbar(x[m], y[m], yerr=err[m], fmt="o", ms=4, capsize=2,
                     color=c, label=rf"$Q^2={q:g}$ GeV$^2$")
        ax1.plot(x[m], pred[m], "-", color=c, lw=1.2)
    ax1.set_xscale("log")
    ax1.set_xlabel("x")
    ax1.set_ylabel(r"$\sigma_{r,\rm NC}$")
    ax1.set_title("(a) HERA restricted subset: data (points) and LO fit (lines)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(1, 3, 3)
    ax2.hist(pulls, bins=np.linspace(-4, 4, 25), density=True,
             color="#1f77b4", alpha=0.75, edgecolor="white")
    z = np.linspace(-4, 4, 200)
    ax2.plot(z, np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi), "r--",
             label="N(0,1)")
    ax2.set_xlabel("whitened pull")
    ax2.set_ylabel("density")
    ax2.set_title(f"(b) pulls: mean {pulls.mean():+.2f}, "
                  f"$\\sigma$ {pulls.std():.2f}")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    chi2 = meta.get("chi2_central", float("nan"))
    fig.suptitle(f"HERA stress test ($LO$, $n_f=3$, $\\sigma_r\\simeq F_2$): "
                 f"$\\chi^2/N={chi2:.2f}$", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"已生成：{args.out}")


if __name__ == "__main__":
    main()
