# -*- coding: utf-8 -*-
"""
校准修正的独立验证
==================
用前 5 个伪实验拟合 k 依赖的带宽膨胀因子 kappa(k)：
    kappa(k) = k / q_emp(p_nom(k)),   p_nom(k)=P(|N(0,1)|<=k)
然后在后 5 个**独立**伪实验上应用 kappa，检查 1σ/90% 覆盖率是否回到
68%/90%。同时报告逐数据集的 kappa。
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_exps(ddir):
    exps = {}
    for f in sorted(glob.glob(os.path.join(ddir, "exp_*.json"))):
        e = json.load(open(f, encoding="utf-8"))
        exps[int(e["exp_id"])] = e
    return exps


def pooled_profile(exps, getter):
    p = np.asarray(getter(exps[0])["p"], dtype=float)
    q = np.stack([np.asarray(getter(e)["q"], dtype=float) for e in exps])
    return p, q.mean(axis=0)


def kappa_at(k, p, q):
    pnom = norm.cdf(k) - norm.cdf(-k)
    qnom = float(np.interp(pnom, p, q))
    return qnom / k, pnom, qnom


def coverage_at(profile, threshold):
    p = np.asarray(profile["p"], dtype=float)
    q = np.asarray(profile["q"], dtype=float)
    return float(np.interp(threshold, q, p))


def analyse(name, cal, val, getter):
    p, q = pooled_profile(cal, getter)
    kap = {}
    for k in (1.0, 1.2816, 1.6449, 1.96):
        kap[k] = kappa_at(k, p, q)[0]
    # 未修正 / 修正后的独立验证覆盖率
    k1, k90 = 1.0, 1.6449
    unc1 = np.mean([coverage_at(getter(e), k1) for e in val])
    unc90 = np.mean([coverage_at(getter(e), k90) for e in val])
    cor1 = np.mean([coverage_at(getter(e), kap[k1] * k1) for e in val])
    cor90 = np.mean([coverage_at(getter(e), kap[k90] * k90) for e in val])
    return dict(name=name, kappa_1sigma=kap[1.0], kappa_80=kap[1.2816],
                kappa_90=kap[1.6449], kappa_95=kap[1.96],
                uncorrected_1sigma=float(unc1), corrected_1sigma=float(cor1),
                uncorrected_90=float(unc90), corrected_90=float(cor90),
                p=p, q=q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(
        os.getcwd(), "outputs", "multi_closure_z"))
    ap.add_argument("--cal", default=",".join(str(i) for i in range(10)))
    ap.add_argument("--val", default=",".join(str(i) for i in range(10, 20)))
    args = ap.parse_args()
    cal_ids = [int(x) for x in args.cal.split(",")]
    val_ids = [int(x) for x in args.val.split(",")]
    exps = load_exps(args.dir)
    cal = [exps[i] for i in cal_ids if i in exps]
    val = [exps[i] for i in val_ids if i in exps]
    if not cal or not val:
        raise SystemExit(f"缺少外部实验：cal={len(cal)} val={len(val)}")

    getters = {
        "Q0 PDF": lambda e: e["q0"]["z_profile"],
        "data points": lambda e: e["data_all_z_profile"],
    }
    for n in ("F2p", "F2d", "xF3vp", "xF3vbp"):
        getters[n] = (lambda nn: (lambda e: e["data"][nn]["z_profile"]))(n)

    results = {}
    for name, getter in getters.items():
        results[name] = analyse(name, cal, val, getter)

    out = dict(cal=cal_ids, val=val_ids,
               results={k: {kk: vv for kk, vv in v.items()
                            if kk not in ("p", "q")}
                        for k, v in results.items()})
    with open(os.path.join(args.dir, "calibration_validation.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"== 校准/验证（cal={cal_ids}, val={val_ids}）==")
    print(f"{'对象':10s} {'k1':>5s} {'k90':>5s} {'1σ未修':>7s} "
          f"{'1σ修正':>7s} {'90%未修':>8s} {'90%修正':>8s}")
    for name, r in results.items():
        print(f"{name:10s} {r['kappa_1sigma']:5.2f} {r['kappa_90']:5.2f} "
              f"{100*r['uncorrected_1sigma']:6.1f}% "
              f"{100*r['corrected_1sigma']:6.1f}% "
              f"{100*r['uncorrected_90']:7.1f}% "
              f"{100*r['corrected_90']:7.1f}%")

    # 图
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    for name, color in [("Q0 PDF", "#1f77b4"), ("data points", "#d62728")]:
        r = results[name]
        ax.plot(r["q"], r["p"], color=color, lw=1.6, label=name)
    zz = np.linspace(0, 4, 200)
    ax.plot(zz, 2 * norm.cdf(zz) - 1, "k--", lw=1.0, label="Gaussian")
    ax.axhline(0.6827, color="gray", ls=":", lw=0.8)
    ax.axhline(0.90, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$|z| = |{\rm fit}-{\rm truth}|/\sigma_{\rm ens}$")
    ax.set_ylabel("empirical CDF")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("(a) calibration set: empirical CDF", fontsize=10)

    ax = axes[1]
    labels = ["1σ", "90%"]
    x = np.arange(2)
    w = 0.36
    r = results["data points"]
    ax.bar(x - w / 2, [r["uncorrected_1sigma"], r["uncorrected_90"]], w,
           color="#d62728", alpha=0.6, label="uncorrected")
    ax.bar(x + w / 2, [r["corrected_1sigma"], r["corrected_90"]], w,
           color="#1f77b4", alpha=0.8, label="corrected")
    ax.axhline(0.6827, color="gray", ls=":", lw=0.8)
    ax.axhline(0.90, color="gray", ls=":", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("validation coverage (data points)")
    ax.set_title(r"(b) independent validation, $\kappa_1=%.2f,\ \kappa_{90}=%.2f$"
                 % (r["kappa_1sigma"], r["kappa_90"]), fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(args.dir, "calibration_validation.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"已保存：{path} 与 calibration_validation.json")


if __name__ == "__main__":
    main()
