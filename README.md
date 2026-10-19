# neural-pdf-toolkit

Differentiable neural parton-distribution-function (PDF) fitting with

- a neural parameterization of $(V_u,V_d,S,g)$ at the input scale,
- exact matrix-exponential LO DGLAP evolution,
- analytic profiling of correlated systematic uncertainties,
- replica ensembles and multi-pseudo-experiment closure tests,
- a $k$-dependent closure calibration factor $\kappa(k)$,
- a HERA neutral-current data loader with a full covariance matrix.

The package is the code accompanying the paper
*Calibrating neural parton-distribution uncertainties with
multi-pseudo-experiment closure tests*.

## Installation

```bash
git clone https://github.com/YuhaoPeng1103/HERA.git
cd HERA
python -m pip install -e .
# optional: python -m pip install -e ".[test]"
```

Dependencies: `numpy`, `scipy`, `torch` (CPU is sufficient), `matplotlib`.

## Quick start

```bash
# physics self-checks: DGLAP moments, sum rules, profile chi2, replicas
neural-pdf-selfcheck

# multi-pseudo-experiment closure (20 experiments x 16 replicas, CPU)
neural-pdf-closure --exp-start 0 --exp-end 20 --n-rep 16 --threads 8
# summary figure and JSON
neural-pdf-closure-summary

# k-dependent calibration on exp 0-9, validation on exp 10-19
neural-pdf-calibrate

# HERA stress test
neural-pdf-hera-load --q2min 2 --q2max 10 --xmin 1e-3
neural-pdf-hera-fit --npz outputs/hera_selected_restricted.npz \
    --q2min 2 --q2max 10 --xmin 1e-3 --n-rep 8
```

All commands write to `./outputs/`.

## Package layout

```
neural_pdf/
├── dglap.py                 # grid, LO splitting matrices, exact evolution
├── model.py                 # neural PDF (PyTorch)
├── truth.py                 # toy truth PDFs for closure tests
├── pseudodata.py            # F2 pseudo-data
├── multidata.py             # four-observable pseudo-data + systematics
├── train_m05.py             # profile chi2 and replica training
├── multi_closure.py         # multi-pseudo-experiment driver
├── multi_closure_summary.py # coverage distributions
├── calibration_check.py     # single-replica vs ensemble-mean coverage
├── calibration_validate.py  # kappa(k) calibration and independent validation
├── selfcheck.py             # numerical self-checks
├── plot_m05.py              # PDF/observable/pull figures
├── tmd.py                   # toy Gaussian TMD + Collins–Soper broadening
└── hera/
    ├── dglap_nf3.py         # LO n_f=3 evolution
    ├── hera_loader.py       # HEPData parser + full covariance
    ├── train_hera.py        # full-covariance fit
    └── plot_hera.py         # stress-test figure
```

## TMD module

`neural_pdf.tmd` provides a minimal, tested Gaussian TMD with a
Collins–Soper broadening kernel: the input-scale Gaussian
$\exp(-k_T^2/\langle k_T^2\rangle)/(\pi\langle k_T^2\rangle)$ evolves to
$\langle k_T^2\rangle(Q)=\langle k_T^2\rangle(Q_0)+2g_2\ln(Q/Q_0)$.
It is a scaffold for the TMD/EIC program, not a full TMD factorization
implementation.

## Validation summary

The key closure result (20 pseudo-experiments, 16 replicas each):

| Quantity | single replica | ensemble mean | nominal |
|---|---|---|---|
| Q0 PDF, $1\sigma$ | 0.637 ± 0.053 | 0.843 ± 0.099 | 0.68 |
| data points, $1\sigma$ | 0.570 ± 0.066 | 0.729 ± 0.148 | 0.68 |
| Q0 PDF, $90\%$ | 0.814 ± 0.035 | 0.981 ± 0.030 | 0.90 |
| data points, $90\%$ | 0.777 ± 0.053 | — | 0.90 |

Calibration on 10 pseudo-experiments and independent validation on 10:

| Object | $\kappa_{1\sigma}$ | $\kappa_{90\%}$ | corrected $1\sigma$ | corrected $90\%$ |
|---|---|---|---|---|
| Q0 PDF | 1.12 | 1.28 | 0.672 | 0.892 |
| data points | 1.32 | 1.36 | 0.684 | 0.899 |

The HERA stress test (`Q^2 in [2,10] GeV^2`, `x >= 1e-3`, 29 points,
8 replicas) gives $\chi^2/N=2.06$ with a pull width of 1.43. This is a
pipeline demonstration, not a competitive PDF extraction; see the paper for
the limitations.

## Data

`neural_pdf/data/hera_nc_eplus_318.csv` is Table 1 of the HERA combined
inclusive DIS measurement (H1 and ZEUS, Eur. Phys. J. C 75 (2015) 580,
arXiv:1506.06042), obtained from HEPData record
[`ins1377206`](https://www.hepdata.net/record/ins1377206). Please cite the
original publication and HEPData when using it.

## Limitations

The default model is leading order, with $n_f=2$ (closure) or $n_f=3$
(HERA stress test), a symmetric sea, and no heavy quarks. It is intended for
methodological studies and closure calibration, not for precision PDF
extraction.

## Citation

If you use this code, please cite the accompanying paper and the repository
(see `CITATION.cff`).

## License

MIT (see `LICENSE`).
