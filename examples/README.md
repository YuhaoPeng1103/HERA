# Examples

Run from the repository root after `pip install -e .`.

```bash
# 1. numerical self-checks
neural-pdf-selfcheck

# 2. closure campaign (20 pseudo-experiments x 16 replicas)
neural-pdf-closure --exp-start 0 --exp-end 20 --n-rep 16 --threads 8
neural-pdf-closure-summary

# 3. calibration and independent validation
neural-pdf-calibrate

# 4. HERA stress test
neural-pdf-hera-load --q2min 2 --q2max 10 --xmin 1e-3
neural-pdf-hera-fit --npz outputs/hera_selected_restricted.npz \
    --q2min 2 --q2max 10 --xmin 1e-3 --n-rep 8
```

The closure campaign is CPU-heavy (about 20 minutes on 16 cores for the
reference configuration); use `--n-rep 4 --epochs 400` for a quick smoke test.
