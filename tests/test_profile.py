import numpy as np
import torch
from scipy.optimize import minimize

from neural_pdf.train_m05 import profile_chi2


def test_profile_against_scipy():
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(0)
    n, k = 40, 3
    t = rng.uniform(0.1, 1.0, n)
    B = np.stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)], axis=1)
    sig = 0.02 * t
    sb = np.array([0.015, 0.010, 0.008])
    y = t * (1 + B @ rng.normal(0, sb)) + rng.normal(0, sig)
    resid = y - t
    chi_t, lam_t = profile_chi2(torch.tensor(resid), torch.tensor(t),
                                torch.tensor(B), torch.tensor(sig),
                                torch.tensor(sb))

    def f(lam):
        A = t[:, None] * B
        r = resid - A @ lam
        return np.sum((r / sig) ** 2) + np.sum((lam / sb) ** 2)

    res = minimize(f, np.zeros(k), method="BFGS", tol=1e-12)
    assert abs(float(chi_t) - res.fun) < 1e-8
    assert np.allclose(lam_t.numpy(), res.x, atol=1e-5)
