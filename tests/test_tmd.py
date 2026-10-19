import numpy as np

from neural_pdf.tmd import (gaussian_tmd, fourier_gaussian, cs_kernel,
                            width_at_scale, tmd_at_scale)


def test_normalization():
    sigma2 = 0.25
    kT = np.linspace(0.0, 10.0, 200000)
    f = gaussian_tmd(kT ** 2, sigma2)
    integral = np.trapezoid(2 * np.pi * kT * f, kT)
    assert abs(integral - 1.0) < 1e-4


def test_fourier_at_zero():
    assert abs(float(fourier_gaussian(0.0, 0.25)) - 1.0) < 1e-12


def test_cs_kernel_at_q0():
    assert abs(float(cs_kernel(4.0, 10.0, 10.0)) - 1.0) < 1e-12


def test_width_broadening():
    s0 = 0.25
    assert width_at_scale(s0, 10.0, 2.0) > s0
    assert width_at_scale(s0, 100.0, 2.0) > width_at_scale(s0, 10.0, 2.0)


def test_tmd_evolves():
    s0 = 0.25
    kT2 = 0.5
    low = float(tmd_at_scale(kT2, s0, 2.0, 2.0))
    high = float(tmd_at_scale(kT2, s0, 100.0, 2.0))
    assert high < low
