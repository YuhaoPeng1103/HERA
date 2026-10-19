# -*- coding: utf-8 -*-
"""
Toy Gaussian TMD + Collins–Soper evolution
=========================================
TMD at the input scale:

    f(x,kT^2;Q0) = f_collinear(x;Q0) * exp(-kT^2/<kT^2>)/(pi <kT^2>)

b_T-space (2D Fourier transform of a Gaussian):

    F~(bT^2) = exp(-<kT^2> bT^2 / 4)

Non-perturbative CS evolution:

    F~(bT^2;Q) = F~(bT^2;Q0) exp(-g2 bT^2 ln(Q/Q0) / 2)

which corresponds to

    <kT^2>(Q) = <kT^2>(Q0) + 2 g2 ln(Q/Q0).

This is a toy model for closure/projection studies, not a full TMD
factorization implementation.
"""
import numpy as np


def gaussian_tmd(kT2, sigma2):
    """Normalized 2D Gaussian: int d^2 kT f = 1."""
    kT2 = np.asarray(kT2, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    return np.exp(-kT2 / sigma2) / (np.pi * sigma2)


def fourier_gaussian(bT2, sigma2):
    """2D Fourier transform of the normalized Gaussian."""
    bT2 = np.asarray(bT2, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    return np.exp(-sigma2 * bT2 / 4.0)


def cs_kernel(bT2, Q, Q0, g2=0.5):
    """Non-perturbative Collins–Soper evolution kernel."""
    bT2 = np.asarray(bT2, dtype=float)
    return np.exp(-0.5 * g2 * bT2 * np.log(Q / Q0))


def width_at_scale(sigma0_2, Q, Q0, g2=0.5):
    """Evolved Gaussian width: sigma^2(Q)=sigma0^2+2 g2 ln(Q/Q0)."""
    return sigma0_2 + 2.0 * g2 * np.log(Q / Q0)


def tmd_at_scale(kT2, sigma0_2, Q, Q0, g2=0.5):
    """Gaussian TMD after CS evolution (normalized)."""
    return gaussian_tmd(kT2, width_at_scale(sigma0_2, Q, Q0, g2))
