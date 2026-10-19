import numpy as np

from neural_pdf.hera.hera_loader import load


def test_hera_loader():
    d = load(q2min=2.0, q2max=10.0, xmin=1e-3)
    assert len(d["y"]) > 10
    assert d["C"].shape == (len(d["y"]), len(d["y"]))
    assert np.all(np.diag(d["C"]) > 0)
    assert np.all(np.linalg.eigvalsh(d["C"]) > 0)
