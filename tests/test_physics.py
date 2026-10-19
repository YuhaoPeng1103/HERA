import numpy as np

from neural_pdf import dglap, pseudodata, truth


def test_dglap_moments():
    x, w, wx, mats = pseudodata.build()[:4]
    Mqq, Mqg, Mgq, Mgg = mats
    checks = dglap.check_moments(x, w, wx, Mqq, Mqg, Mgq, Mgg, verbose=False)
    for name, (got, ref) in checks.items():
        assert abs(got - ref) < 5e-3, (name, got, ref)


def test_sum_rules():
    x, w, wx, mats, pdf0 = pseudodata.build()
    sr = truth.check_sum_rules(pdf0, x, wx, verbose=False)
    assert abs(sr["n_u"] - 2.0) < 1e-3
    assert abs(sr["n_d"] - 1.0) < 1e-3
    assert abs(sr["mom"] - 1.0) < 1e-3


def test_pseudodata_size():
    x, w, wx, mats, pdf0 = pseudodata.build()
    data, evolved = pseudodata.make_data(x, w, mats, pdf0)
    assert len(data) == 98
    assert len(evolved) == 7
