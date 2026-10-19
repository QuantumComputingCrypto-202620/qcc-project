"""End-to-end smoke test of every experiment runner on a tiny LiH problem (marked slow)."""

import json

import pytest

from vqe_lih.cli import main

SPACE = """
label = "2e3o"
freeze_core = true
num_electrons = 2
num_spatial_orbitals = 3
orbitals = "sigma"
"""


def write(path, text):
    path.write_text(text)
    return str(path)


@pytest.mark.slow
def test_all_runners_on_tiny_problem(tmp_path):
    out = str(tmp_path / "out")
    curve = 'molecule = "LiH"\nequilibrium = 1.6\nbond_lengths = [1.5, 1.6, 2.5, 3.5]\nseed = 1\n'
    e2 = write(
        tmp_path / "e2_uccsd.toml",
        curve + '[optimizer]\nname = "l_bfgs_b"\n'
        '[[ansatze]]\ntype = "uccsd"\n[[active_spaces]]' + SPACE,
    )
    e3 = write(
        tmp_path / "e3_hea.toml",
        curve + '[optimizer]\nname = "l_bfgs_b"\nmaxiter = 20\n'
        '[first_point]\ninit = "random"\nrestarts = 2\n'
        '[[ansatze]]\ntype = "hea"\nreps = 1\n[[ansatze]]\ntype = "hea"\nreps = 2\n'
        "[[active_spaces]]" + SPACE,
    )
    e4 = write(tmp_path / "e4.toml", 'inputs = ["e2_uccsd", "e3_hea"]\n')
    e5 = write(
        tmp_path / "e5.toml",
        curve + 'points = [1.6, 3.5]\noptimizers = ["cobyla", "spsa"]\n'
        "random_starts = 2\nworkers = 2\n[maxiter]\ncobyla = 30\nspsa = 10\n"
        "[active_space]" + SPACE + '[[ansatze]]\ntype = "uccsd"\nsource = "e2_uccsd"\n'
        '[[ansatze]]\ntype = "hea"\nreps = 2\nsource = "e3_hea"\n',
    )
    e6 = write(
        tmp_path / "e6.toml",
        'molecule = "LiH"\npoint = 1.6\nshots = [100, 10000]\n'
        'optimizers = ["spsa"]\nrepetitions = 2\nseed = 3\n[maxiter]\nspsa = 10\n'
        "[active_space]" + SPACE + '[ansatz]\ntype = "uccsd"\n',
    )
    e7 = write(
        tmp_path / "e7.toml",
        'molecule = "LiH"\npoints = [1.6]\nshots = 2000\nseed = 4\n'
        "[active_space]" + SPACE + '[[ansatze]]\ntype = "hea"\nreps = 1\nsource = "e3_hea"\n',
    )
    for exp, cfg in [("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5), ("e6", e6), ("e7", e7)]:
        assert main([exp, "--config", cfg, "--results-dir", out]) == 0, exp
    claim = json.loads((tmp_path / "out" / "summary" / "e4_claim.json").read_text())
    assert claim["2e3o"]["Y"] == 4
    figures = {p.name for p in (tmp_path / "out" / "figures").glob("*.png")}
    assert {
        "e4_error_vs_depth_2e3o.png",
        "e5_uccsd.png",
        "e5_histories.png",
        "e6.png",
        "e7.png",
    } <= figures
