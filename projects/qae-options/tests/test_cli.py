import json

import pytest

from qae_options.cli import main


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "qae-options" in capsys.readouterr().out


def test_config_for_another_experiment_is_rejected(tmp_path, capsys):
    config = tmp_path / "config.toml"
    config.write_text('experiment = "e2"\nseed = 1\n')
    assert main(["e3", "--config", str(config)]) == 2
    assert "not 'e3'" in capsys.readouterr().err


MODEL = """
[model]
distribution = "bimodal"
w = 0.4
center1 = 85.0
sigma1 = 0.10
center2 = 115.0
sigma2 = 0.10
r = 0.03
maturity = 0.5

[payoff]
type = "digital"
strike = 100.0
payout = 1.0
"""


@pytest.mark.slow
def test_all_runners_on_tiny_problem(tmp_path):
    out = str(tmp_path / "out")

    def run(exp, text):
        path = tmp_path / f"{exp}.toml"
        path.write_text(text)
        assert main([exp, "--config", str(path), "--results-dir", out]) == 0, exp

    run(
        "e2",
        "n = 3\nshots = 50\nrepetitions = 4\nchunk = 2\nseed = 1\nepsilon = 1e-2\nworkers = 2\n"
        "[sweeps]\nqae = [2, 3, 4]\niqae = [0.1, 0.03, 0.01]\nmlae = [0, 1, 2, 3]\n"
        "mc = [100, 1000, 10000]\nqmc = [128, 512, 2048]\n" + MODEL,
    )
    run("e3", 'source = "e2"\nn = 3\n' + MODEL)
    run(
        "e4",
        "n = 3\nshots = 50\nrepetitions = 2\nseed = 1\nnoise_scales = [0.3]\n"
        "contrast_powers = [0, 1, 2]\ncontrast_shots = 500\n[sweeps]\nmlae = [0, 1]\n" + MODEL,
    )
    run(
        "e5",
        'n_values = [3, 4]\nstrike = 100.0\n[models.lognormal]\ndistribution = "lognormal"\n'
        "s0 = 100.0\nr = 0.04\nsigma = 0.2\nmaturity = 1.0\n",
    )
    run(
        "e1",
        'n_values = [3]\nestimators = [["iqae", 1e-2]]\nshots = 50\nrepetitions = 2\nseed = 1\n'
        'payoffs = [{ type = "digital", strike = 100.0 }]\n'
        + MODEL.split("[payoff]")[0].replace(
            'distribution = "bimodal"', 'distribution = "lognormal"\ns0 = 100.0\nsigma = 0.2'
        ),
    )
    claim = json.loads((tmp_path / "out" / "summary" / "e3_claim.json").read_text())
    assert claim["Z_extra_qubits_qae"] >= 1
    figures = {p.name for p in (tmp_path / "out" / "figures").glob("*.png")}
    assert {"e2.png", "e4.png", "e4_contrast.png", "e5.png"} <= figures
