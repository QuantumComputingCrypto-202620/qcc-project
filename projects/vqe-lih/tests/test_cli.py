import pytest

from vqe_lih.cli import main


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "vqe-lih" in capsys.readouterr().out


def test_config_for_another_experiment_is_rejected(tmp_path, capsys):
    config = tmp_path / "config.toml"
    config.write_text('experiment = "e2"\nseed = 1\n')
    assert main(["e3", "--config", str(config)]) == 2
    assert "not 'e3'" in capsys.readouterr().err


def test_e1_runs_end_to_end(tmp_path):
    config = tmp_path / "e1_tiny.toml"
    config.write_text(
        """
experiment = "e1"
molecule = "H2"
bond_lengths = [0.7, 1.0]
warm_start = false
seed = 1
[optimizer]
name = "cobyla"
[[ansatze]]
type = "uccsd"
"""
    )
    assert main(["e1", "--config", str(config), "--results-dir", str(tmp_path / "out")]) == 0
    summary = (tmp_path / "out" / "summary" / "e1_tiny.csv").read_text()
    assert summary.count("\n") == 3  # header + 2 bond lengths
    assert (tmp_path / "out" / "figures" / "e1_tiny_full.png").is_file()
    assert (tmp_path / "out" / "raw" / "e1_tiny.json").is_file()
