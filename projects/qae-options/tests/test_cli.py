import pytest

from qae_options.cli import KNOWN_EXPERIMENTS, main


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "qae-options" in capsys.readouterr().out


def test_unimplemented_experiment_reports_error(tmp_path, capsys):
    config = tmp_path / "config.toml"
    config.write_text("seed = 1\n")
    assert main([KNOWN_EXPERIMENTS[0], "--config", str(config)]) == 2
    assert "not implemented" in capsys.readouterr().err
