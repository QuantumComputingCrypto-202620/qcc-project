import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from qcc_common.plotting import SERIES_COLORS, apply_style, save_figure, series_style
from qcc_common.results import ExperimentRecord, to_jsonable


def test_record_round_trip(tmp_path):
    record = ExperimentRecord(
        experiment="demo",
        config={"seed": 3},
        seed=3,
        payload=[{"energy": np.float64(-1.5), "params": np.zeros(2), "ok": np.bool_(True)}],
    )
    path = record.save(tmp_path / "raw" / "demo.json")
    loaded = ExperimentRecord.load(path)
    assert loaded.payload == [{"energy": -1.5, "params": [0.0, 0.0], "ok": True}]
    assert loaded.seed == 3
    assert "qiskit" in loaded.package_versions
    assert loaded.uv_lock_sha256 is not None and len(loaded.uv_lock_sha256) == 64


def test_complex_values_are_serialized():
    assert to_jsonable(np.complex128(2.0 + 0j)) == 2.0
    assert json.dumps(to_jsonable({"z": 1 + 2j})) == '{"z": {"real": 1.0, "imag": 2.0}}'


def test_save_figure_writes_png_and_pdf(tmp_path):
    apply_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], **series_style(0))
    paths = save_figure(fig, tmp_path / "fig")
    assert [p.suffix for p in paths] == [".png", ".pdf"]
    assert all(p.stat().st_size > 0 for p in paths)


def test_series_colors_do_not_cycle():
    with pytest.raises(ValueError):
        series_style(len(SERIES_COLORS))
