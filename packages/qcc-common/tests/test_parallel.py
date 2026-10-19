from qcc_common.parallel import job_key, parallel_map
from qcc_common.results import to_jsonable


def square(job):
    return {"x": job["x"], "y": job["x"] ** 2}


def test_parallel_map_keeps_order_and_checkpoints(tmp_path):
    jobs = [{"x": i} for i in range(5)]
    # Worker processes need an importable function (test modules are not importable there).
    assert parallel_map(to_jsonable, jobs, workers=2) == jobs
    first = parallel_map(square, jobs, workers=1, checkpoint_dir=tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 5
    (tmp_path / f"{job_key({'x': 3})}.json").write_text('{"x": 3, "y": -1}')
    again = parallel_map(square, jobs, workers=1, checkpoint_dir=tmp_path)
    assert [r["y"] for r in first] == [0, 1, 4, 9, 16]
    assert again[3]["y"] == -1  # loaded from the checkpoint, not recomputed
