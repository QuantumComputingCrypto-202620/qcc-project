import pytest
from qiskit import QuantumCircuit, transpile

from qcc_common import backends
from qcc_common.backends import (
    noisy_simulator,
    read_ibm_token,
    shot_simulator,
    statevector_simulator,
)


def bell() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


def run_counts(backend, circuit, shots=None) -> dict[str, int]:
    tqc = transpile(circuit, backend, seed_transpiler=7)
    job = backend.run(tqc) if shots is None else backend.run(tqc, shots=shots)
    return job.result().get_counts()


def odd_parity_fraction(counts: dict[str, int]) -> float:
    return (counts.get("01", 0) + counts.get("10", 0)) / sum(counts.values())


def test_bell_on_statevector_simulator():
    counts = run_counts(statevector_simulator(seed=1), bell(), shots=2000)
    assert set(counts) == {"00", "11"}
    assert abs(counts["00"] / 2000 - 0.5) < 0.05


def test_shot_simulator_is_seeded():
    first = run_counts(shot_simulator(shots=500, seed=42), bell())
    second = run_counts(shot_simulator(shots=500, seed=42), bell())
    assert first == second
    assert sum(first.values()) == 500


def test_shot_simulator_rejects_non_positive_shots():
    with pytest.raises(ValueError):
        shot_simulator(shots=0, seed=1)


def test_noise_grows_with_scale():
    errors = {
        scale: odd_parity_fraction(run_counts(noisy_simulator(scale=scale, seed=3), bell(), 4000))
        for scale in (0.0, 1.0, 20.0)
    }
    assert errors[0.0] == 0.0
    assert 0.0 < errors[1.0] < errors[20.0]


def test_noisy_simulator_validates_arguments():
    with pytest.raises(ValueError, match="non-negative"):
        noisy_simulator(scale=-1.0)
    with pytest.raises(ValueError, match="unknown fake backend"):
        noisy_simulator(fake_backend="FakeDoesNotExist")


def test_ibm_token_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(backends.TOKEN_ENV_VAR, "from-env")
    assert read_ibm_token(tmp_path / ".env") == "from-env"


def test_ibm_token_from_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv(backends.TOKEN_ENV_VAR, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text('OTHER=1\nQISKIT_IBM_TOKEN="from-file"\n')
    assert read_ibm_token(env_file) == "from-file"


def test_missing_ibm_token_raises(monkeypatch, tmp_path):
    monkeypatch.delenv(backends.TOKEN_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match="QISKIT_IBM_TOKEN"):
        read_ibm_token(tmp_path / ".env")


def test_noise_scale_applies_to_gates_and_relaxation():
    """Gate/readout errors scale by ``scale`` and T1/T2 by 1/scale (regression: T1/T2 edits
    in place were silently lost, leaving full-strength relaxation at every scale)."""
    from qcc_common.backends import get_fake_backend, scaled_noise_model

    totals, t1s = {}, {}
    for scale in (0.1, 1.0):
        backend = get_fake_backend()
        qargs = next(iter(backend.target["ecr"]))
        model = scaled_noise_model(backend, scale)
        totals[scale] = 1 - model._local_quantum_errors["ecr"][qargs].probabilities[0]
        t1s[scale] = backend.target.qubit_properties[0].t1
    assert t1s[0.1] == pytest.approx(10 * t1s[1.0])
    assert totals[0.1] == pytest.approx(0.1 * totals[1.0], rel=0.2)
