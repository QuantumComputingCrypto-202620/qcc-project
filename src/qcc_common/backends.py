"""Simulator backends shared by all projects, plus optional IBM Quantum hardware access.

All simulators are Aer ``AerSimulator`` instances, so they work with ``transpile`` and with the
Aer V2 primitives (``qiskit_aer.primitives.SamplerV2`` / ``EstimatorV2``).
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

from qiskit.providers import QubitProperties
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

if TYPE_CHECKING:
    from qiskit.providers import BackendV2
    from qiskit_ibm_runtime import Session

TOKEN_ENV_VAR = "QISKIT_IBM_TOKEN"


def statevector_simulator(seed: int | None = None) -> AerSimulator:
    """Noiseless statevector simulator.

    Use it for exact expectation values (with ``save_statevector`` or an exact Estimator).
    Measured circuits run on it are still sampled, seeded by ``seed``.
    """
    return AerSimulator(method="statevector", seed_simulator=seed)


def shot_simulator(shots: int, seed: int) -> AerSimulator:
    """Noiseless shot-based simulator with a fixed default shot count and seed."""
    if shots < 1:
        raise ValueError(f"shots must be positive, got {shots}")
    return AerSimulator(shots=shots, seed_simulator=seed)


def get_fake_backend(name: str = "FakeSherbrooke") -> BackendV2:
    """Instantiate a fake IBM backend by class name, e.g. ``"FakeSherbrooke"``."""
    fake_provider = importlib.import_module("qiskit_ibm_runtime.fake_provider")
    try:
        backend_cls = getattr(fake_provider, name)
    except AttributeError:
        raise ValueError(f"unknown fake backend {name!r}") from None
    return backend_cls()


def scaled_noise_model(backend: BackendV2, scale: float) -> NoiseModel:
    """Noise model from ``backend`` with every error rate multiplied by ``scale``.

    Gate and readout error probabilities are multiplied by ``scale`` (capped at 1), and T1/T2
    are divided by ``scale``, so relaxation errors also grow roughly linearly with ``scale``.
    ``scale=1`` reproduces ``NoiseModel.from_backend(backend)``; ``scale=0`` gives no noise.
    The backend's target is modified in place, so pass a fresh instance.
    """
    if scale < 0:
        raise ValueError(f"noise scale must be non-negative, got {scale}")
    if scale == 0:
        return NoiseModel(basis_gates=list(backend.operation_names))
    if scale != 1:
        target = backend.target
        for op_name in target.operation_names:
            for props in target[op_name].values():
                if props is not None and props.error is not None:
                    props.error = min(props.error * scale, 1.0)
        if target.qubit_properties:
            # Assign a new list: editing the QubitProperties objects in place does not persist.
            target.qubit_properties = [
                QubitProperties(
                    t1=None if qp.t1 is None else qp.t1 / scale,
                    t2=None if qp.t2 is None else qp.t2 / scale,
                    frequency=qp.frequency,
                )
                for qp in target.qubit_properties
            ]
    return NoiseModel.from_backend(backend)


def noisy_simulator(
    fake_backend: str = "FakeSherbrooke", scale: float = 1.0, seed: int | None = None
) -> AerSimulator:
    """Shot-based simulator with the coupling map, basis gates and (scaled) noise of a fake
    IBM backend.

    ``scale`` multiplies every error rate (see :func:`scaled_noise_model`).
    """
    backend = get_fake_backend(fake_backend)
    noise_model = scaled_noise_model(backend, scale)
    return AerSimulator.from_backend(backend, noise_model=noise_model, seed_simulator=seed)


def read_ibm_token(env_file: Path | str = ".env") -> str:
    """Return the IBM Quantum token from ``$QISKIT_IBM_TOKEN`` or a ``QISKIT_IBM_TOKEN=...``
    line in ``env_file``. Raises ``RuntimeError`` if neither is set."""
    token = os.environ.get(TOKEN_ENV_VAR)
    if token:
        return token
    env_path = Path(env_file)
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key.strip() == TOKEN_ENV_VAR and value.strip():
                return value.strip().strip("'\"")
    raise RuntimeError(
        f"No IBM Quantum token: set ${TOKEN_ENV_VAR} or add it to {env_path} (gitignored)."
    )


def ibm_runtime_session(backend_name: str | None = None) -> Session:
    """Open a Qiskit Runtime session on ``backend_name``, or on the least busy real backend.

    The token is read with :func:`read_ibm_token`; it is never stored anywhere else.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, Session

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=read_ibm_token())
    if backend_name is None:
        backend = service.least_busy(operational=True, simulator=False)
    else:
        backend = service.backend(backend_name)
    return Session(backend=backend)
