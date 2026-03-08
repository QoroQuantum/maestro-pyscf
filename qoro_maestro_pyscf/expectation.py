# Copyright 2026 Qoro Quantum Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Expectation value engine using Maestro's native QuantumCircuit.estimate().

Wraps circuit evaluation so that the rest of the library can call a single
function without worrying about backend configuration details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from maestro.circuits import QuantumCircuit

from qoro_maestro_pyscf.backends import BackendConfig


def evaluate_expectation(
    circuit: QuantumCircuit,
    pauli_labels: list[str],
    config: BackendConfig,
) -> np.ndarray:
    """
    Evaluate expectation values of Pauli observables on a Maestro circuit.

    All observables are batched into a single ``qc.estimate()`` call, so
    Maestro evaluates them in one statevector (or MPS) pass on the GPU.

    Parameters
    ----------
    circuit : QuantumCircuit
        The prepared (parameterised) circuit.
    pauli_labels : list[str]
        Pauli observable strings, e.g. ``["ZZII", "IXYZ"]``.
    config : BackendConfig
        Maestro backend configuration.

    Returns
    -------
    expectation_values : np.ndarray, shape (len(pauli_labels),)
        Real-valued expectation values ⟨ψ|Pᵢ|ψ⟩.
    """
    if not pauli_labels:
        return np.array([], dtype=float)

    estimate_kwargs = {
        "observables": pauli_labels,
        "simulator_type": config.simulator_type,
        "simulation_type": config.simulation_type,
    }
    if config.mps_bond_dim is not None:
        estimate_kwargs["max_bond_dimension"] = config.mps_bond_dim

    result = circuit.estimate(**estimate_kwargs)

    return np.array(result["expectation_values"], dtype=float)


def compute_energy(
    circuit: QuantumCircuit,
    identity_offset: float,
    pauli_labels: list[str],
    pauli_coeffs: np.ndarray,
    config: BackendConfig,
) -> float:
    """
    Compute the total energy ⟨H⟩ = c₀ + Σᵢ Re(cᵢ)·⟨Pᵢ⟩ for a given circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The prepared circuit.
    identity_offset : float
        Coefficient of the identity term.
    pauli_labels : list[str]
        Non-identity Pauli terms.
    pauli_coeffs : np.ndarray
        Complex coefficients for each Pauli term.
    config : BackendConfig
        Maestro backend configuration.

    Returns
    -------
    energy : float
        The expectation value of the Hamiltonian.
    """
    exp_vals = evaluate_expectation(circuit, pauli_labels, config)
    # Coefficients are real for a Hermitian Hamiltonian (physical observable).
    # We use .real to drop any floating-point imaginary noise from OpenFermion.
    return identity_offset + float(np.dot(pauli_coeffs.real, exp_vals))


def get_state_probabilities(
    circuit: QuantumCircuit,
    config: BackendConfig,
) -> np.ndarray:
    """
    Get the full probability distribution |⟨k|ψ⟩|² for each computational basis state.

    Wraps Maestro's native ``get_probabilities()`` with the configured backend.

    Parameters
    ----------
    circuit : QuantumCircuit
        The prepared circuit.
    config : BackendConfig
        Maestro backend configuration.

    Returns
    -------
    probabilities : np.ndarray, shape (2**n_qubits,)
        Probability of each computational basis state.
    """
    import maestro

    kwargs = {
        "simulator_type": config.simulator_type,
        "simulation_type": config.simulation_type,
    }
    if config.mps_bond_dim is not None:
        kwargs["max_bond_dimension"] = config.mps_bond_dim

    probs = maestro.get_probabilities(circuit, **kwargs)
    return np.array(probs, dtype=float)


def compute_state_fidelity(
    circuit_a: QuantumCircuit,
    circuit_b: QuantumCircuit,
    config: BackendConfig,
) -> float:
    """
    Compute the classical fidelity between two circuit states.

    Uses the Bhattacharyya coefficient: F = (Σ √(pᵢ·qᵢ))², which equals
    the true quantum fidelity |⟨ψ_a|ψ_b⟩|² when both states are pure and
    have non-negative real amplitudes (common for VQE ground states).

    For general states with complex phases, this is a lower bound on the
    true fidelity.

    Parameters
    ----------
    circuit_a, circuit_b : QuantumCircuit
        The two circuits to compare.
    config : BackendConfig
        Maestro backend configuration.

    Returns
    -------
    fidelity : float
        Classical fidelity in [0, 1].  1.0 = identical probability distributions.
    """
    p = get_state_probabilities(circuit_a, config)
    q = get_state_probabilities(circuit_b, config)
    bhatt = float(np.sum(np.sqrt(p * q)))
    return bhatt ** 2
