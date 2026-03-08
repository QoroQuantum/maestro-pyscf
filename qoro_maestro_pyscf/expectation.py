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

import numpy as np

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
