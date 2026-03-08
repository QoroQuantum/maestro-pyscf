# This file is part of qoro-maestro-pyscf.
#
# Copyright (C) 2026 Qoro Quantum Ltd.
#
# qoro-maestro-pyscf is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# qoro-maestro-pyscf is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with qoro-maestro-pyscf. If not, see <https://www.gnu.org/licenses/\>.

"""
Hamiltonian construction: PySCF integrals → OpenFermion QubitOperator.

Converts one- and two-electron integrals (as delivered by PySCF's CASCI/CASSCF
``kernel`` call) into a qubit Hamiltonian via the Jordan-Wigner transformation.
The resulting Pauli terms are formatted for Maestro's ``estimate()`` API.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from pyscf import ao2mo

from openfermion import InteractionOperator, QubitOperator, jordan_wigner


# Type aliases matching PySCF's calling convention
OneBodyIntegrals = Union[np.ndarray, tuple[np.ndarray, np.ndarray]]
TwoBodyIntegrals = Union[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]


def _restore_eri(eri: np.ndarray, norb: int) -> np.ndarray:
    """
    Restore two-electron integrals to a full 4D tensor.

    PySCF's CASCI/CASSCF kernel passes ERIs in compressed formats:
      - 1D array of length norb*(norb+1)/2 * (norb*(norb+1)/2+1)/2 (8-fold)
      - 2D array of shape (norb*(norb+1)//2, norb*(norb+1)//2) (4-fold)
      - Already 4D (norb, norb, norb, norb)

    We always need the full (norb, norb, norb, norb) tensor.
    """
    if eri.ndim == 4:
        return eri
    # ao2mo.restore(1, ...) restores any compressed format to full 4D
    return ao2mo.restore(1, eri, norb)


def integrals_to_qubit_hamiltonian(
    h1: OneBodyIntegrals,
    h2: TwoBodyIntegrals,
    norb: int,
) -> tuple[QubitOperator, float]:
    """
    Convert PySCF one- and two-electron integrals to a qubit Hamiltonian.

    Supports both RHF (single arrays) and UHF (tuple of spin-resolved arrays)
    integral formats, matching the calling convention of PySCF's
    ``fcisolver.kernel``.

    Parameters
    ----------
    h1 : ndarray or (ndarray, ndarray)
        One-electron integrals. Single array for RHF; tuple (h1_a, h1_b)
        for UHF.
    h2 : ndarray or (ndarray, ndarray, ndarray)
        Two-electron integrals in chemist's notation (possibly compressed).
        Single array for RHF; tuple (h2_aa, h2_ab, h2_bb) for UHF.
    norb : int
        Number of spatial orbitals.

    Returns
    -------
    qubit_op : QubitOperator
        Qubit Hamiltonian (Jordan-Wigner mapped).
    identity_offset : float
        Coefficient of the identity term (energy offset).
    """
    n_qubits = 2 * norb

    # --- Unpack integrals ---
    if isinstance(h1, tuple):
        h1_a, h1_b = h1
    else:
        h1_a = h1
        h1_b = h1  # RHF: alpha = beta

    if isinstance(h2, tuple):
        h2_aa = _restore_eri(h2[0], norb)
        h2_ab = _restore_eri(h2[1], norb)
        h2_bb = _restore_eri(h2[2], norb)
    else:
        h2_full = _restore_eri(h2, norb)
        h2_aa = h2_full
        h2_ab = h2_full
        h2_bb = h2_full  # RHF: all spin blocks are the same

    # --- Build spin-orbital one-body tensor ---
    one_body = np.zeros((n_qubits, n_qubits))
    for p in range(norb):
        for q in range(norb):
            one_body[2 * p, 2 * q] = h1_a[p, q]          # α-α
            one_body[2 * p + 1, 2 * q + 1] = h1_b[p, q]  # β-β

    # --- Build spin-orbital two-body tensor ---
    # PySCF delivers chemist notation (pq|rs).
    # InteractionOperator expects physicist notation with the convention:
    #   H = Σ_{pq} h1[p,q] a†_p a_q
    #     + 0.5 Σ_{pqrs} h2[p,q,r,s] a†_p a†_r a_s a_q
    #
    # From chemist (pq|rs) we construct the spin-orbital two-body tensor.
    two_body = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))

    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    # αα block
                    two_body[2*p, 2*q, 2*r, 2*s] = h2_aa[p, q, r, s]
                    # ββ block
                    two_body[2*p+1, 2*q+1, 2*r+1, 2*s+1] = h2_bb[p, q, r, s]
                    # αβ block
                    two_body[2*p, 2*q+1, 2*r+1, 2*s] = h2_ab[p, q, r, s]
                    # βα block
                    two_body[2*p+1, 2*q, 2*r, 2*s+1] = h2_ab[q, p, s, r]

    # --- Build InteractionOperator and apply Jordan-Wigner ---
    iop = InteractionOperator(
        constant=0.0,
        one_body_tensor=one_body,
        two_body_tensor=0.5 * two_body,
    )
    qubit_op = jordan_wigner(iop)

    # --- Extract identity offset ---
    identity_offset = 0.0
    if () in qubit_op.terms:
        identity_offset = qubit_op.terms[()].real

    return qubit_op, identity_offset


def qubit_op_to_pauli_list(
    qubit_op: QubitOperator,
    n_qubits: int,
) -> tuple[float, list[str], np.ndarray]:
    """
    Convert an OpenFermion QubitOperator into Maestro-compatible observables.

    Parameters
    ----------
    qubit_op : QubitOperator
        The qubit Hamiltonian.
    n_qubits : int
        Number of qubits (spin-orbitals).

    Returns
    -------
    identity_coeff : float
        Coefficient of the all-identity term (classical energy offset).
    pauli_labels : list[str]
        Pauli strings for Maestro, e.g. ``["XZIY", "ZZII"]``.
    pauli_coeffs : np.ndarray
        Complex coefficients for each Pauli term.
    """
    identity_coeff = 0.0
    pauli_labels = []
    pauli_coeffs = []

    for term, coeff in qubit_op.terms.items():
        if not term:
            identity_coeff = coeff.real
            continue

        label = ["I"] * n_qubits
        for qubit_idx, pauli_char in term:
            label[qubit_idx] = pauli_char

        pauli_labels.append("".join(label))
        pauli_coeffs.append(coeff)

    return identity_coeff, pauli_labels, np.array(pauli_coeffs, dtype=complex)
