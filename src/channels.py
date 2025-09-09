"""
Kraus operator construction for common quantum noise channels.

References:
    Nielsen & Chuang (2010) - "Quantum Computation and Quantum Information"
    Chapter 8: Quantum noise and quantum operations
"""

import numpy as np


def depolarizing_channel(p):
    """
    Build Kraus operators for the single-qubit depolarizing channel.

    With probability p, the qubit is replaced by the maximally mixed state.
    With probability (1-p), it remains unchanged.

    E(rho) = (1 - p) * rho + (p/3) * (X rho X + Y rho Y + Z rho Z)

    Args:
        p: depolarizing probability, 0 <= p <= 1

    Returns:
        list of 4 Kraus matrices (2x2 numpy arrays)
    """
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1], got {p}")

    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    E0 = np.sqrt(1 - 3 * p / 4) * I
    E1 = np.sqrt(p / 4) * X
    E2 = np.sqrt(p / 4) * Y
    E3 = np.sqrt(p / 4) * Z

    return [E0, E1, E2, E3]


def amplitude_damping(gamma):
    """
    Build Kraus operators for the amplitude damping channel.

    Models T1 relaxation — the excited state |1> decays to |0> with
    probability gamma. This is the dominant noise in superconducting qubits.

    Args:
        gamma: decay probability, 0 <= gamma <= 1

    Returns:
        list of 2 Kraus matrices (2x2 numpy arrays)
    """
    if not 0 <= gamma <= 1:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")

    E0 = np.array([[1, 0],
                    [0, np.sqrt(1 - gamma)]], dtype=complex)
    E1 = np.array([[0, np.sqrt(gamma)],
                    [0, 0]], dtype=complex)

    return [E0, E1]


def phase_damping(lambda_):
    """
    Build Kraus operators for the phase damping channel.

    Models T2 dephasing without energy loss — coherences decay
    while populations remain unchanged.

    Args:
        lambda_: dephasing probability, 0 <= lambda_ <= 1

    Returns:
        list of 2 Kraus matrices (2x2 numpy arrays)
    """
    if not 0 <= lambda_ <= 1:
        raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")

    E0 = np.array([[1, 0],
                    [0, np.sqrt(1 - lambda_)]], dtype=complex)
    E1 = np.array([[0, 0],
                    [0, np.sqrt(lambda_)]], dtype=complex)

    return [E0, E1]


def apply_channel(rho, kraus_ops):
    """
    Apply a quantum channel (given as Kraus operators) to a density matrix.

    E(rho) = sum_k E_k @ rho @ E_k^dagger

    Args:
        rho: input density matrix (numpy array)
        kraus_ops: list of Kraus operator matrices

    Returns:
        output density matrix (numpy array)
    """
    rho = np.array(rho, dtype=complex)
    result = np.zeros_like(rho)

    for E in kraus_ops:
        result += E @ rho @ E.conj().T

    return result


def verify_kraus_completeness(kraus_ops, tol=1e-10):
    """
    Check that the Kraus operators satisfy the completeness relation:
    sum_k E_k^dagger @ E_k = I

    Returns True if the relation holds within tolerance.
    """
    d = kraus_ops[0].shape[0]
    total = np.zeros((d, d), dtype=complex)

    for E in kraus_ops:
        total += E.conj().T @ E

    return np.allclose(total, np.eye(d), atol=tol)


def compose_channels(kraus_ops_1, kraus_ops_2):
    """
    Compose two channels: first apply channel 1, then channel 2.

    The resulting Kraus operators are all products E2_j @ E1_i.

    Args:
        kraus_ops_1: Kraus operators for the first channel
        kraus_ops_2: Kraus operators for the second channel

    Returns:
        list of composed Kraus matrices
    """
    composed = []
    for E2 in kraus_ops_2:
        for E1 in kraus_ops_1:
            composed.append(E2 @ E1)

    return composed


def channel_to_choi(kraus_ops):
    """
    Convert a channel (Kraus representation) to its Choi matrix.

    The Choi matrix is defined as:
    J = sum_k |E_k>> <<E_k|
    where |E_k>> is the vectorization of E_k.

    For a d-dimensional channel, the Choi matrix is d^2 x d^2.

    Args:
        kraus_ops: list of Kraus operator matrices

    Returns:
        Choi matrix (numpy array)
    """
    d = kraus_ops[0].shape[0]
    choi = np.zeros((d * d, d * d), dtype=complex)

    # build Choi via: J = (I x E)(|Omega><Omega|)
    # equivalently: J_ij,kl = sum_m E_m[i,k] * conj(E_m[j,l])
    for E in kraus_ops:
        vec = E.flatten(order='F')  # column-major vectorization
        choi += np.outer(vec, vec.conj())

    return choi
# TODO: add thermal relaxation channel
