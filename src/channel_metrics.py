"""
Metrics for comparing quantum channels — process fidelity, average gate fidelity,
and diamond distance estimation.

References:
    Gilchrist et al. (2005) - "Distance measures to compare real and ideal quantum processes"
    https://arxiv.org/abs/quant-ph/0408063
"""

import numpy as np
from scipy.linalg import sqrtm

from .channels import channel_to_choi, apply_channel


def process_fidelity(channel_kraus, ideal_unitary):
    """
    Compute the process fidelity between a noisy channel (Kraus operators)
    and an ideal unitary operation.

    F_pro = (1/d) * sum_k |Tr(U^dagger @ E_k)|^2

    where d is the dimension and U is the ideal unitary.

    Args:
        channel_kraus: list of Kraus operator matrices for the noisy channel
        ideal_unitary: the ideal unitary matrix (numpy array)

    Returns:
        process fidelity (float between 0 and 1)
    """
    d = ideal_unitary.shape[0]
    U_dag = ideal_unitary.conj().T

    fidelity = 0.0
    for E in channel_kraus:
        overlap = np.trace(U_dag @ E)
        fidelity += np.abs(overlap) ** 2

    return float(np.real(fidelity / d))


def average_gate_fidelity(channel_kraus, n_qubits):
    """
    Compute the average gate fidelity of a channel relative to the identity.

    F_avg = (d * F_pro + 1) / (d + 1)

    where d = 2^n_qubits and F_pro is the process fidelity with identity.

    This gives the average state fidelity over all input pure states,
    which is the standard measure for benchmarking quantum gates.

    Args:
        channel_kraus: list of Kraus operator matrices
        n_qubits: number of qubits the channel acts on

    Returns:
        average gate fidelity (float between 0 and 1)
    """
    d = 2 ** n_qubits
    ideal = np.eye(d, dtype=complex)
    f_pro = process_fidelity(channel_kraus, ideal)
    return float((d * f_pro + 1) / (d + 1))


def diamond_distance_estimate(channel1_kraus, channel2_kraus):
    """
    Estimate the diamond distance between two channels using the Choi
    matrix approach.

    The diamond distance is the maximum trace distance between outputs
    over all possible inputs (including entangled ones). Computing it
    exactly requires an SDP, so we use the Choi matrix bound:

    d_diamond <= d * ||J1 - J2||_1

    where d is the system dimension and ||.||_1 is the trace norm.

    This is an upper bound, but it's tight enough for practical comparison.

    Args:
        channel1_kraus: Kraus operators for channel 1
        channel2_kraus: Kraus operators for channel 2

    Returns:
        estimated diamond distance (float >= 0)
    """
    choi_1 = channel_to_choi(channel1_kraus)
    choi_2 = channel_to_choi(channel2_kraus)

    diff = choi_1 - choi_2

    # trace norm = sum of singular values
    singular_values = np.linalg.svd(diff, compute_uv=False)
    trace_norm = np.sum(singular_values)

    d = channel1_kraus[0].shape[0]
    return float(d * trace_norm)


def state_fidelity(rho, sigma):
    """
    Compute the fidelity between two density matrices.

    F(rho, sigma) = (Tr(sqrt(sqrt(rho) @ sigma @ sqrt(rho))))^2

    Args:
        rho: first density matrix
        sigma: second density matrix

    Returns:
        fidelity (float between 0 and 1)
    """
    rho = np.array(rho, dtype=complex)
    sigma = np.array(sigma, dtype=complex)

    sqrt_rho = sqrtm(rho)
    product = sqrt_rho @ sigma @ sqrt_rho
    sqrt_product = sqrtm(product)

    fidelity = np.real(np.trace(sqrt_product)) ** 2
    return float(np.clip(fidelity, 0.0, 1.0))


def channel_fidelity_sweep(channel_fn, param_range, ideal_unitary=None):
    """
    Sweep a noise parameter and compute process fidelity at each point.

    Args:
        channel_fn: function that takes a noise parameter and returns Kraus ops
        param_range: array of noise parameter values to sweep
        ideal_unitary: ideal unitary (defaults to identity)

    Returns:
        dict with 'params' and 'fidelities' arrays
    """
    if ideal_unitary is None:
        # get dimension from a test call
        test_kraus = channel_fn(0.0)
        d = test_kraus[0].shape[0]
        ideal_unitary = np.eye(d, dtype=complex)

    fidelities = []
    for p in param_range:
        kraus = channel_fn(p)
        fid = process_fidelity(kraus, ideal_unitary)
        fidelities.append(fid)

    return {
        'params': np.array(param_range),
        'fidelities': np.array(fidelities),
    }
