"""
Noisy circuit simulation using Qiskit Aer noise models.
Sweeps noise parameters and collects measurement statistics.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error

from .channels import depolarizing_channel, amplitude_damping, phase_damping, apply_channel
from .channel_metrics import state_fidelity


def build_noise_model(channel_fn, noise_param):
    """
    Build a Qiskit noise model from a channel type and noise parameter.

    Maps our channel functions to Qiskit's built-in noise model errors
    so we can run noisy simulations through AerSimulator.

    Args:
        channel_fn: one of depolarizing_channel, amplitude_damping, phase_damping
        noise_param: noise strength parameter

    Returns:
        NoiseModel configured with the specified noise on all single-qubit gates
    """
    noise_model = NoiseModel()

    if channel_fn is depolarizing_channel:
        error = depolarizing_error(noise_param, 1)
    elif channel_fn is amplitude_damping:
        error = amplitude_damping_error(noise_param)
    elif channel_fn is phase_damping:
        error = phase_damping_error(noise_param)
    else:
        raise ValueError(f"Unsupported channel function: {channel_fn}")

    # apply noise after every single-qubit gate
    noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3', 'id',
                                                      'x', 'y', 'z', 'h',
                                                      's', 't', 'rx', 'ry', 'rz'])
    return noise_model


def simulate_channel_circuit(circuit, channel_fn, noise_param, shots=4096, seed=42):
    """
    Run a circuit with a specified noise channel applied.

    Args:
        circuit: QuantumCircuit to simulate
        channel_fn: noise channel function (depolarizing_channel, etc.)
        noise_param: noise strength
        shots: number of measurement shots
        seed: random seed for reproducibility

    Returns:
        dict with 'counts', 'noise_param', 'shots'
    """
    noise_model = build_noise_model(channel_fn, noise_param)

    backend = AerSimulator(noise_model=noise_model)
    job = backend.run(circuit, shots=shots, seed_simulator=seed)
    result = job.result()
    counts = result.get_counts()

    return {
        'counts': counts,
        'noise_param': noise_param,
        'shots': shots,
    }


def sweep_noise_parameter(circuit, channel_fn, param_range, shots=4096, seed=42):
    """
    Run a circuit at multiple noise levels and collect results.

    For each noise parameter value, runs the circuit through a noisy
    simulation and records the measurement distribution. Also computes
    the probability of measuring the ideal (noiseless) outcome.

    Args:
        circuit: QuantumCircuit to simulate
        channel_fn: noise channel function
        param_range: array of noise parameter values
        shots: number of shots per run
        seed: random seed

    Returns:
        dict with 'params', 'counts_list', 'ideal_probs'
    """
    # first run the ideal (noiseless) circuit to find the target outcome
    ideal_backend = AerSimulator()
    ideal_job = ideal_backend.run(circuit, shots=shots, seed_simulator=seed)
    ideal_counts = ideal_job.result().get_counts()
    ideal_bitstring = max(ideal_counts, key=ideal_counts.get)

    counts_list = []
    ideal_probs = []

    for param in param_range:
        if param == 0.0:
            # no noise — use ideal result
            counts_list.append(ideal_counts)
            ideal_probs.append(ideal_counts.get(ideal_bitstring, 0) / shots)
            continue

        result = simulate_channel_circuit(circuit, channel_fn, param,
                                           shots=shots, seed=seed)
        counts = result['counts']
        counts_list.append(counts)

        # probability of getting the ideal outcome
        ideal_prob = counts.get(ideal_bitstring, 0) / shots
        ideal_probs.append(ideal_prob)

    return {
        'params': np.array(param_range),
        'counts_list': counts_list,
        'ideal_probs': np.array(ideal_probs),
        'ideal_bitstring': ideal_bitstring,
    }


def compute_channel_output_states(input_rho, channel_fn, param_range):
    """
    Apply a channel at different noise strengths to an input state.

    This uses the Kraus operator representation directly (no circuit
    simulation), which gives exact results.

    Args:
        input_rho: input density matrix
        channel_fn: channel function returning Kraus operators
        param_range: noise parameter values

    Returns:
        dict with 'params', 'output_states', 'fidelities'
    """
    input_rho = np.array(input_rho, dtype=complex)
    output_states = []
    fidelities = []

    for param in param_range:
        kraus_ops = channel_fn(param)
        rho_out = apply_channel(input_rho, kraus_ops)
        output_states.append(rho_out)

        fid = state_fidelity(input_rho, rho_out)
        fidelities.append(fid)

    return {
        'params': np.array(param_range),
        'output_states': output_states,
        'fidelities': np.array(fidelities),
    }


def compare_channels(input_rho, param_range):
    """
    Compare all three channel types on the same input state.

    Args:
        input_rho: input density matrix
        param_range: noise parameter values to sweep

    Returns:
        dict mapping channel name to results dict
    """
    channels = {
        'Depolarizing': depolarizing_channel,
        'Amplitude Damping': amplitude_damping,
        'Phase Damping': phase_damping,
    }

    results = {}
    for name, channel_fn in channels.items():
        results[name] = compute_channel_output_states(
            input_rho, channel_fn, param_range
        )

    return results
