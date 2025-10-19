# Quantum Noise Channel Characterization

Modeling and characterization of common quantum noise channels — depolarizing, amplitude damping, and phase damping. Computes process fidelity, average gate fidelity, and diamond distance estimates to quantify how noise degrades quantum information. Built as part of independent study work on quantum error characterization.

## Background

### Quantum Noise Channels

Real quantum hardware is noisy. Every gate operation, idle period, and measurement introduces errors that degrade quantum information. These errors can be modeled as **quantum channels** — completely positive, trace-preserving (CPTP) maps that transform density matrices.

The three most important single-qubit noise channels are:

1. **Depolarizing channel**: Replaces the qubit state with the maximally mixed state with probability $p$. This is the "worst case" noise model — it scrambles the state uniformly.

2. **Amplitude damping**: Models energy relaxation ($T_1$ decay). The excited state $|1\rangle$ decays to the ground state $|0\rangle$ with probability $\gamma$. This is the dominant error in superconducting qubits.

3. **Phase damping**: Models dephasing ($T_2$ decay) without energy loss. The off-diagonal elements of the density matrix decay, destroying coherence while preserving populations.

### Kraus Representation

Any quantum channel $\mathcal{E}$ can be written in the Kraus operator form:

$$\mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger$$

where the Kraus operators satisfy $\sum_k E_k^\dagger E_k = I$.

### Channel Metrics

- **Process fidelity**: How close the channel is to the ideal (identity) operation
- **Average gate fidelity**: Average state fidelity over all input states
- **Diamond distance**: Worst-case distinguishability between two channels

## Project Structure

```
quantum-noise-channels/
├── src/
│   ├── channels.py            # Kraus operator construction
│   ├── channel_metrics.py     # Fidelity and distance metrics
│   ├── simulation.py          # Noisy circuit simulation
│   └── plotting.py            # Visualization functions
├── notebooks/
│   └── noise_analysis.ipynb   # Full analysis walkthrough
├── results/
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

```bash
git clone https://github.com/jrouhana/quantum-noise-channels.git
cd quantum-noise-channels
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from src.channels import depolarizing_channel, apply_channel
from src.channel_metrics import process_fidelity

# Build depolarizing channel with 10% noise
kraus_ops = depolarizing_channel(p=0.1)

# Apply to |+> state
rho_plus = 0.5 * np.array([[1, 1], [1, 1]])
rho_noisy = apply_channel(rho_plus, kraus_ops)

# Measure how close the channel is to identity
fid = process_fidelity(kraus_ops, np.eye(2))
print(f"Process fidelity: {fid:.4f}")
```

### Jupyter Notebook

```bash
jupyter notebook notebooks/noise_analysis.ipynb
```

## Results

### Channel Fidelity vs Noise Parameter

Process fidelity for each channel type as noise strength increases:

| Noise Parameter | Depolarizing | Amplitude Damping | Phase Damping |
|----------------|-------------|-------------------|--------------|
| 0.00           | 1.000       | 1.000             | 1.000        |
| 0.05           | 0.963       | 0.975             | 0.975        |
| 0.10           | 0.925       | 0.950             | 0.950        |
| 0.20           | 0.850       | 0.900             | 0.900        |
| 0.30           | 0.775       | 0.852             | 0.850        |
| 0.50           | 0.625       | 0.750             | 0.750        |

*Process fidelity measures overlap with the ideal identity channel.*

### Key Findings

- Depolarizing noise degrades fidelity fastest since it scrambles in all directions
- Amplitude and phase damping have similar fidelity curves but affect different state properties
- The |+> state is most sensitive to phase damping, while |1> is most sensitive to amplitude damping
- Diamond distance provides tighter bounds than process fidelity for distinguishing channels

## References

1. Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information*. Cambridge University Press.
2. Wilde, M. M. (2017). *Quantum Information Theory*. Cambridge University Press.
3. Gilchrist, A., et al. (2005). "Distance measures to compare real and ideal quantum processes." [arXiv:quant-ph/0408063](https://arxiv.org/abs/quant-ph/0408063)

## License

MIT License — see [LICENSE](LICENSE) for details.
