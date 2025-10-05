"""
Visualization functions for noise channel analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_fidelity_vs_noise(results_dict, save_dir='results'):
    """
    Plot state fidelity vs noise parameter for multiple channels.

    Args:
        results_dict: dict mapping channel name to results dict
                      (each must have 'params' and 'fidelities')
        save_dir: directory to save the plot
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'Depolarizing': '#FF6B6B', 'Amplitude Damping': '#4ECDC4',
              'Phase Damping': '#45B7D1'}
    markers = {'Depolarizing': 'o', 'Amplitude Damping': 's',
               'Phase Damping': '^'}

    for name, data in results_dict.items():
        ax.plot(data['params'], data['fidelities'],
                marker=markers.get(name, 'o'),
                color=colors.get(name, 'gray'),
                linewidth=2, markersize=6, label=name)

    ax.set_xlabel('Noise Parameter', fontsize=12)
    ax.set_ylabel('State Fidelity', fontsize=12)
    ax.set_title('State Fidelity Degradation Under Noise Channels')
    ax.legend(fontsize=11)
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'fidelity_vs_noise.png', dpi=150)
    plt.close()


def plot_channel_comparison(channel_results, save_dir='results'):
    """
    Side-by-side comparison of process fidelity for different channels.

    Args:
        channel_results: dict mapping channel name to dict with
                         'params' and 'fidelities' arrays
        save_dir: directory to save the plot
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for idx, (name, data) in enumerate(channel_results.items()):
        ax = axes[idx]
        ax.plot(data['params'], data['fidelities'], 'o-',
                color=colors[idx], linewidth=2, markersize=5)
        ax.fill_between(data['params'], data['fidelities'], alpha=0.15,
                        color=colors[idx])
        ax.set_xlabel('Noise Parameter', fontsize=11)
        ax.set_ylabel('Fidelity', fontsize=11)
        ax.set_title(name, fontsize=12)
        ax.set_ylim(0.3, 1.05)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Channel Comparison: Fidelity vs Noise Strength', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path / 'channel_comparison.png', dpi=150)
    plt.close()


def plot_bloch_trajectory(states, labels=None, save_dir='results'):
    """
    Plot the Bloch vector trajectory as a state evolves under noise.

    Takes a list of 2x2 density matrices and extracts the Bloch vector
    coordinates (rx, ry, rz) for each.

    Args:
        states: list of 2x2 density matrices
        labels: optional list of labels for each point
        save_dir: directory to save the plot
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # extract Bloch coordinates from each density matrix
    rx_vals, ry_vals, rz_vals = [], [], []
    for rho in states:
        rho = np.array(rho)
        rx = 2 * np.real(rho[0, 1])
        ry = 2 * np.imag(rho[1, 0])
        rz = np.real(rho[0, 0] - rho[1, 1])
        rx_vals.append(rx)
        ry_vals.append(ry)
        rz_vals.append(rz)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    pairs = [('rx', 'ry', rx_vals, ry_vals),
             ('rx', 'rz', rx_vals, rz_vals),
             ('ry', 'rz', ry_vals, rz_vals)]

    for ax, (xlabel, ylabel, xdata, ydata) in zip(axes, pairs):
        # draw unit circle
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.2, linewidth=1)

        # trajectory with color gradient
        n = len(xdata)
        colors = plt.cm.viridis(np.linspace(0, 0.9, n))
        for i in range(n - 1):
            ax.plot([xdata[i], xdata[i + 1]], [ydata[i], ydata[i + 1]],
                    color=colors[i], linewidth=2)

        ax.scatter(xdata[0], ydata[0], color='green', s=100,
                   zorder=5, label='Start')
        ax.scatter(xdata[-1], ydata[-1], color='red', s=100,
                   zorder=5, label='End')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.suptitle('Bloch Vector Trajectory Under Noise', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path / 'bloch_trajectory.png', dpi=150)
    plt.close()


def plot_density_matrix(rho, title='', save_dir='results', filename='density_matrix.png'):
    """
    Visualize a density matrix as a pair of heatmaps (real and imaginary parts).

    Args:
        rho: density matrix (numpy array)
        title: plot title
        save_dir: directory to save the plot
        filename: output filename
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    vmax = max(np.abs(np.real(rho)).max(), np.abs(np.imag(rho)).max())

    im0 = axes[0].imshow(np.real(rho), cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0].set_title('Real Part')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(np.imag(rho), cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1].set_title('Imaginary Part')
    plt.colorbar(im1, ax=axes[1])

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path / filename, dpi=150)
    plt.close()


def plot_diamond_distance_comparison(param_range, distances_dict, save_dir='results'):
    """
    Plot diamond distance vs noise parameter for channel pairs.

    Args:
        param_range: array of noise parameter values
        distances_dict: dict mapping comparison name to array of distances
        save_dir: directory to save the plot
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#9B59B6']

    for idx, (name, distances) in enumerate(distances_dict.items()):
        ax.plot(param_range, distances, 'o-',
                color=colors[idx % len(colors)],
                linewidth=2, markersize=6, label=name)

    ax.set_xlabel('Noise Parameter', fontsize=12)
    ax.set_ylabel('Diamond Distance (upper bound)', fontsize=12)
    ax.set_title('Diamond Distance: Noisy Channel vs Identity')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'diamond_distance.png', dpi=150)
    plt.close()
