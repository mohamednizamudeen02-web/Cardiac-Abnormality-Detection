"""
Visualization utilities for ECG signals and model results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import MODELS_DIR

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = '#0a0e1a'
plt.rcParams['axes.facecolor'] = '#151b2e'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['grid.color'] = '#1e293b'


def plot_ecg_signal(ecg_signal, title="ECG Signal", save_path=None):
    """
    Plot a single ECG signal.
    
    Args:
        ecg_signal: 1D array of ECG values
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(15, 4))
    plt.plot(ecg_signal, color='#00d4ff', linewidth=1.5)
    plt.title(title, fontsize=16, fontweight='bold', color='#00d4ff')
    plt.xlabel('Sample', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0a0e1a')
    
    plt.show()


def plot_multiple_ecg(ecg_signals, labels=None, title="ECG Signals", save_path=None):
    """
    Plot multiple ECG signals in subplots.
    
    Args:
        ecg_signals: List of ECG signal arrays
        labels: List of labels for each signal
        title: Overall title
        save_path: Optional path to save figure
    """
    n_signals = len(ecg_signals)
    fig, axes = plt.subplots(n_signals, 1, figsize=(15, 3*n_signals))
    
    if n_signals == 1:
        axes = [axes]
    
    for i, (signal, ax) in enumerate(zip(ecg_signals, axes)):
        ax.plot(signal, color='#00d4ff', linewidth=1.5)
        label = labels[i] if labels else f"Signal {i+1}"
        ax.set_title(label, fontsize=12, color='#00d4ff')
        ax.set_xlabel('Sample', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', color='white')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0a0e1a')
    
    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Keras training history object or dict
        save_path: Optional path to save figure
    """
    if hasattr(history, 'history'):
        history = history.history
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history['loss'], label='Training Loss', color='#00d4ff', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', color='#6366f1', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold', color='#00d4ff')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['accuracy'], label='Training Accuracy', color='#00d4ff', linewidth=2)
    ax2.plot(history['val_accuracy'], label='Validation Accuracy', color='#6366f1', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold', color='#00d4ff')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0a0e1a')
    
    plt.show()


def plot_feature_importance(feature_names, importance_values, top_n=20, save_path=None):
    """
    Plot feature importance.
    
    Args:
        feature_names: List of feature names
        importance_values: Array of importance values
        top_n: Number of top features to display
        save_path: Optional path to save figure
    """
    # Sort by importance
    indices = np.argsort(importance_values)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importance_values[indices], color='#00d4ff')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance', fontsize=16, fontweight='bold', color='#00d4ff')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0a0e1a')
    
    plt.show()


def plot_signal_comparison(original, processed, title="Signal Comparison", save_path=None):
    """
    Compare original and processed signals.
    
    Args:
        original: Original ECG signal
        processed: Processed ECG signal
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    ax1.plot(original, color='#f59e0b', linewidth=1.5)
    ax1.set_title('Original Signal', fontsize=12, color='#f59e0b')
    ax1.set_xlabel('Sample', fontsize=10)
    ax1.set_ylabel('Amplitude', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(processed, color='#00d4ff', linewidth=1.5)
    ax2.set_title('Processed Signal', fontsize=12, color='#00d4ff')
    ax2.set_xlabel('Sample', fontsize=10)
    ax2.set_ylabel('Amplitude', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', color='white')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0a0e1a')
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Generating example visualizations...")
    
    # Generate sample ECG
    t = np.linspace(0, 10, 3600)
    sample_ecg = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 0.2 * t)
    
    # Plot single signal
    plot_ecg_signal(sample_ecg, "Sample ECG Signal")
    
    print("[OK] Visualization utilities working correctly!")
