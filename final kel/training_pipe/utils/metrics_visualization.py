#!/usr/bin/env python3
"""
Metrics Visualization for Kelly MIDI Companion ML Training
============================================================
Provides plotting and visualization of training metrics.
"""

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting features will be disabled.")

import numpy as np
from pathlib import Path
from typing import List, Optional, Dict

# Handle both relative and absolute imports
try:
    from .training_utils import TrainingMetrics
except ImportError:
    from training_utils import TrainingMetrics


def plot_training_curves(
    metrics: TrainingMetrics,
    output_path: Path,
    model_name: str = "Model",
    show_learning_rate: bool = True
):
    """
    Plot training and validation curves.
    
    Args:
        metrics: TrainingMetrics object with training history
        output_path: Path to save the plot
        model_name: Name of the model for title
        show_learning_rate: Whether to plot learning rate curve
    """
    if not MATPLOTLIB_AVAILABLE:
        print(f"Warning: matplotlib not available. Skipping plot generation for {output_path}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_name} - Training Curves', fontsize=16, fontweight='bold')

    epochs = range(1, len(metrics.train_loss) + 1)

    # Loss curve
    ax = axes[0, 0]
    ax.plot(epochs, metrics.train_loss, 'b-', label='Train Loss', linewidth=2)
    if metrics.val_loss:
        ax.plot(epochs, metrics.val_loss, 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Accuracy curve (if available)
    ax = axes[0, 1]
    if metrics.train_accuracy and len(metrics.train_accuracy) > 0:
        ax.plot(epochs, metrics.train_accuracy, 'b-', label='Train Accuracy', linewidth=2)
    if metrics.val_accuracy and len(metrics.val_accuracy) > 0:
        ax.plot(epochs, metrics.val_accuracy, 'r-', label='Val Accuracy', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Learning rate curve
    ax = axes[1, 0]
    if metrics.learning_rate and len(metrics.learning_rate) > 0 and show_learning_rate:
        ax.plot(epochs, metrics.learning_rate, 'g-', label='Learning Rate', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No learning rate data',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')

    # Epoch times
    ax = axes[1, 1]
    if metrics.epoch_times and len(metrics.epoch_times) > 0:
        ax.plot(epochs, metrics.epoch_times, 'm-', label='Epoch Time', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No timing data',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Training Time per Epoch', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved to {output_path}")


def plot_comparison_curves(
    metrics_list: List[TrainingMetrics],
    labels: List[str],
    output_path: Path,
    metric_type: str = 'loss'
):
    """
    Plot comparison of multiple training runs.
    
    Args:
        metrics_list: List of TrainingMetrics objects
        labels: List of labels for each run
        output_path: Path to save the plot
        metric_type: 'loss' or 'accuracy'
    """
    if not MATPLOTLIB_AVAILABLE:
        print(f"Warning: matplotlib not available. Skipping comparison plot for {output_path}")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'Training Comparison - {metric_type.capitalize()}',
                 fontsize=16, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_list)))

    for metrics, label, color in zip(metrics_list, labels, colors):
        epochs = range(1, len(metrics.train_loss) + 1)

        if metric_type == 'loss':
            train_metric = metrics.train_loss
            val_metric = metrics.val_loss
            ylabel = 'Loss'
        else:  # accuracy
            train_metric = metrics.train_accuracy if metrics.train_accuracy else []
            val_metric = metrics.val_accuracy if metrics.val_accuracy else []
            ylabel = 'Accuracy'

        # Training curves
        ax1.plot(epochs, train_metric, color=color, label=f'{label} (Train)',
                linewidth=2, alpha=0.7)

        # Validation curves
        if val_metric and len(val_metric) > 0:
            ax2.plot(epochs, val_metric, color=color, label=f'{label} (Val)',
                    linewidth=2, alpha=0.7)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_title('Training', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=12)
    ax2.set_title('Validation', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    if metric_type == 'accuracy':
        ax1.set_ylim([0, 1])
        ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Comparison plot saved to {output_path}")


def create_training_summary(
    metrics: TrainingMetrics,
    output_path: Path,
    model_name: str = "Model"
):
    """
    Create a text summary of training results.

    Args:
        metrics: TrainingMetrics object
        output_path: Path to save the summary
        model_name: Name of the model
    """
    with open(output_path, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"{model_name} - Training Summary\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"Total Epochs: {len(metrics.train_loss)}\n\n")

        if metrics.train_loss:
            f.write("Training Loss:\n")
            f.write(f"  Initial: {metrics.train_loss[0]:.6f}\n")
            f.write(f"  Final: {metrics.train_loss[-1]:.6f}\n")
            f.write(f"  Best: {min(metrics.train_loss):.6f} (epoch {metrics.train_loss.index(min(metrics.train_loss)) + 1})\n")
            f.write(f"  Improvement: {((metrics.train_loss[0] - metrics.train_loss[-1]) / metrics.train_loss[0] * 100):.2f}%\n\n")

        if metrics.val_loss:
            f.write("Validation Loss:\n")
            f.write(f"  Initial: {metrics.val_loss[0]:.6f}\n")
            f.write(f"  Final: {metrics.val_loss[-1]:.6f}\n")
            f.write(f"  Best: {min(metrics.val_loss):.6f} (epoch {metrics.val_loss.index(min(metrics.val_loss)) + 1})\n")
            f.write(f"  Improvement: {((metrics.val_loss[0] - metrics.val_loss[-1]) / metrics.val_loss[0] * 100):.2f}%\n\n")

        if metrics.train_accuracy and len(metrics.train_accuracy) > 0:
            f.write("Training Accuracy:\n")
            f.write(f"  Initial: {metrics.train_accuracy[0]:.4f}\n")
            f.write(f"  Final: {metrics.train_accuracy[-1]:.4f}\n")
            f.write(f"  Best: {max(metrics.train_accuracy):.4f} (epoch {metrics.train_accuracy.index(max(metrics.train_accuracy)) + 1})\n\n")

        if metrics.val_accuracy and len(metrics.val_accuracy) > 0:
            f.write("Validation Accuracy:\n")
            f.write(f"  Initial: {metrics.val_accuracy[0]:.4f}\n")
            f.write(f"  Final: {metrics.val_accuracy[-1]:.4f}\n")
            f.write(f"  Best: {max(metrics.val_accuracy):.4f} (epoch {metrics.val_accuracy.index(max(metrics.val_accuracy)) + 1})\n\n")

        if metrics.learning_rate and len(metrics.learning_rate) > 0:
            f.write("Learning Rate:\n")
            f.write(f"  Initial: {metrics.learning_rate[0]:.6f}\n")
            f.write(f"  Final: {metrics.learning_rate[-1]:.6f}\n\n")

        if metrics.epoch_times and len(metrics.epoch_times) > 0:
            total_time = sum(metrics.epoch_times)
            avg_time = np.mean(metrics.epoch_times)
            f.write("Training Time:\n")
            f.write(f"  Total: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
            f.write(f"  Average per epoch: {avg_time:.2f} seconds\n\n")

        f.write(f"{'='*60}\n")

    print(f"Training summary saved to {output_path}")
