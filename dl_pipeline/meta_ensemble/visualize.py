"""
Visualisation Module for Meta-Ensemble Results
===============================================

Produces a 4-panel dashboard saved as PNG:

  Panel 1 (top-left)    : Average Accuracy Bar Chart  — all models, ± std error bars
  Panel 2 (top-right)   : Accuracy Distribution       — stacked % bar (0-4 correct)
  Panel 3 (bottom-left) : Accuracy Over Simulation    — rolling-avg lines for 5 new models
  Panel 4 (bottom-right): Ensemble Weight Evolution   — stacked area chart

Usage:
    from meta_ensemble.visualize import create_full_dashboard
    create_full_dashboard(previous_results, new_results, accuracy_traces,
                          weight_history, model_names, output_dir)
"""

import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')          # non-interactive backend (safe for scripts)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    _MPL = True
except ImportError:
    _MPL = False
    print("  [visualize] matplotlib not available — skipping plots.")


# ============================================================
# Colour palette
# ============================================================
PREV_COLORS = {
    'Baseline LSTM':      '#95a5a6',
    'Enhanced LSTM':      '#7f8c8d',
    'LSTM+Markov Hybrid': '#bdc3c7',
    'Gradient Boosting':  '#aab7b8',
    'Bayesian BB':        '#a9cce3',
}
NEW_COLORS = {
    'Markov':       '#e74c3c',
    'XGBoost':      '#27ae60',
    'HistGB':       '#27ae60',
    'Logistic':     '#3498db',
    'LSTM':         '#9b59b6',
    'Meta-Ensemble': '#f39c12',
}


def _get_color(name: str) -> str:
    for k, v in {**PREV_COLORS, **NEW_COLORS}.items():
        if name.startswith(k) or k.startswith(name):
            return v
    return '#2c3e50'


# ============================================================
# Panel 1: Average Accuracy Bar Chart
# ============================================================
def _panel_accuracy_bars(ax, previous_results: dict, new_results: dict):
    """Horizontal bar chart: avg accuracy ± std for all models."""
    all_res = {**previous_results, **new_results}
    names   = list(all_res.keys())
    avgs    = [all_res[n]['average_accuracy']   for n in names]
    stds    = [all_res[n]['std_deviation']       for n in names]
    colors  = [_get_color(n)                     for n in names]

    y_pos = np.arange(len(names))
    bars  = ax.barh(y_pos, avgs, xerr=stds, height=0.65,
                    color=colors, edgecolor='white', linewidth=0.5,
                    error_kw=dict(ecolor='#2c3e50', capsize=4, elinewidth=1.5))

    # Value labels
    for bar, avg, std in zip(bars, avgs, stds):
        ax.text(bar.get_width() + std + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{avg:.1f}%', va='center', ha='left', fontsize=8.5,
                color='#2c3e50', fontweight='bold')

    # Highlight best
    best_idx = int(np.argmax(avgs))
    bars[best_idx].set_edgecolor('#f39c12')
    bars[best_idx].set_linewidth(2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Average Accuracy (%)', fontsize=9)
    ax.set_title('Average Accuracy ± Std Deviation', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlim(0, max(avgs) + max(stds) + 8)
    ax.axvline(x=max(avgs), color='#f39c12', linestyle='--', linewidth=1, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')


# ============================================================
# Panel 2: Accuracy Distribution
# ============================================================
def _panel_distribution(ax, new_results: dict):
    """Grouped horizontal stacked bar showing % of 0–4 correct per model."""
    models  = list(new_results.keys())
    n_cats  = 5   # 0/6 … 4/6 correct
    cmap    = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
    labels  = [f'{c}/6' for c in range(n_cats)]

    y_pos   = np.arange(len(models))
    bar_h   = 0.60

    for c_idx in range(n_cats):
        starts  = []
        widths  = []
        for m in models:
            # cumulative left edge
            left = sum(new_results[m]['distribution'].get(ci, 0) for ci in range(c_idx))
            starts.append(left)
            widths.append(new_results[m]['distribution'].get(c_idx, 0))
        bars = ax.barh(y_pos, widths, left=starts, height=bar_h,
                       color=cmap[c_idx], label=labels[c_idx],
                       edgecolor='white', linewidth=0.4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel('Percentage of Predictions (%)', fontsize=9)
    ax.set_title('Accuracy Distribution (% predictions)', fontsize=11,
                 fontweight='bold', pad=10)
    ax.set_xlim(0, 105)
    ax.legend(loc='lower right', fontsize=8, ncol=5, framealpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')


# ============================================================
# Panel 3: Accuracy Over Simulation
# ============================================================
def _panel_accuracy_over_time(ax, accuracy_traces: dict, window: int = 8):
    """
    Rolling-average accuracy line chart for each new model over 50 steps.
    Thin raw line in the background + bold smoothed line in front.
    """
    for name, accs in accuracy_traces.items():
        color = _get_color(name)
        steps = np.arange(1, len(accs) + 1)

        # Raw (thin, low alpha)
        ax.plot(steps, accs, color=color, alpha=0.18, linewidth=1)

        # Rolling average
        if len(accs) >= window:
            kernel = np.ones(window) / window
            smooth = np.convolve(accs, kernel, mode='valid')
            ax.plot(steps[window - 1:], smooth, color=color, linewidth=2.2,
                    label=f"{name} ({np.mean(accs):.1f}%)")
        else:
            ax.plot(steps, accs, color=color, linewidth=2.2,
                    label=f"{name} ({np.mean(accs):.1f}%)")

    ax.axhline(y=100 * 6 / 55, color='grey', linestyle=':', linewidth=1,
               label='Random baseline')
    ax.set_xlabel('Simulation Step', fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontsize=9)
    ax.set_title(f'Accuracy Over Simulation  ({window}-step rolling avg)',
                 fontsize=11, fontweight='bold', pad=10)
    ax.legend(fontsize=8, loc='upper left', framealpha=0.7)
    ax.set_ylim(-2, 65)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')


# ============================================================
# Panel 4: Ensemble Weight Evolution
# ============================================================
def _panel_weight_evolution(ax, weight_history: np.ndarray, model_names: list):
    """Stacked area chart of ensemble weight evolution over simulation steps."""
    steps   = np.arange(len(weight_history))
    colors  = [_get_color(n) for n in model_names]

    # Build cumulative stacks
    cumsum  = np.zeros(len(weight_history))
    for i, name in enumerate(model_names):
        weights_i = weight_history[:, i]
        ax.fill_between(steps, cumsum, cumsum + weights_i,
                        color=colors[i], alpha=0.80, label=name)
        ax.plot(steps, cumsum + weights_i, color=colors[i],
                linewidth=0.6, alpha=0.6)
        cumsum += weights_i

    # Annotate final weights
    for i, name in enumerate(model_names):
        final_w = weight_history[-1, i]
        mid_y   = weight_history[-1, :i].sum() + final_w / 2
        ax.text(len(weight_history) - 1, mid_y, f'{final_w:.3f}',
                va='center', ha='right', fontsize=8.5, color='white',
                fontweight='bold')

    ax.set_xlabel('Simulation Step', fontsize=9)
    ax.set_ylabel('Weight', fontsize=9)
    ax.set_title('Ensemble Weight Evolution (EMA-Adaptive)',
                 fontsize=11, fontweight='bold', pad=10)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(weight_history) - 1)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')


# ============================================================
# Main Entry Point
# ============================================================
def create_full_dashboard(
    previous_results: dict,
    new_results: dict,
    accuracy_traces: dict,
    weight_history: np.ndarray,
    model_names: list,
    output_dir: Path,
):
    """
    Create and save the 4-panel comparison dashboard.

    Args:
        previous_results : dict of previous model stats (for bar chart)
        new_results      : dict of new model stats (all 5: 4 base + ensemble)
        accuracy_traces  : dict of accuracy lists per simulation step
        weight_history   : (n_steps+1, 4) array of ensemble weights
        model_names      : list of 4 base model names
        output_dir       : Path to save PNG files
    """
    if not _MPL:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Colour style ----
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('ggplot')

    # ---- Create dashboard ----
    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor('#f8f9fa')
    gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('#ffffff')

    _panel_accuracy_bars(ax1, previous_results, new_results)
    _panel_distribution(ax2, new_results)
    _panel_accuracy_over_time(ax3, accuracy_traces)
    _panel_weight_evolution(ax4, weight_history, model_names)

    # ---- Super title ----
    best_name = max(new_results, key=lambda n: new_results[n]['average_accuracy'])
    best_avg  = new_results[best_name]['average_accuracy']
    fig.suptitle(
        f'Student Selection Prediction — Meta-Ensemble Model Comparison\n'
        f'Best model: {best_name}  ({best_avg:.2f}% avg accuracy, 50 predictions)',
        fontsize=13, fontweight='bold', color='#2c3e50', y=0.98
    )

    # ---- Save ----
    dashboard_path = output_dir / 'comparison_dashboard.png'
    plt.savefig(dashboard_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Dashboard saved: {dashboard_path}")

    # ---- Separate high-res accuracy-over-time chart ----
    fig2, ax = plt.subplots(figsize=(12, 6))
    fig2.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')
    _panel_accuracy_over_time(ax, accuracy_traces, window=6)
    fig2.suptitle('Meta-Ensemble — Per-Model Accuracy Over Simulation',
                  fontsize=12, fontweight='bold', color='#2c3e50')
    detail_path = output_dir / 'accuracy_over_time.png'
    plt.savefig(detail_path, dpi=150, bbox_inches='tight',
                facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"  Accuracy chart saved: {detail_path}")

    # ---- Separate weight evolution chart ----
    fig3, ax = plt.subplots(figsize=(10, 5))
    fig3.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')
    _panel_weight_evolution(ax, weight_history, model_names)
    fig3.suptitle('Meta-Ensemble — Adaptive Weight Evolution',
                  fontsize=12, fontweight='bold', color='#2c3e50')
    weight_path = output_dir / 'weight_evolution.png'
    plt.savefig(weight_path, dpi=150, bbox_inches='tight',
                facecolor=fig3.get_facecolor())
    plt.close(fig3)
    print(f"  Weight chart saved: {weight_path}")
