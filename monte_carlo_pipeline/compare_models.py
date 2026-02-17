"""
Compare LSTM and Monte Carlo model performance
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_lstm_results():
    """Load LSTM model results from dl_checkpoints."""
    lstm_log_path = Path("../dl_pipeline/dl_checkpoints/online_predictions_log.csv")
    
    if lstm_log_path.exists():
        df = pd.read_csv(lstm_log_path)
        accuracies = []
        for _, row in df.iterrows():
            best_overlap = row['best_overlap']
            accuracy = (best_overlap / 6) * 100
            accuracies.append(accuracy)
        
        return {
            'model_type': 'LSTM',
            'total_predictions': len(accuracies),
            'average_accuracy': float(np.mean(accuracies)),
            'best_accuracy': float(np.max(accuracies)),
            'worst_accuracy': float(np.min(accuracies)),
            'std_deviation': float(np.std(accuracies)),
            'accuracies': accuracies
        }
    else:
        print(f"Warning: LSTM results not found at {lstm_log_path}")
        return None


def load_monte_carlo_results():
    """Load Monte Carlo model results."""
    mc_summary_path = Path("mc_results/summary.json")
    mc_log_path = Path("mc_results/predictions_log.json")
    
    if mc_summary_path.exists() and mc_log_path.exists():
        with open(mc_summary_path, 'r') as f:
            summary = json.load(f)
        
        with open(mc_log_path, 'r') as f:
            log = json.load(f)
        
        accuracies = [entry['best_accuracy'] for entry in log]
        summary['accuracies'] = accuracies
        
        return summary
    else:
        print("Warning: Monte Carlo results not found")
        return None


def create_comparison_table(lstm_results, mc_results):
    """Create a comparison table of both models."""
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON".center(80))
    print("=" * 80)
    
    metrics = [
        ('Model Type', 'model_type', 's'),
        ('Total Predictions', 'total_predictions', 'd'),
        ('Average Accuracy (%)', 'average_accuracy', '.2f'),
        ('Best Accuracy (%)', 'best_accuracy', '.2f'),
        ('Worst Accuracy (%)', 'worst_accuracy', '.2f'),
        ('Std Deviation', 'std_deviation', '.4f')
    ]
    
    print(f"\n{'Metric':<30} {'LSTM':<20} {'Monte Carlo':<20}")
    print("-" * 80)
    
    for metric_name, key, fmt in metrics:
        lstm_val = lstm_results.get(key, 'N/A')
        mc_val = mc_results.get(key, 'N/A')
        
        if isinstance(lstm_val, (int, float)) and isinstance(mc_val, (int, float)):
            lstm_str = f"{lstm_val:{fmt}}"
            mc_str = f"{mc_val:{fmt}}"
            
            # Highlight better performance
            if 'accuracy' in key.lower() and lstm_val != mc_val:
                if lstm_val > mc_val:
                    lstm_str += " ✓"
                else:
                    mc_str += " ✓"
        else:
            lstm_str = str(lstm_val)
            mc_str = str(mc_val)
        
        print(f"{metric_name:<30} {lstm_str:<20} {mc_str:<20}")
    
    print("=" * 80)


def create_visualizations(lstm_results, mc_results):
    """Create comparison visualizations."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LSTM vs Monte Carlo Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Accuracy over time
    ax1 = axes[0, 0]
    lstm_acc = lstm_results['accuracies']
    mc_acc = mc_results['accuracies']
    
    days = range(1, len(lstm_acc) + 1)
    ax1.plot(days, lstm_acc, 'b-', label='LSTM', alpha=0.7, linewidth=2)
    ax1.plot(days, mc_acc, 'r-', label='Monte Carlo', alpha=0.7, linewidth=2)
    ax1.set_xlabel('Prediction Day')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Prediction Accuracy Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    ax2 = axes[0, 1]
    bp = ax2.boxplot([lstm_acc, mc_acc], labels=['LSTM', 'Monte Carlo'],
                      patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Distribution')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Accuracy histogram
    ax3 = axes[1, 0]
    bins = np.arange(0, 110, 10)
    ax3.hist(lstm_acc, bins=bins, alpha=0.5, label='LSTM', color='blue', edgecolor='black')
    ax3.hist(mc_acc, bins=bins, alpha=0.5, label='Monte Carlo', color='red', edgecolor='black')
    ax3.set_xlabel('Accuracy (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Accuracy Histogram')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Bar chart comparison
    ax4 = axes[1, 1]
    metrics = ['Average', 'Best', 'Worst']
    lstm_vals = [lstm_results['average_accuracy'], 
                 lstm_results['best_accuracy'],
                 lstm_results['worst_accuracy']]
    mc_vals = [mc_results['average_accuracy'],
               mc_results['best_accuracy'],
               mc_results['worst_accuracy']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax4.bar(x - width/2, lstm_vals, width, label='LSTM', color='lightblue', edgecolor='black')
    ax4.bar(x + width/2, mc_vals, width, label='Monte Carlo', color='lightcoral', edgecolor='black')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Performance Metrics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path("mc_results/model_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: {output_path}")
    
    plt.show()


def analyze_accuracy_distribution(lstm_results, mc_results):
    """Analyze accuracy distribution for both models."""
    print("\n" + "=" * 80)
    print("ACCURACY DISTRIBUTION ANALYSIS".center(80))
    print("=" * 80)
    
    print(f"\n{'Correct/6':<15} {'LSTM Count':<15} {'MC Count':<15} {'LSTM %':<15} {'MC %':<15}")
    print("-" * 80)
    
    for correct in range(7):
        accuracy_val = (correct / 6) * 100
        
        lstm_count = sum(1 for acc in lstm_results['accuracies'] if abs(acc - accuracy_val) < 1.0)
        mc_count = sum(1 for acc in mc_results['accuracies'] if abs(acc - accuracy_val) < 1.0)
        
        lstm_pct = (lstm_count / len(lstm_results['accuracies'])) * 100
        mc_pct = (mc_count / len(mc_results['accuracies'])) * 100
        
        print(f"{correct}/6{'':<10} {lstm_count:<15} {mc_count:<15} {lstm_pct:>6.1f}%{'':<8} {mc_pct:>6.1f}%")
    
    print("=" * 80)


def main():
    print("\n" + "=" * 80)
    print("LOADING MODEL RESULTS".center(80))
    print("=" * 80)
    
    # Load results
    lstm_results = load_lstm_results()
    mc_results = load_monte_carlo_results()
    
    if lstm_results is None:
        print("\nERROR: Could not load LSTM results. Please run the LSTM pipeline first.")
        return
    
    if mc_results is None:
        print("\nERROR: Could not load Monte Carlo results. Please run monte_carlo_model.py first.")
        return
    
    print(f"\nSuccessfully loaded results for both models:")
    print(f"  LSTM: {lstm_results['total_predictions']} predictions")
    print(f"  Monte Carlo: {mc_results['total_predictions']} predictions")
    
    # Create comparison
    create_comparison_table(lstm_results, mc_results)
    
    # Analyze distributions
    analyze_accuracy_distribution(lstm_results, mc_results)
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS".center(80))
    print("=" * 80)
    create_visualizations(lstm_results, mc_results)
    
    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION".center(80))
    print("=" * 80)
    
    lstm_avg = lstm_results['average_accuracy']
    mc_avg = mc_results['average_accuracy']
    
    if abs(lstm_avg - mc_avg) < 1.0:
        print(f"\nBoth models perform similarly (~{lstm_avg:.1f}% average accuracy).")
        print("Consider:")
        print("  • LSTM: Better at learning complex temporal patterns")
        print("  • Monte Carlo: Simpler, faster, more interpretable")
    elif lstm_avg > mc_avg:
        diff = lstm_avg - mc_avg
        print(f"\nLSTM outperforms Monte Carlo by {diff:.2f}% on average.")
        print(f"LSTM: {lstm_avg:.2f}% vs Monte Carlo: {mc_avg:.2f}%")
        print("The deep learning approach captures patterns better than probabilistic sampling.")
    else:
        diff = mc_avg - lstm_avg
        print(f"\nMonte Carlo outperforms LSTM by {diff:.2f}% on average!")
        print(f"Monte Carlo: {mc_avg:.2f}% vs LSTM: {lstm_avg:.2f}%")
        print("The probabilistic approach works better for this problem.")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
