"""
Quick comparison of LSTM and Monte Carlo results
"""

print("\n" + "="*80)
print("MODEL PERFORMANCE COMPARISON".center(80))
print("="*80)

# LSTM Results (from the previous run)
lstm_results = {
    'model_type': 'LSTM Neural Network',
    'total_predictions': 50,
    'average_accuracy': 26.00,
    'best_accuracy': 66.67,
    'worst_accuracy': 0.00,
    'std_deviation': 12.54
}

# Monte Carlo Results
mc_results = {
    'model_type': 'Monte Carlo Simulation',
    'total_predictions': 66,
    'average_accuracy': 23.48,
    'best_accuracy': 50.00,
    'worst_accuracy': 0.00,
    'std_deviation': 11.60
}

print(f"\n{'Metric':<30} {'LSTM':<20} {'Monte Carlo':<20}")
print("-"*80)

metrics = [
    ('Model Type', 'model_type', lambda x: x),
    ('Total Predictions', 'total_predictions', lambda x: f"{x}"),
    ('Average Accuracy (%)', 'average_accuracy', lambda x: f"{x:.2f}"),
    ('Best Accuracy (%)', 'best_accuracy', lambda x: f"{x:.2f}"),
    ('Worst Accuracy (%)', 'worst_accuracy', lambda x: f"{x:.2f}"),
    ('Std Deviation', 'std_deviation', lambda x: f"{x:.2f}")
]

for metric_name, key, fmt in metrics:
    lstm_val = fmt(lstm_results[key])
    mc_val = fmt(mc_results[key])
    
    # Highlight better performance for numeric metrics
    if isinstance(lstm_results[key], (int, float)) and 'accuracy' in key.lower():
        if key == 'average_accuracy' or key == 'best_accuracy':
            if lstm_results[key] > mc_results[key]:
                lstm_val += " (BETTER)"
            elif mc_results[key] > lstm_results[key]:
                mc_val += " (BETTER)"
    
    print(f"{metric_name:<30} {lstm_val:<20} {mc_val:<20}")

print("="*80)

# Conclusion
print("\n" + "="*80)
print("CONCLUSION".center(80))
print("="*80)

diff = lstm_results['average_accuracy'] - mc_results['average_accuracy']

if abs(diff) < 3.0:
    print(f"\nBoth models perform similarly (difference: {abs(diff):.2f}%)")
    print(f"  LSTM: {lstm_results['average_accuracy']:.2f}%")
    print(f"  Monte Carlo: {mc_results['average_accuracy']:.2f}%")
elif lstm_results['average_accuracy'] > mc_results['average_accuracy']:
    print(f"\nLSTM outperforms Monte Carlo by {diff:.2f}%")
    print(f"  LSTM: {lstm_results['average_accuracy']:.2f}% average")
    print(f"  Monte Carlo: {mc_results['average_accuracy']:.2f}% average")
    print(f"\nThe deep learning approach captures patterns slightly better.")
else:
    print(f"\nMonte Carlo outperforms LSTM by {abs(diff):.2f}%")
    print(f"  Monte Carlo: {mc_results['average_accuracy']:.2f}% average")
    print(f"  LSTM: {lstm_results['average_accuracy']:.2f}% average")
    print(f"\nThe probabilistic approach works better for this problem.")

print("\nKey Insights:")
print("-" * 80)
print(f"  - LSTM achieved best single prediction: {lstm_results['best_accuracy']:.1f}%")
print(f"  - Monte Carlo achieved best prediction: {mc_results['best_accuracy']:.1f}%")
print(f"  - LSTM has higher variance ({lstm_results['std_deviation']:.2f} vs {mc_results['std_deviation']:.2f})")
print(f"  - Both models significantly outperform random (9.1% baseline)")
print("\nModel Characteristics:")
print("-" * 80)
print("  LSTM Advantages:")
print("    + Higher average accuracy (+2.52%)")
print("    + Better best performance (66.67% vs 50%)")
print("    + Learns complex temporal patterns")
print("\n  Monte Carlo Advantages:")
print("    + More predictions made (66 vs 50)")
print("    + Lower variance (more consistent)")
print("    + Faster and simpler")
print("    + More interpretable (probability-based)")
print("\n" + "="*80)
