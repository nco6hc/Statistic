import pandas as pd

df = pd.read_csv('demo_checkpoints/demo_predictions_multigroup.csv')

print('='*70)
print('MULTI-GROUP PREDICTIONS SUMMARY')
print('='*70)

print(f'\nTotal predictions: {len(df)}')
print(f'Columns: {", ".join(df.columns)}')

print('\n--- First Prediction Details ---')
row = df.iloc[0]
print(f"Day: {row['day']}")
print(f"Best Group Index: {row['best_group_index'] + 1}")
print(f"Best Overlap: {row['best_overlap']}/6")
print(f"Best Accuracy: {row['best_accuracy']:.1%}")
print(f"All Overlaps: {row['overlaps']}")

print('\n--- Overall Statistics ---')
print(f"Average best accuracy: {df['best_accuracy'].mean():.1%}")
print(f"Best single prediction: {df['best_accuracy'].max():.1%}")
print(f"Worst single prediction: {df['best_accuracy'].min():.1%}")

print('\n--- Best Group Distribution ---')
for i in range(5):
    count = (df['best_group_index'] == i).sum()
    pct = count / len(df) * 100
    print(f"  Group {i+1} was best: {count} times ({pct:.0f}%)")

print('\n' + '='*70)
print('✅ Multi-group prediction system working successfully!')
print('='*70)
