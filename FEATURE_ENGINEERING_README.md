# Feature Engineering Summary

## Overview
Created comprehensive feature matrix for predicting next student selections with **4,019 features** per sample.

## Feature Categories

### 1. Frequency Features (220 features)
**What**: How often each student was selected in recent history
- **Windows**: 5, 10, 20, 30 days
- **Calculation**: Count of selections / window size (normalized frequency)
- **Features**: 55 students × 4 windows = 220 features
- **Example**: `freq_10d_student_22` = frequency of Student 22 in last 10 days

### 2. Days Since Last Selected (55 features)
**What**: Time elapsed since each student was last chosen
- **Calculation**: Current day - last selection day
- **Capped at**: 1000 days to prevent extreme values
- **Features**: 55 features (one per student)
- **Example**: `days_since_student_41` = days since Student 41 was selected

### 3. Pairwise Co-occurrence (3,025 features)
**What**: How often students are selected together
- **Window**: 30 days
- **Calculation**: Co-occurrence matrix (student i with student j)
- **Features**: 55 × 55 = 3,025 features
- **Example**: `cooccur_s22_s41` = frequency Student 22 and 41 selected together
- **Use case**: Captures student pairing patterns

### 4. Rolling Window Statistics (660 features)
**What**: Statistical measures over rolling windows
- **Windows**: 5, 10, 20 days
- **Statistics**: Mean, Std, Min, Max
- **Features**: 55 students × 3 windows × 4 stats = 660 features
- **Example**: `rolling_10d_mean_student_22` = average selection rate

### 5. Pattern Features (59 features)
**What**: Additional temporal and behavioral patterns
- **Overall activity**: Total selections in last 5, 10, 20 days (3 features)
- **Consecutive streaks**: Current selection streak for each student (55 features)
- **Time feature**: Normalized day in sequence (1 feature)

## Output Files

### Feature Matrices
- `X_train_features.npy`: (1,132 × 4,019) - Training features
- `X_test_features.npy`: (126 × 4,019) - Test features
- `Y_train_labels.npy`: (1,132 × 55) - Training labels (binary)
- `Y_test_labels.npy`: (126 × 55) - Test labels (binary)
- `feature_names.txt`: List of all 4,019 feature names

### Statistics
- **Feature sparsity**: 65.26% zeros (many students not selected recently)
- **Label distribution**: 10.91% (6 out of 55 students selected per day)
- **Training samples**: 1,132 (removed first 50 samples for warmup)
- **Test samples**: 126 (last ~10%)

## Data Preparation Notes

1. **Warmup period**: First 50 samples removed (insufficient historical data)
2. **Chronological split**: Maintains temporal order (no data leakage)
3. **Feature computation**: Uses only past data (no future information)
4. **Normalization**: Frequencies normalized by window size

## Usage Example

```python
import numpy as np

# Load features and labels
X_train = np.load('X_train_features.npy')
X_test = np.load('X_test_features.npy')
Y_train = np.load('Y_train_labels.npy')
Y_test = np.load('Y_test_labels.npy')

# Multi-label classification problem
# Each sample predicts 55 binary outputs (one per student)
# Typically 6 students are selected per day

print(f"Training: {X_train.shape}")  # (1132, 4019)
print(f"Labels: {Y_train.shape}")     # (1132, 55)
```

## Modeling Recommendations

### Approach 1: Multi-label Classification
- Train 55 binary classifiers (one per student)
- Use logistic regression, random forest, or neural networks
- Predict probability for each student
- Select top 6 students with highest probabilities

### Approach 2: Neural Network
- Multi-output neural network with 55 sigmoid outputs
- Loss: Binary cross-entropy per output
- Final selection: Top-k (k=6) students by probability

### Approach 3: Ensemble
- Combine multiple models
- Use feature importance to select key features
- Pairwise co-occurrence features may be most informative

## Feature Importance Insights

**Expected important features**:
1. Days since selected (students selected long ago more likely)
2. Recent frequency (patterns in selection rotation)
3. Co-occurrence patterns (certain students often selected together)
4. Consecutive streak (negative streak = overdue for selection)
