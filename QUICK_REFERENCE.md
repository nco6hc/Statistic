# Quick Reference: Multi-Group Predictions

## 🚀 Quick Start

```python
from dl_pipeline.online_pipeline import OnlineLearningPipeline

# Make prediction
groups, probs = pipeline.make_prediction(n_groups=5)

# Output: 5 groups of 6 students each
# Example: [[12, 36, 54, 33, 15, 30], [37, 42, 12, 54, 23, 24], ...]
```

## 📋 Prediction Format

Each prediction returns:
```python
predicted_groups = [
    [Student IDs for Group 1],  # 6 students
    [Student IDs for Group 2],  # 6 students  
    [Student IDs for Group 3],  # 6 students
    [Student IDs for Group 4],  # 6 students
    [Student IDs for Group 5],  # 6 students
]

probabilities = [prob1, prob2, ..., prob55]  # All 55 students
```

## 🎯 Group Characteristics

| Group | Temperature | Style | Use Case |
|-------|-------------|-------|----------|
| **1** | 1.05 | Conservative | High-confidence picks |
| **2** | 1.20 | Balanced | Safe + diverse mix |
| **3** | 1.35 | Moderate | Reasonable exploration |
| **4** | 1.50 | Diverse | Include outliers |
| **5** | 1.65 | Exploratory | Maximum variety |

## 📊 Reading Results

### Console Output:
```
DAY 5: MAKING PREDICTION

Predicted 5 groups of students:
  Group 1: [37, 5, 54, 10, 47, 23]
  Group 2: [37, 47, 42, 1, 38, 6]
  Group 3: [34, 54, 48, 29, 26, 20]  ← Best match!
  Group 4: [36, 10, 24, 4, 30, 31]
  Group 5: [36, 5, 35, 45, 4, 52]

Day 5 Feedback:
  Predicted Groups (5 total):
    Group 1: [37, 5, 54, 10, 47, 23] (overlap: 0/6)
    Group 2: [37, 47, 42, 1, 38, 6] (overlap: 0/6)
    Group 3: [34, 54, 48, 29, 26, 20] (overlap: 3/6) ⭐ BEST
    Group 4: [36, 10, 24, 4, 30, 31] (overlap: 2/6)
    Group 5: [36, 5, 35, 45, 4, 52] (overlap: 1/6)
  Actual: [24, 25, 26, 29, 36, 48]
  Best Overlap: 3 / 6
  Best Accuracy: 50.00%
```

### CSV Format:
```csv
day,predicted_groups,best_group,best_group_index,actual,overlaps,accuracies,best_overlap,best_accuracy
5,"[[37,5,54,...],[37,47,42,...],...]","[34,54,48,29,26,20]",2,"[24,25,26,29,36,48]","[0,0,3,2,1]","[0.0,0.0,0.5,0.33,0.16]",3,0.5
```

## 🔍 Interpretation Guide

### High Agreement = High Confidence
```python
# If same students appear in multiple groups:
Group 1: [12, 35, 8, ...]
Group 2: [35, 12, 54, ...]  # 12 and 35 appear again
Group 3: [35, 8, 12, ...]    # 12, 35, 8 appear again

→ Students 12, 35, 8 have HIGH confidence
```

### Low Agreement = Uncertainty
```python
# If groups are completely different:
Group 1: [1, 2, 3, 4, 5, 6]
Group 2: [7, 8, 9, 10, 11, 12]
Group 3: [13, 14, 15, 16, 17, 18]

→ Model is UNCERTAIN, all groups equally likely
```

## ✅ Validation Checks

### Within-Group Uniqueness
```python
for i, group in enumerate(predicted_groups):
    assert len(set(group)) == 6, f"Group {i+1} has duplicates!"
    assert all(1 <= s <= 55 for s in group), f"Invalid student IDs!"
```

### Cross-Group Diversity
```python
all_students = set()
for group in predicted_groups:
    all_students.update(group)

diversity = len(all_students) / 30  # 30 = 5 groups × 6 students
print(f"Diversity: {diversity:.1%}")  # Typical: 60-80%
```

## 📈 Performance Metrics

### Individual Group Accuracy
```python
for i, group in enumerate(predicted_groups):
    overlap = len(set(group) & set(actual))
    accuracy = overlap / 6
    print(f"Group {i+1}: {accuracy:.1%} ({overlap}/6)")
```

### Best Group Performance
```python
best_overlap = max(len(set(g) & set(actual)) for g in predicted_groups)
best_accuracy = best_overlap / 6
print(f"Best: {best_accuracy:.1%} ({best_overlap}/6)")
```

## 🎨 Manual Group Selection

If using predictions for decision support:

```python
# Get all groups
groups, probs = pipeline.make_prediction(n_groups=5)

# Teacher reviews and selects preferred group
print("Choose a group:")
for i, group in enumerate(groups):
    print(f"{i+1}. {group}")

selected_group_idx = int(input("Enter choice (1-5): ")) - 1
selected_group = groups[selected_group_idx]

# Use selected group
print(f"Using: {selected_group}")
```

## 🔄 Feedback Flow

```python
# 1. Make prediction
groups, probs = pipeline.make_prediction(n_groups=5)

# 2. Get actual selection (after the day)
actual = [5, 10, 15, 20, 25, 30]  # Real selection

# 3. Provide feedback (system finds best group automatically)
feedback = pipeline.receive_feedback(actual, groups, probs)

# 4. Check results
print(f"Best group was #{feedback['best_group_index'] + 1}")
print(f"Accuracy: {feedback['best_accuracy']:.1%}")

# 5. Update model
pipeline.update_model(actual)
```

## 🛠️ Customization

### Change Number of Groups
```python
# Generate 3 groups instead of 5
groups, probs = pipeline.make_prediction(n_groups=3)
```

### Adjust Temperature Range
Edit `trainer.py`:
```python
# Line ~250
group_temp = temperature * (0.7 + 0.15 * group_idx)

# Change to more conservative:
group_temp = temperature * (0.8 + 0.1 * group_idx)  # 0.8-1.2

# Change to more exploratory:
group_temp = temperature * (0.5 + 0.3 * group_idx)  # 0.5-2.0
```

## ⚠️ Common Pitfalls

### ❌ Don't assume Group 1 is always best
```python
# WRONG:
best_group = predicted_groups[0]

# CORRECT:
feedback = pipeline.receive_feedback(actual, groups, probs)
best_group = feedback['best_group']
```

### ❌ Don't expect 100% diversity
```python
# Students WILL repeat across groups (by design)
# Typical: 18-22 unique students across 5 groups (60-73%)
```

### ❌ Don't compare single-group to multi-group accuracy
```python
# Single group: 11% average
# Multi-group best: 33% average
# These are DIFFERENT metrics!
```

## 📞 Quick Debug

### No diversity between groups?
→ Check if temperature sampling is working
→ Should see different groups, not 5 copies

### All groups performing badly?
→ Model may need more training
→ Check if model checkpoint loaded correctly

### One group always wins?
→ Expected! Conservative groups often perform better
→ This is by design (exploration vs exploitation)

---

**For full documentation, see [README.md](dl_pipeline/README.md)**  
**For update details, see [MULTIGROUP_UPDATE.md](MULTIGROUP_UPDATE.md)**
