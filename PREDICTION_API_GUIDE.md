# Prediction API Documentation

## 🎯 Overview

The Prediction API provides a simple interface for making predictions and submitting corrections as separate operations, perfect for real-world classroom scenarios.

## 🔄 Workflow

```
DAY 1309 (Before Class)
  ↓
Make Prediction → 5 groups generated and saved
  ↓
(Class happens - teacher selects students)
  ↓
DAY 1309 (After Class)
  ↓
Submit Correction → Actual students recorded, accuracy calculated
  ↓
(Optional) Update Model → Model learns from correction
```

## 📋 Commands

### 1. Make Prediction (Before Class)

```bash
python prediction_api.py predict <day_number>
```

**Example:**
```bash
python prediction_api.py predict 1309
```

**Output:**
```
MAKING PREDICTION FOR DAY 1309

✓ Generated 5 groups for Day 1309:
  Group 1: [37, 12, 9, 22, 28, 51] (avg prob: 19.4%)
  Group 2: [35, 54, 50, 22, 5, 13] (avg prob: 14.4%)
  Group 3: [35, 9, 49, 28, 19, 13] (avg prob: 10.9%)
  Group 4: [42, 35, 33, 17, 48, 13] (avg prob: 13.2%)
  Group 5: [52, 28, 34, 21, 30, 53] (avg prob: 4.9%)

💾 Prediction saved. Status: PENDING correction
```

**Saved Files:**
- `predictions/prediction_day_1309.json` - Full prediction data

---

### 2. Submit Correction (After Class)

```bash
python prediction_api.py correct <day_number> <student_ids>
```

**Example:**
```bash
python prediction_api.py correct 1309 5,12,22,28,35,42
```

**Output:**
```
SUBMITTING CORRECTION FOR DAY 1309

✓ Correction received for Day 1309
   Actual students: [5, 12, 22, 28, 35, 42]

📊 Group Performance:
   Group 1: [37, 12, 9, 22, 28, 51] (overlap: 3/6) ⭐ BEST
   Group 2: [35, 54, 50, 22, 5, 13] (overlap: 3/6)
   Group 3: [35, 9, 49, 28, 19, 13] (overlap: 2/6)
   Group 4: [42, 35, 33, 17, 48, 13] (overlap: 2/6)
   Group 5: [52, 28, 34, 21, 30, 53] (overlap: 1/6)

🎯 Best Performance:
   Group 1 matched 3/6 students
   Accuracy: 50.0%
```

**Saved Files:**
- `predictions/correction_day_1309.json` - Correction data
- `predictions/predictions_summary.csv` - Updated summary

---

### 3. View Status

```bash
python prediction_api.py status
```

**Output:**
```
PREDICTION SERVICE STATUS

📊 Overall Statistics:
   Total predictions made: 3
   Pending corrections: 1
   Completed corrections: 2

🎯 Accuracy Metrics:
   Average best group accuracy: 45.0%
   Best prediction: 50.0%
   Worst prediction: 33.3%
   Completion rate: 66.7%

⏳ Pending Predictions:
   Day 1311

✅ Completed Predictions:
   Day 1309: 3/6 correct (50.0%)
   Day 1310: 2/6 correct (33.3%)
```

---

### 4. Update Model

```bash
python prediction_api.py update
```

**Output:**
```
UPDATING MODEL WITH CORRECTIONS

Training on 2 corrections:
  Days: [1309, 1310]

  Iteration 5/10, Loss: 0.2156
  Iteration 10/10, Loss: 0.2043

✓ Model updated
   Average loss: 0.2089
   Trained on 2 corrections
   Model saved: model_after_corrections.pth
```

---

## 🔧 Options

### Number of Groups

```bash
python prediction_api.py predict 1309 --groups 3
```
Generate 3 groups instead of 5.

### Auto-Update Model

```bash
python prediction_api.py correct 1309 5,12,22,28,35,42 --auto-update
```
Automatically update model immediately after correction.

---

## 📁 File Structure

```
predictions/
├── prediction_day_1309.json      # Prediction data
├── correction_day_1309.json      # Correction data
├── prediction_day_1310.json
├── correction_day_1310.json
└── predictions_summary.csv       # Summary of all completed predictions
```

### prediction_day_XXXX.json
```json
{
  "day": 1309,
  "timestamp": "2026-02-16T...",
  "predicted_groups": [[37,12,9,22,28,51], [35,54,50,22,5,13], ...],
  "probabilities": [0.123, 0.456, ...],
  "status": "pending",
  "n_groups": 5
}
```

### correction_day_XXXX.json
```json
{
  "day": 1309,
  "timestamp": "2026-02-16T...",
  "actual_students": [5,12,22,28,35,42],
  "predicted_groups": [[37,12,9,22,28,51], ...],
  "overlaps": [3, 3, 2, 2, 1],
  "accuracies": [0.5, 0.5, 0.33, 0.33, 0.17],
  "best_group": [37,12,9,22,28,51],
  "best_group_index": 0,
  "best_overlap": 3,
  "best_accuracy": 0.5,
  "status": "corrected"
}
```

### predictions_summary.csv
```csv
day,prediction_timestamp,correction_timestamp,predicted_groups,best_group,best_group_index,actual_students,overlaps,best_overlap,best_accuracy
1309,2026-02-16...,2026-02-16...,"[[37,12,9,...]]","[37,12,9,...]",0,"[5,12,22,...]","[3,3,2,2,1]",3,0.5
```

---

## 🐍 Python API Usage

Instead of command-line, you can use the Python API directly:

```python
from prediction_api import PredictionAPI

# Initialize
api = PredictionAPI()

# Make prediction
prediction = api.predict(day_number=1309, n_groups=5)
print(f"Generated {len(prediction['predicted_groups'])} groups")

# Submit correction (after class)
correction = api.correct(
    day_number=1309,
    actual_students=[5, 12, 22, 28, 35, 42],
    auto_update=False
)
print(f"Best group accuracy: {correction['best_accuracy']:.1%}")

# Update model with all corrections
result = api.update_model()
print(f"Model trained on {result['n_corrections']} corrections")

# Check status
api.status()
```

---

## ⚠️ Error Handling

### Day Already Predicted
```bash
python prediction_api.py predict 1309
# Output: ⚠ Day 1309 already has a pending prediction
```

### No Prediction Found
```bash
python prediction_api.py correct 1312 5,10,15,20,25,30
# Error: No prediction found for day 1312. Please make a prediction first.
```

### Invalid Student IDs
```bash
python prediction_api.py correct 1309 5,10,15,20,25
# Error: Expected 6 students, got 5
```

```bash
python prediction_api.py correct 1309 5,10,15,20,25,99
# Error: Student IDs must be between 1 and 55
```

---

## 📊 Typical Daily Workflow

### Morning (Before Class)
```bash
# Make prediction for today
python prediction_api.py predict 1309

# Review 5 groups, prepare for class
```

### Evening (After Class)
```bash
# Submit actual students selected
python prediction_api.py correct 1309 5,12,22,28,35,42

# Check how well prediction did
python prediction_api.py status
```

### Weekly (Update Model)
```bash
# Update model with week's corrections
python prediction_api.py update

# Model improves with more data
```

---

## 🔄 State Persistence

The API automatically saves and loads state:

- **Predictions persist** across sessions
- **Corrections persist** across sessions
- **Model updates** create new checkpoints
- **Summary CSV** continuously updated

You can safely close and restart the API - all predictions and corrections are preserved.

---

## 🚀 Advanced Usage

### Custom Model Path
```python
api = PredictionAPI(
    model_path='custom_model.pth',
    data_path='custom_data.xlsx',
    predictions_dir='custom_predictions'
)
```

### Batch Predictions
```python
for day in range(1309, 1320):
    api.predict(day)

# Later, submit corrections as they become available
for day, actual in corrections_dict.items():
    api.correct(day, actual)

# Update model once with all corrections
api.update_model()
```

### Check Specific Day Status
```python
status = api.service.get_prediction_status(1309)
# Returns: 'pending', 'completed', or 'not_found'
```

---

## 📈 Benefits of Separated Workflow

✅ **Real-world alignment**: Predictions made before class, corrections after  
✅ **Flexible timing**: No need to provide corrections immediately  
✅ **Batch training**: Collect multiple corrections, train once  
✅ **State tracking**: Clear visibility of pending vs completed  
✅ **Error recovery**: Can re-submit corrections if needed  
✅ **Audit trail**: Full history of predictions and corrections  

---

## 🎓 Example Session

```bash
# Monday morning
$ python prediction_api.py predict 1309
✓ Generated 5 groups for Day 1309

# Monday evening
$ python prediction_api.py correct 1309 5,12,22,28,35,42
✓ Group 1 matched 3/6 students (50.0%)

# Tuesday morning
$ python prediction_api.py predict 1310
✓ Generated 5 groups for Day 1310

# Wednesday morning (forgot Tuesday correction)
$ python prediction_api.py predict 1311
✓ Generated 5 groups for Day 1311

$ python prediction_api.py status
⏳ Pending: 1310, 1311
✅ Completed: 1309

# Wednesday evening (submit both)
$ python prediction_api.py correct 1310 1,5,10,15,20,25
$ python prediction_api.py correct 1311 3,8,13,18,23,28

# Friday (batch update model)
$ python prediction_api.py update
✓ Model trained on 3 corrections
```

---

## 📝 Summary

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `predict` | Generate 5 groups | **Before** class |
| `correct` | Submit actual students | **After** class |
| `status` | View statistics | Anytime |
| `update` | Train model | Periodically (weekly) |

**The API keeps prediction and correction separate, matching real-world workflow!**

---

**For technical details, see:** [prediction_service.py](dl_pipeline/prediction_service.py)  
**For multi-group info, see:** [MULTIGROUP_UPDATE.md](MULTIGROUP_UPDATE.md)
