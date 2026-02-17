# Student Predictor Batch File Guide

## 📝 Overview

The `student_predictor.bat` file provides an **interactive menu** for the Student Selection Prediction System.

---

## 🚀 How to Use

### Launch the Program

Double-click `student_predictor.bat` or run in terminal:
```cmd
student_predictor.bat
```

### Menu Options

```
================================================================
          STUDENT SELECTION PREDICTION SYSTEM
================================================================

  1. PREDICT    - Generate predictions for a day
  2. CORRECT    - Submit actual results after class
  3. TRAIN      - Update model with corrections
  4. STATUS     - View current status
  5. EXIT       - Close program

================================================================
```

---

## 📋 Step-by-Step Workflow

### 1️⃣ **PREDICT** - Before Class

**When to use:** Before the class day starts, to get predictions.

**Steps:**
1. Select option `1` from menu
2. Enter day number (e.g., `1310`)
3. System generates 5 groups of 6 students each
4. Review the predictions (printed on screen)

**Example:**
```
Enter day number: 1310

Group 1: [5, 12, 18, 25, 34, 41]
Group 2: [3, 12, 19, 28, 35, 42]
Group 3: [5, 11, 22, 28, 33, 44]
Group 4: [2, 13, 20, 27, 36, 45]
Group 5: [6, 14, 21, 29, 37, 43]
```

---

### 2️⃣ **CORRECT** - After Class

**When to use:** After class ends and you know which 6 students were actually selected.

**Steps:**
1. Select option `2` from menu
2. Enter day number (e.g., `1310`)
3. Enter the 6 student IDs that were actually selected
   - Format: `student1,student2,student3,student4,student5,student6`
   - Example: `6,11,22,33,55,12`
4. System compares with predictions and shows accuracy
5. **Database automatically updated** with actual results

**Example:**
```
Enter day number: 1310
Enter student IDs: 5,12,22,28,35,42

Group 1: [5, 12, 18, 25, 34, 41]
         Overlap: 2/6 (33.3%)
Group 2: [3, 12, 19, 28, 35, 42]
         Overlap: 4/6 (66.7%)
Group 3: [5, 11, 22, 28, 33, 44] ⭐ BEST
         Overlap: 4/6 (66.7%)

💾 Database updated with Day 1310 actual results.
```

**Important:** The Excel database (`Database.xlsx`) is **automatically updated** with the actual student selections.

---

### 3️⃣ **TRAIN** - Weekly Model Update

**When to use:** Every 1-2 weeks after collecting multiple corrections.

**Steps:**
1. Select option `3` from menu
2. Confirm training with `Y`
3. Model trains on all collected corrections
4. Improved model ready for next predictions

**Example:**
```
This will update the model with all collected corrections.
This should be done every 1-2 weeks.

Continue? (Y/N): Y

Training on 10 corrections:
  Days: [1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310]

Training... [████████████████] 100%

✓ Model updated successfully
  Average accuracy improved from 45.2% to 52.8%
```

---

### 4️⃣ **STATUS** - Check Progress

**When to use:** Anytime to see current system status.

**Shows:**
- Number of pending predictions
- Number of completed predictions
- Average accuracy across all predictions
- List of pending and completed days

**Example:**
```
📊 Current Status:
   Pending predictions: 2
   Days: [1311, 1312]
   
   Completed predictions: 10
   Days: [1301-1310]
   
   Average accuracy: 52.8%
   Best accuracy: 66.7% (Day 1310)
```

---

## 🔄 Typical Weekly Workflow

### Monday - Friday (Daily)
1. **Morning:** Run `PREDICT` for today's class
2. **After class:** Run `CORRECT` with actual students
3. Repeat for each class day

### Weekend (Weekly)
1. Run `STATUS` to review weekly performance
2. Run `TRAIN` to update model with week's data
3. Model is now improved for next week

---

## 💾 Database Auto-Update

When you submit a correction (Option 2), the system **automatically**:

1. ✅ Saves correction to `predictions/` folder
2. ✅ Updates in-memory data for future predictions
3. ✅ **Writes back to `Database.xlsx`** with actual students
4. ✅ Records accuracy metrics

**No manual database editing needed!**

---

## 📁 Files Created

The system creates these files automatically:

```
predictions/
├── prediction_day_1310.json      ← Prediction record
├── correction_day_1310.json      ← Correction record
└── predictions_summary.csv       ← All results summary
```

---

## ⚠️ Important Notes

1. **Always predict before correcting** - You must run PREDICT first for a day
2. **Database backup** - System writes to `Database.xlsx` automatically
3. **Training frequency** - Train every 1-2 weeks for best results
4. **Student IDs** - Must be between 1 and 55
5. **Format** - Use commas without spaces: `6,11,22,33,55,12`

---

## 🐛 Troubleshooting

### Error: "No prediction found for day X"
- **Solution:** Run PREDICT first before CORRECT

### Error: "Expected 6 students, got Y"
- **Solution:** Make sure you enter exactly 6 student IDs

### Error: "Student IDs must be between 1 and 55"
- **Solution:** Check that all IDs are valid (1-55)

### Database not updating
- **Solution:** Check that `Database.xlsx` is not open in Excel

---

## 🎯 Quick Reference

| Option | Command | When to Use |
|--------|---------|-------------|
| 1 | PREDICT | Before class (morning) |
| 2 | CORRECT | After class (with actual students) |
| 3 | TRAIN | Weekly (1-2 weeks) |
| 4 | STATUS | Anytime (check progress) |
| 5 | EXIT | Close program |

---

## 💡 Tips

1. **Keep predictions up-to-date:** Predict and correct every class day
2. **Train regularly:** Weekly training keeps model accurate
3. **Review status weekly:** Check accuracy trends
4. **Backup database:** Copy `Database.xlsx` before major updates
5. **Use consistently:** More data = better predictions

---

**Ready to start?** Run `student_predictor.bat` now!
