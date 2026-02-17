# 📖 Batch File - Visual User Guide

## Step-by-Step Instructions with Examples

---

## 🚀 STEP 1: Open the Program

**Action:** Double-click `student_predictor.bat`

**You will see:**
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
Enter your choice (1-5):
```

---

## 📅 STEP 2: Make a Prediction (Before Class)

### When to do this?
**BEFORE** the class starts (e.g., in the morning)

### What to do?

**1. Type `1` and press Enter**
```
Enter your choice (1-5): 1
```

**2. Enter the day number (e.g., 1312)**
```
================================================================
                   GENERATE PREDICTIONS
================================================================

Enter day number (e.g., 1310): 1312
```

### What you'll see:
```
Generating predictions for Day 1312...

✓ Generated 5 groups for Day 1312:
  Group 1: [5, 12, 18, 25, 34, 41] (avg prob: 18.2%)
  Group 2: [3, 12, 19, 28, 35, 42] (avg prob: 16.5%)
  Group 3: [5, 11, 22, 28, 33, 44] (avg prob: 14.8%)
  Group 4: [2, 13, 20, 27, 36, 45] (avg prob: 12.3%)
  Group 5: [6, 14, 21, 29, 37, 43] (avg prob: 10.1%)

💾 Prediction saved. Status: PENDING correction

================================================================
Press any key to continue...
```

**3. Press any key to return to menu**

---

## ✅ STEP 3: Submit Correction (After Class)

### When to do this?
**AFTER** the class ends and you know which 6 students were selected

### What to do?

**1. Type `2` and press Enter**
```
Enter your choice (1-5): 2
```

**2. Enter the same day number (e.g., 1312)**
```
================================================================
                  SUBMIT CORRECTIONS
================================================================

Enter day number (e.g., 1310): 1312
```

**3. Enter the 6 student IDs (actual selected students)**
```
Enter the 6 student IDs that were actually selected
Format: student1,student2,student3,student4,student5,student6
Example: 6,11,22,33,55,12

Enter student IDs: 5,12,22,28,35,42
```

### What you'll see:
```
Submitting correction for Day 1312...

✓ Correction received for Day 1312
   Actual students: [5, 12, 22, 28, 35, 42]

📊 Group Performance:
   Group 1: [5, 12, 18, 25, 34, 41]
            Overlap: 2/6 (33.3%)
   Group 2: [3, 12, 19, 28, 35, 42]
            Overlap: 4/6 (66.7%)
   Group 3: [5, 11, 22, 28, 33, 44] ⭐ BEST
            Overlap: 4/6 (66.7%)
   Group 4: [2, 13, 20, 27, 36, 45]
            Overlap: 0/6 (0.0%)
   Group 5: [6, 14, 21, 29, 37, 43]
            Overlap: 0/6 (0.0%)

🎯 Best Performance:
   Group 3 matched 4/6 students
   Accuracy: 66.7%

💾 ✅ Database updated with Day 1312 actual results.
   Excel file: Database.xlsx
   Data ready for future predictions.

💡 Tip: Call update_model() to train on collected corrections

================================================================
Press any key to continue...
```

**4. Press any key to return to menu**

---

## 📊 STEP 4: Check Status (Anytime)

### When to do this?
Anytime you want to see your progress

### What to do?

**Type `4` and press Enter**
```
Enter your choice (1-5): 4
```

### What you'll see:
```
================================================================
                    SYSTEM STATUS
================================================================

📊 Overall Statistics:
   Total predictions made: 4
   Pending corrections: 0
   Completed corrections: 4

🎯 Accuracy Metrics:
   Average best group accuracy: 58.3%
   Best prediction: 66.7%
   Worst prediction: 50.0%
   Completion rate: 100.0%

✅ Completed Predictions (last 10):
   Day 1309: 3/6 correct (50.0%)
   Day 1310: 3/6 correct (50.0%)
   Day 1311: 4/6 correct (66.7%)
   Day 1312: 4/6 correct (66.7%)

================================================================
Press any key to continue...
```

---

## 🔄 STEP 5: Train Model (Weekly)

### When to do this?
Every **1-2 weeks** after collecting multiple corrections

### What to do?

**1. Type `3` and press Enter**
```
Enter your choice (1-5): 3
```

**2. Confirm training**
```
================================================================
                   UPDATE MODEL
================================================================

This will update the model with all collected corrections.
This should be done every 1-2 weeks.

Continue? (Y/N): Y
```

### What you'll see:
```
Updating model...

Training on 10 corrections:
  Days: [1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318]

Epoch 1/10: Loss = 0.234
Epoch 2/10: Loss = 0.198
...
Epoch 10/10: Loss = 0.156

✅ Model updated successfully
   Previous accuracy: 55.2%
   New accuracy: 62.8%
   Improvement: +7.6%

================================================================
Press any key to continue...
```

---

## 🔄 Weekly Workflow Example

### Monday
```
Morning:
  → Run bat file
  → Option 1: PREDICT
  → Day: 1312
  → Get 5 groups

After class:
  → Run bat file
  → Option 2: CORRECT
  → Day: 1312
  → Students: 5,12,22,28,35,42
  → Database auto-updates ✅
```

### Tuesday-Friday
```
(Repeat Monday's steps for each day)
  → PREDICT in morning
  → CORRECT after class
```

### Saturday (Weekend)
```
Review performance:
  → Run bat file
  → Option 4: STATUS
  → Check weekly accuracy

Update model:
  → Run bat file
  → Option 3: TRAIN
  → Confirm with Y
  → Model improves for next week
```

---

## ⚠️ Common Mistakes to Avoid

### ❌ Wrong: Correcting without predicting first
```
Day 1312: Only CORRECT (no PREDICT first)
→ Error: "No prediction found for day 1312"
```
**✅ Right:** Always PREDICT first, then CORRECT later

### ❌ Wrong: Student IDs with spaces
```
Enter student IDs: 5, 12, 22, 28, 35, 42
→ Error: Invalid format
```
**✅ Right:** No spaces between commas
```
Enter student IDs: 5,12,22,28,35,42
```

### ❌ Wrong: Excel file is open
```
💾 ⚠️  Cannot update Database.xlsx - file is open in Excel
```
**✅ Right:** Close Excel before correcting

### ❌ Wrong: Only 5 students entered
```
Enter student IDs: 5,12,22,28,35
→ Error: "Expected 6 students, got 5"
```
**✅ Right:** Always enter exactly 6 students
```
Enter student IDs: 5,12,22,28,35,42
```

---

## 💡 Pro Tips

### Tip 1: Keep a Daily Log
Write down which group was closest each day to see patterns

### Tip 2: Train Regularly
Weekly training = better predictions

### Tip 3: Check Status Weekly
Monitor your accuracy trends

### Tip 4: Close Excel
Always close Database.xlsx before correcting

### Tip 5: Use Consistent Format
Always: `student1,student2,student3,student4,student5,student6`

---

## 🎯 Quick Reference Card

| Step | When | Action | Input Example |
|------|------|--------|---------------|
| 1 | Morning | PREDICT (Option 1) | Day: `1312` |
| 2 | After class | CORRECT (Option 2) | Day: `1312`<br>Students: `5,12,22,28,35,42` |
| 3 | Anytime | STATUS (Option 4) | (no input) |
| 4 | Weekly | TRAIN (Option 3) | Confirm: `Y` |
| 5 | When done | EXIT (Option 5) | (no input) |

---

## 📞 Need Help?

**Problem:** Can't update database
**Solution:** Close Excel and try again

**Problem:** No prediction found
**Solution:** Run PREDICT first before CORRECT

**Problem:** Invalid student IDs
**Solution:** Check IDs are between 1-55

**Problem:** Wrong number of students
**Solution:** Enter exactly 6 student IDs

---

**Ready to start?** Double-click `student_predictor.bat` now! 🚀

---

## 📚 More Documentation

- [QUICK_START.md](QUICK_START.md) - Quick start guide
- [BATCH_FILE_GUIDE.md](BATCH_FILE_GUIDE.md) - Complete reference
- [PREDICTION_API_GUIDE.md](PREDICTION_API_GUIDE.md) - Technical details
