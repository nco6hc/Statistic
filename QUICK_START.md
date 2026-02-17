# 🚀 Quick Start Guide

## Welcome to the Student Selection Prediction System!

This system helps you predict and track which 6 students will be selected each day.

---

## ⚡ Getting Started (3 Simple Steps)

### 1. Run the Program
Double-click: **`student_predictor.bat`**

### 2. Make Your First Prediction
- Choose option `1` (PREDICT)
- Enter day number (e.g., `1312`)
- System generates 5 groups of students

### 3. Submit Results After Class
- Choose option `2` (CORRECT)
- Enter the day number
- Enter the 6 students that were actually selected (e.g., `5,12,22,28,35,42`)
- Database automatically updates ✅

---

## 📋 Daily Workflow

### Morning (Before Class)
```
Run: student_predictor.bat
Choose: 1 (PREDICT)
Enter day: 1312
```
**Output:** 5 groups of 6 students each

### After Class
```
Run: student_predictor.bat
Choose: 2 (CORRECT)
Enter day: 1312
Enter students: 5,12,22,28,35,42
```
**Result:** System shows accuracy & updates database

### Weekly (Weekend)
```
Run: student_predictor.bat
Choose: 3 (TRAIN)
Confirm: Y
```
**Result:** Model improves from collected data

---

## 💡 Tips

✅ **Always PREDICT before CORRECT** - Make prediction first, correct later

✅ **Close Excel** - If database update fails, close Excel and try again

✅ **Train weekly** - Better performance with regular training

✅ **Use exact format** - Student IDs: `5,12,22,28,35,42` (commas, no spaces)

---

## 📁 What Gets Created

```
predictions/
├── prediction_day_1312.json     ← Your prediction
├── correction_day_1312.json     ← Actual results  
└── predictions_summary.csv      ← All results

Database.xlsx                     ← Auto-updated with corrections ✅
```

---

## 🐛 Common Issues

**"No prediction found for day X"**
→ Run PREDICT first before CORRECT

**"Permission denied: Database.xlsx"**
→ Close Excel and try again

**"Expected 6 students"**
→ Enter exactly 6 student IDs

**"Student IDs must be between 1 and 55"**
→ Check all IDs are valid

---

## 🎯 Example Session

```bash
# Monday morning
Option: 1 (PREDICT)
Day: 1312
→ Gets 5 groups

# Monday after class
Option: 2 (CORRECT)  
Day: 1312
Students: 5,12,22,28,35,42
→ Shows accuracy, updates database

# Repeat daily...

# Friday
Option: 4 (STATUS)
→ Check weekly performance

# Weekend
Option: 3 (TRAIN)
→ Improve model
```

---

## 📖 Full Documentation

- [BATCH_FILE_GUIDE.md](BATCH_FILE_GUIDE.md) - Complete guide
- [PREDICTION_API_GUIDE.md](PREDICTION_API_GUIDE.md) - API reference
- [SEPARATED_API_SUMMARY.md](SEPARATED_API_SUMMARY.md) - Technical details

---

## ✨ Features

✅ **5 prediction groups per day** - Multiple options to choose from

✅ **Automatic database updates** - No manual Excel editing

✅ **Accuracy tracking** - See how well predictions perform

✅ **Weekly model training** - Continuous improvement

✅ **Easy batch interface** - No command-line knowledge needed

---

**Ready to start?** Double-click `student_predictor.bat` now! 🎉

---

## 📞 Need Help?

1. Check [BATCH_FILE_GUIDE.md](BATCH_FILE_GUIDE.md) for detailed instructions
2. Run option `4` (STATUS) to see current state
3. Make sure Excel is closed when correcting

**Happy Predicting! 🎯**
