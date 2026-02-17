# 🧹 Cleanup Recommendations

## Files That Can Be Removed

Your workspace contains old files from previous implementations that are now **superseded by the new deep learning pipeline**.

---

## 📁 Unnecessary Files (Safe to Remove)

### 1. **Old Traditional ML Scripts** (Superseded by Deep Learning)
- ❌ `predict_students.py` - Old prediction with XGBoost/RF
- ❌ `train_models.py` - Old training script
- ❌ `markov_chain_model.py` - Old Markov Chain approach
- ❌ `process_student_data.py` - Old data processing
- ❌ `feature_engineering.py` - Old 4019-feature approach
- ❌ `model_summary.py` - Old results viewer

**Why remove?** The new `prediction_api.py` + `dl_pipeline/` replaces all of these.

### 2. **Old Model Files** (Outdated)
- ❌ `trained_model_xgboost.pkl`
- ❌ `trained_model_random_forest.pkl`
- ❌ `trained_model_logistic_regression.pkl`
- ❌ `trained_model_neural_network.pth` (old, not the LSTM)
- ❌ `trained_model_scaler.pkl`
- ❌ `markov_model.pkl`
- ❌ `model_comparison.csv`

**Why remove?** Using new LSTM model in `dl_pipeline/dl_checkpoints/`

### 3. **Old Data Files** (Superseded)
- ❌ `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`
- ❌ `X_train_features.npy`, `X_test_features.npy`
- ❌ `Y_train_labels.npy`, `Y_test_labels.npy`
- ❌ `feature_names.txt`

**Why remove?** Old feature-engineered data, not used by LSTM pipeline.

### 4. **Test/Demo Scripts** (Not Core)
- ❌ `test_multi_groups.py` - Test file
- ❌ `demo_multigroup.py` - Demo script
- ❌ `show_multigroup_summary.py` - Demo viewer
- ❌ `example_workflow.py` - Example (documentation is better)
- ❌ `test_randomness.py` - Analysis script
- ❌ `randomness_test_results.csv`

**Why remove?** Only needed during development, not for production.

### 5. **Demo Checkpoints**
- ❌ `demo_checkpoints/` - Demo files from testing

**Why remove?** Just demo data, keep `dl_pipeline/dl_checkpoints/` instead.

### 6. **Old Prediction Outputs**
- ❌ `predictions_xgboost.csv`
- ❌ `predictions_random_forest.csv`
- ❌ `predictions_logistic_regression.csv`
- ❌ `predictions_neural_network.csv`

**Why remove?** Old outputs from superseded models.

---

## ✅ Essential Files (KEEP)

### Core Application
- ✅ `prediction_api.py` - Main API
- ✅ `dl_pipeline/` - Complete deep learning pipeline
  - `prediction_service.py`
  - `trainer.py`
  - `models.py`
  - `data_loader.py`
  - `online_pipeline.py`
  - `main_pipeline.py`
  - `dl_checkpoints/` - Trained LSTM models
- ✅ `Database.xlsx` - Your data
- ✅ `visualizations.py` - Visualization tool

### Documentation
- ✅ All `*.md` files (README, guides)
- ✅ `PREDICTION_API_GUIDE.md`
- ✅ `MULTIGROUP_UPDATE.md`
- ✅ `SEPARATED_API_SUMMARY.md`

### Active Data
- ✅ `predictions/` - Your predictions and corrections
- ✅ `visualizations/` - Generated charts

---

## 🚀 Quick Cleanup

### Option 1: Automated Cleanup (Recommended)

```bash
python cleanup.py
```

This will:
1. Show you what can be removed
2. Calculate disk space to free
3. Let you choose: Archive or Delete

### Option 2: Manual Cleanup

**Windows PowerShell:**
```powershell
# Create archive folder
mkdir _old_files_archive

# Move old files
mv predict_students.py _old_files_archive/
mv train_models.py _old_files_archive/
mv markov_chain_model.py _old_files_archive/
mv process_student_data.py _old_files_archive/
mv feature_engineering.py _old_files_archive/
mv model_summary.py _old_files_archive/
mv test_randomness.py _old_files_archive/

# Move old model files
mv *.pkl _old_files_archive/ 2>$null
mv trained_model_*.pth _old_files_archive/ 2>$null

# Move old data files
mv X_*.npy _old_files_archive/ 2>$null
mv Y_*.npy _old_files_archive/ 2>$null
mv y_*.npy _old_files_archive/ 2>$null

# Move test/demo files
mv test_multi_groups.py _old_files_archive/ 2>$null
mv demo_multigroup.py _old_files_archive/ 2>$null
mv show_multigroup_summary.py _old_files_archive/ 2>$null
mv example_workflow.py _old_files_archive/ 2>$null

# Move demo checkpoints
mv demo_checkpoints _old_files_archive/ 2>$null

# Move old prediction outputs
mv predictions_*.csv _old_files_archive/ 2>$null
```

Then later, if you don't need the archive:
```powershell
rm -r _old_files_archive
```

---

## 📊 After Cleanup

Your clean workspace will contain:

```
Statistic/
├── prediction_api.py          ← Your main API
├── Database.xlsx              ← Your data
├── visualizations.py          ← Visualization tool
├── cleanup.py                 ← This cleanup script
│
├── dl_pipeline/               ← Deep learning pipeline
│   ├── __init__.py
│   ├── data_loader.py
│   ├── models.py
│   ├── trainer.py
│   ├── online_pipeline.py
│   ├── prediction_service.py
│   ├── main_pipeline.py
│   ├── README.md
│   └── dl_checkpoints/        ← Trained models
│
├── predictions/               ← Your predictions/corrections
│   ├── prediction_day_*.json
│   ├── correction_day_*.json
│   └── predictions_summary.csv
│
├── visualizations/            ← Generated charts
│   ├── frequency_heatmap.png
│   ├── coselection_heatmap.png
│   └── probability_ranking.png
│
└── *.md                       ← Documentation
```

**Clean, organized, and production-ready!** 🎉

---

## ⚠️ Important Notes

1. **Keep `Database.xlsx`** - This is your original data
2. **Keep `dl_pipeline/dl_checkpoints/`** - These are your trained models
3. **Keep `predictions/`** - These are your active predictions
4. **Archive first, delete later** - Safer approach

---

## 💡 Recommendation

Run the automated cleanup script:

```bash
python cleanup.py
```

Choose **Option 1 (Archive)** to safely move old files to `_old_files_archive/`.

After a few days, if you don't need them, delete the archive folder.

---

**Ready to clean up?** Run `python cleanup.py` now!
