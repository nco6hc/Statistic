# Quick Start Guide

## For New Machine Setup

### Step 1: Verify Python Installation
```bash
python --version
# Should be 3.10 or higher
```

### Step 2: Install Required Packages
```bash
# Navigate to project directory
cd C:\Workspace\Develop\Statistic

# Install all dependencies
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python verify_installation.py
```

Expected output:
```
✓ All packages installed successfully!
```

### Step 4: Run Models

#### LSTM Model (Deep Learning)
```bash
cd dl_pipeline
python main_pipeline.py
```
⏱️ Takes ~3-5 minutes  
📊 Accuracy: ~26%

#### Monte Carlo Model (Probability-based)
```bash
cd monte_carlo_pipeline
python monte_carlo_model.py
```
⏱️ Takes ~30 seconds  
📊 Accuracy: ~23.48%

#### Compare Both Models
```bash
cd monte_carlo_pipeline
python quick_comparison.py
```

## Troubleshooting

### Issue: "No module named 'torch'"
**Solution:**
```bash
pip install torch
```

### Issue: "Excel file not found"
**Solution:**
Ensure `Database.xlsx` is in the root directory:
```
Statistic/
├── Database.xlsx  ← Must be here
├── dl_pipeline/
└── monte_carlo_pipeline/
```

### Issue: "CUDA not available"
**Solution:**
This is normal if you don't have an NVIDIA GPU. The models will run on CPU (slower but functional).

### Issue: Slow LSTM training
**Solution:**
- Reduce number of epochs in `main_pipeline.py` (line ~200)
- Use smaller batch size
- Or just wait - it's normal on CPU

## Package Details

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.0.0 | Reading Excel data |
| numpy | ≥1.24.0 | Numerical operations |
| openpyxl | ≥3.1.0 | Excel file format |
| torch | ≥2.0.0 | LSTM neural network |
| scikit-learn | ≥1.3.0 | Data preprocessing |
| matplotlib | ≥3.7.0 | Plotting comparisons |

## Next Steps

1. ✅ Verify installation
2. ✅ Run both models
3. ✅ Compare results
4. 📈 Analyze predictions in `dl_checkpoints/` and `mc_results/`
5. 🔧 Experiment with model parameters

## Support

Check [README.md](README.md) for detailed documentation.
