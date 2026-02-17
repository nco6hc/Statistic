# Student Selection Prediction System

A dual-approach machine learning system for predicting student selections using both LSTM neural networks and Monte Carlo simulation.

## Requirements

- Python 3.10 or higher
- Windows/Linux/macOS

## Installation

### 1. Clone or Download the Project

```bash
cd C:\Workspace\Develop\Statistic
```

### 2. Install Dependencies

#### Option A: Using pip

```bash
pip install -r requirements.txt
```

#### Option B: Install packages individually

```bash
# Core packages
pip install pandas numpy openpyxl

# Deep Learning (for LSTM)
pip install torch scikit-learn

# Visualization (for model comparison)
pip install matplotlib
```

### 3. Verify Installation

```bash
python -c "import torch; import pandas; import numpy; print('All packages installed successfully!')"
```

## Project Structure

```
Statistic/
├── Database.xlsx                    # Historical student selection data (1,308 days)
├── requirements.txt                 # Python package dependencies
├── README.md                        # This file
│
├── dl_pipeline/                     # LSTM Deep Learning Pipeline
│   ├── main_pipeline.py            # Main training & evaluation script
│   ├── models.py                   # LSTM model architecture
│   ├── trainer.py                  # Training logic
│   ├── data_loader.py              # Data loading utilities
│   └── dl_checkpoints/             # Saved models and training logs
│       ├── best_model.pth
│       ├── final_online_model.pth
│       └── online_predictions_log.csv
│
└── monte_carlo_pipeline/            # Monte Carlo Simulation Pipeline
    ├── monte_carlo_model.py        # Monte Carlo implementation
    ├── compare_models.py           # LSTM vs Monte Carlo comparison
    ├── quick_comparison.py         # Quick results comparison
    ├── README.md                   # Monte Carlo documentation
    └── mc_results/                 # Monte Carlo results
        ├── predictions_log.json
        └── summary.json
```

## Usage

### Running LSTM Model

```bash
cd dl_pipeline
python main_pipeline.py
```

**Output:**
- Trains LSTM model with 90/10 train/test split
- Runs 100-day online learning simulation
- Saves model checkpoints to `dl_checkpoints/`
- Average accuracy: ~26%

### Running Monte Carlo Model

```bash
cd monte_carlo_pipeline
python monte_carlo_model.py
```

**Output:**
- Trains Monte Carlo probability model
- Runs online learning evaluation
- Saves results to `mc_results/`
- Average accuracy: ~23.48%

### Comparing Both Models

```bash
cd monte_carlo_pipeline
python quick_comparison.py
```

**Output:**
- Side-by-side performance comparison
- Statistical analysis
- Strengths/weaknesses of each approach

## Model Performance

| Metric | LSTM | Monte Carlo |
|--------|------|-------------|
| Average Accuracy | 26.00% | 23.48% |
| Best Accuracy | 66.67% | 50.00% |
| Consistency (Std) | 12.54 | 11.60 |
| Training Time | ~5 minutes | ~30 seconds |
| Interpretability | Low | High |

**Baseline:** Random selection = 9.1% accuracy (6/55 students)

## Dataset

- **File:** `Database.xlsx`
- **Size:** 1,308 days of historical data
- **Format:** Each row = 1 day, 6 student IDs selected (0-54)
- **Students:** 55 total (IDs 0-54)
- **Selections per day:** 6 students

## Models

### 1. LSTM Neural Network
- **Architecture:** 2-layer bidirectional LSTM (128 hidden units)
- **Input:** 14-day historical sequence
- **Parameters:** 300,599 trainable parameters
- **Training:** 30 epochs with early stopping
- **Approach:** Learns complex temporal patterns

### 2. Monte Carlo Simulation
- **Approach:** Probability-based sampling
- **Features:**
  - Historical selection frequencies
  - Recency weighting (exponential decay)
  - Co-occurrence pattern analysis
- **Simulations:** 5,000 per prediction
- **Advantages:** Fast, interpretable, consistent

## Common Issues

### 1. Excel File Not Found
```
Error: FileNotFoundError: Database.xlsx
Solution: Ensure Database.xlsx is in the root directory
```

### 2. CUDA/GPU Issues
```
Error: CUDA not available
Solution: LSTM will automatically use CPU (slower but works)
```

### 3. Import Errors
```
Error: ModuleNotFoundError: No module named 'torch'
Solution: pip install torch
```

## System Requirements

### Minimum:
- CPU: Dual-core 2.0 GHz
- RAM: 4 GB
- Storage: 500 MB

### Recommended:
- CPU: Quad-core 3.0 GHz or GPU
- RAM: 8 GB
- Storage: 1 GB

## License

Educational/Research Project

## Authors

Student Selection Prediction System - 2026
