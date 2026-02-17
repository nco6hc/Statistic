# Monte Carlo Student Selection Model

This directory contains a Monte Carlo simulation-based approach for student selection prediction as an alternative to the LSTM deep learning model.

## Model Overview

The Monte Carlo model uses probabilistic methods instead of neural networks:

1. **Historical Probability Analysis**: Calculates how often each student has been selected
2. **Recency Weighting**: Recent selections weighted more heavily (exponential decay)
3. **Co-occurrence Patterns**: Tracks which students tend to be selected together
4. **Monte Carlo Simulation**: Runs thousands of simulations to find most likely combinations
5. **Online Learning**: Updates probabilities after each prediction

## Files

- **monte_carlo_model.py**: Main model implementation and evaluation pipeline
- **compare_models.py**: Comparison tool for LSTM vs Monte Carlo results
- **README.md**: This file

## How to Run

### 1. Run Monte Carlo Evaluation

```powershell
cd monte_carlo_pipeline
python monte_carlo_model.py
```

This will:
- Train on 90% of the data (same split as LSTM)
- Evaluate on 10% using online learning
- Save results to `mc_results/` directory

### 2. Compare with LSTM

```powershell
python compare_models.py
```

This will:
- Load both LSTM and Monte Carlo results
- Create comparison tables and visualizations
- Generate analysis showing which model performs better

## Model Details

### Probability Calculation

Each student's probability is calculated as:
```
P(student) = Σ (weight_day * selected_day) / Σ (weight_day)
where weight_day = 0.95^(days_ago)
```

### Co-occurrence Matrix

Tracks joint selection patterns:
```
C[i,j] = probability that student j is selected given student i is selected
```

### Prediction Algorithm

1. Run 5,000 Monte Carlo simulations
2. Each simulation:
   - Sample first student based on base probabilities
   - Boost probabilities of co-occurring students
   - Continue until 6 students selected
3. Return top 5 most common combinations

### Temperature Parameter

Controls prediction diversity:
- **Low temperature (< 1.0)**: More deterministic, favors high-probability students
- **High temperature (> 1.0)**: More random, increases diversity
- **Default**: 1.2 for good balance

## Expected Performance

Based on the same 90/10 split used for LSTM:
- **Training data**: ~1,177 days
- **Test data**: ~131 days  
- **Predictions**: 50 (every 2 days with online learning)

Performance will be compared against LSTM's ~26% average accuracy.

## Advantages vs LSTM

✅ **Simpler**: No neural network training required  
✅ **Faster**: Training and prediction in seconds vs minutes  
✅ **Interpretable**: Clear probability distributions  
✅ **No overfitting**: Rule-based approach  
✅ **Fewer dependencies**: Only NumPy and Pandas

## Disadvantages vs LSTM

❌ **Limited pattern learning**: Can't learn complex temporal patterns  
❌ **Linear assumptions**: Assumes independent student probabilities  
❌ **No sequence modeling**: Doesn't use 14-day historical sequences  
❌ **Fixed strategy**: Can't adapt to non-linear patterns

## Results Location

After running the model:
- `mc_results/predictions_log.json`: Detailed day-by-day predictions
- `mc_results/summary.json`: Summary statistics
- `mc_results/model_comparison.png`: Visualization comparing both models
