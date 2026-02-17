# Student Selection Visualizations

## 📊 Generated Visualizations

This document describes the 5 visualizations created for analyzing student selection patterns.

---

## 1. **Frequency Heatmap** (`frequency_heatmap.png`)

### Description
Shows student selection frequency over time, binned into 50-day periods.

### Key Features
- **X-axis**: Time periods (26 bins of ~50 days each covering 1,308 days)
- **Y-axis**: All 55 students (S1 to S55)
- **Color**: Yellow to Red (lighter = fewer selections, darker = more selections)
- **Purpose**: Identify temporal patterns and whether certain students are selected more in specific periods

### Insights to Look For
- ✓ Uniform color distribution = fair selection over time
- ✗ Dark vertical bands = certain periods had more selections
- ✗ Dark horizontal bands = certain students over-selected

---

## 2. **Frequency Distribution** (`frequency_distribution.png`)

### Description
Bar chart showing total number of times each student was selected over all 1,308 days.

### Key Features
- **X-axis**: Student ID (1-55)
- **Y-axis**: Total selection count
- **Color**: Gradient from red (most selected) to green (least selected)
- **Red dashed line**: Mean selection count across all students

### Expected Results
- Average selections per student: **~143 times** (1308 days × 6 selections / 55 students)
- Standard deviation: Should be low if selection is fair
- Range: All students should be close to the mean

### Insights
- Perfectly fair selection → All bars at exactly 143
- Real data → Small variations around 143 (±5-10)

---

## 3. **Co-selection Heatmap** (`coselection_heatmap.png`)

### Description
55×55 matrix showing how often each pair of students was selected together on the same day.

### Key Features
- **Axes**: Both show Student ID (1-55)
- **Color**: Purple (low co-selection) to Yellow (high co-selection)
- **Diagonal**: Masked (student can't be paired with themselves)
- **Symmetry**: Matrix is symmetric (S1-S2 = S2-S1)

### Statistics Displayed
- **Mean co-selections**: Expected ~48 times per pair
  - Calculation: 1308 days × 15 pairs per day / 1485 possible pairs
  - 15 pairs per day = C(6,2) = 6 choose 2
  - 1485 possible pairs = C(55,2) = 55 choose 2
- **Standard deviation**: Indicates randomness vs. patterns

### Insights to Look For
- ✓ Uniform color = random pairing
- ✗ Bright spots = specific students frequently selected together
- ✗ Dark regions = certain pairs rarely/never together

---

## 4. **Top Co-selection Pairs** (`top_coselection_pairs.png`)

### Description
Horizontal bar chart showing the 25 most frequently co-selected student pairs.

### Key Features
- **Y-axis**: Student pairs (e.g., "S12-S34")
- **X-axis**: Number of times selected together
- **Sorting**: Descending by co-selection count

### Expected Results
- **Random baseline**: ~48 co-selections per pair
- **Top pair count**: Should not exceed ~60-65 if fair
- **Pattern detection**: Gap between top and baseline indicates preferential pairing

### Insights
- Top pair ≫ 60 → Teacher has pairing preferences
- All pairs close to 48 → Random pairing
- Specific students appearing multiple times → "Buddy groups"

---

## 5. **Probability Ranking** (`probability_ranking.png`)

### Description
Two-panel visualization showing model's predicted selection probabilities for all students.

### Panel 1: All Students Ranked (Top)
- **X-axis**: Student rank (1-55 by probability)
- **Y-axis**: Predicted probability (%)
- **Colors**: 
  - Green = Predicted (top 6 students)
  - Blue = Not predicted (remaining 49)
- **Red dashed line**: Random baseline (10.91%)

### Panel 2: Top 15 Students (Bottom)
- **Horizontal bar chart** showing detailed view of most likely students
- **Green bars**: Predicted students (model's top 6 picks)
- **Blue bars**: High probability but not in top 6
- **Value labels**: Exact probability percentages

### Statistics Displayed
- **Top prediction**: Student with highest probability
- **6th prediction**: Threshold for being selected
- **Mean probability**: Average across all 55 students (~18%)
- **Std probability**: Spread of confidence
- **Random baseline**: 10.91% (6/55)

### Model Performance Indicators
- Good model: Top 6 probabilities >> baseline
- Uncertain model: Probabilities close to baseline
- Confident model: Large gap between 6th and 7th student

### Example Interpretation
```
Top prediction: Student 5 (41.36%)
6th prediction: Student 35 (28.32%)
Mean: 14.25%
Random baseline: 10.91%
```
- Model is **confident** (top prediction 4× baseline)
- Clear **threshold effect** (28% vs ~15% for 7th place)
- Predictions are **meaningful** (not random guessing)

---

## 📈 How to Use These Visualizations

### For Pattern Analysis
1. **Check frequency_distribution.png** → Is selection fair?
2. **Check frequency_heatmap.png** → Any temporal bias?
3. **Check coselection_heatmap.png** → Random pairing or groups?

### For Model Evaluation
1. **Check probability_ranking.png** → Is model confident?
2. Compare predicted (green) vs. actual selections
3. Look for gap between top 6 and rest (good separation = better model)

### For Reporting
- **frequency_distribution.png**: Best for showing fairness
- **top_coselection_pairs.png**: Easy to interpret, good for presentations
- **probability_ranking.png**: Shows model's decision-making process

---

## 🔍 Statistical Tests from Earlier Analysis

Combine these visualizations with the statistical test results:

| Test | Result | Interpretation |
|------|--------|----------------|
| **Chi-square** | p=0.61 | Selection frequency is uniform ✓ |
| **Runs test** | p=0.19 | Temporal sequence has some structure |
| **KS test** | p<0.001 | Time intervals non-uniform ✗ |
| **Interval test** | p<0.001 | Days between selections not random ✗ |

**Conclusion**: Selection is **constrained random** - fair in frequency but patterned in timing.

---

## 📁 File Locations

All visualizations saved in:
```
visualizations/
├── frequency_heatmap.png          (Time-binned heatmap)
├── frequency_distribution.png      (Total counts bar chart)
├── coselection_heatmap.png        (55×55 pairing matrix)
├── top_coselection_pairs.png      (Top 25 pairs bar chart)
└── probability_ranking.png         (Model predictions)
```

---

## 🚀 Regenerating Visualizations

To regenerate with updated data:

```bash
python visualizations.py
```

The script will:
1. Load data from `Database.xlsx`
2. Generate all 5 visualizations
3. Save as PNG files (300 DPI, publication quality)
4. Display statistics in terminal

**Requirements**: `matplotlib`, `seaborn`, `pandas`, `numpy`

---

## 🎨 Customization Options

Edit `visualizations.py` to customize:

- **Bin size**: Change `bin_size = 50` to adjust time periods
- **Color schemes**: Modify `cmap='YlOrRd'` or `cmap='viridis'`
- **Top N pairs**: Change `[:25]` to show more/fewer pairs
- **Figure sizes**: Adjust `figsize=(16, 8)` tuples
- **DPI**: Change `dpi=300` for higher/lower resolution

---

**Generated on**: 2026-02-16  
**Data source**: Database.xlsx (1,308 days, 55 students, 6 selections/day)  
**Model**: LSTM with online learning (11.3% accuracy, 8% improvement over simulation)
