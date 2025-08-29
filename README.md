
# Healthcare Survival Analysis

Logistic regression on a synthetic healthcare dataset with age, log10 WBC, and treatment group.

## Files
- `healthcare_data.csv` — synthetic dataset
- `survival_analysis.py` — training & evaluation script

## How to Run
```bash
pip install pandas scikit-learn
python survival_analysis.py
```

## Extensions
- Add interaction terms (age × treatment)
- Try tree-based models and calibration
- Export coefficients and odds ratios
