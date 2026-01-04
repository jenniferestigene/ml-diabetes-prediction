# Analysis Notebook

This folder contains the complete end-to-end machine learning analysis for diabetes prediction.

## Notebook

**[ML_Diabetes_Prediction.ipynb](ML_Diabetes_Prediction.ipynb)** - Complete analysis pipeline

## Contents

### 1. Introduction and Setup
- Project objectives and key questions
- Library imports and configuration
- Reproducibility settings (random seed)

### 2. Data Loading and Initial Exploration
- Dataset overview (768 samples, 9 columns)
- Basic statistics and data types
- Missing value detection

### 3. Exploratory Data Analysis
- Target variable distribution and class imbalance analysis
- Feature distributions with statistical summaries
- Correlation matrix and feature relationships
- Comparison of diabetic vs non-diabetic groups

### 4. Data Preprocessing
- Missing value handling (zeros → NaN conversion)
- Missingness indicator feature creation
- Median imputation strategy
- Stratified train-test split (80-20)
- Feature scaling (StandardScaler)

### 5. Model Development
- Algorithm comparison (6 models)
- 5-fold stratified cross-validation
- Performance metrics tracking
- Model selection based on F1-score

### 6. Model Evaluation
- Confusion matrices (training and test sets)
- Classification reports
- ROC and Precision-Recall curves
- Feature importance analysis
- Prediction probability distributions
- Error analysis (false positives and negatives)

### 7. Hyperparameter Optimization
- GridSearchCV configuration (48 combinations)
- Parameter grid: C, penalty, solver, class_weight
- Best model selection
- Performance comparison (baseline vs tuned)

### 8. Conclusions and Key Findings
- Summary of results
- Key insights and recommendations
- Future improvements

## How to Run

1. Install dependencies:
```bash
   pip install -r ../requirements.txt
```

2. Launch Jupyter:
```bash
   jupyter notebook
```

3. Open `ML_Diabetes_Prediction.ipynb` and run all cells sequentially

## Key Results

- **Best Model:** Logistic Regression
- **Test Accuracy:** 73.4%
- **Test Recall:** 72.2%
- **Test F1-Score:** 65.6%
- **ROC-AUC:** 81.7%

## Technologies

- Python 3.12
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook
