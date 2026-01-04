# Visualizations

All figures generated during exploratory data analysis, model training, and evaluation.

## Exploratory Data Analysis

### Data Quality
[![Missing Data Pattern](missing_data_pattern.png)](missing_data_pattern.png)
**Missing Data Pattern** - Distribution of zero values representing missing data across features

[![Missing Data Heatmap](missing_data_heatmap.png)](missing_data_heatmap.png)
**Missing Data Heatmap** - Visual pattern of missing values before imputation

[![Imputation Comparison](imputation_comparison.png)](imputation_comparison.png)
**Imputation Comparison** - Feature distributions before and after median imputation

### Target Variable
[![Target Distribution](target_distribution.png)](target_distribution.png)
**Target Distribution** - Class balance: 65% non-diabetic vs 35% diabetic

### Features
[![Feature Distributions](feature_distributions.png)](feature_distributions.png)
**Feature Distributions** - Histogram of all 8 diagnostic features with mean/median

[![Correlation Matrix](correlation_matrix.png)](correlation_matrix.png)
**Correlation Matrix** - Feature correlations and relationships with diabetes outcome

[![Feature Comparison Boxplots](feature_comparison_boxplots.png)](feature_comparison_boxplots.png)
**Feature Comparison** - Diabetic vs non-diabetic feature value distributions

## Model Performance

### Model Comparison
[![Model Comparison Metrics](model_comparison_metrics.png)](model_comparison_metrics.png)
**Model Comparison** - Performance across 6 algorithms using 5-fold cross-validation

### Baseline Model Evaluation
[![Confusion Matrices](confusion_matrices.png)](confusion_matrices.png)
**Confusion Matrices** - Training and test set performance for baseline Logistic Regression

[![ROC Curves](roc_curves.png)](roc_curves.png)
**ROC Curves** - Receiver Operating Characteristic curves for training and test sets

[![Precision-Recall Curves](precision_recall_curves.png)](precision_recall_curves.png)
**Precision-Recall Curves** - Trade-off analysis between precision and recall

[![Feature Importance](feature_importance.png)](feature_importance.png)
**Feature Importance** - Logistic Regression coefficients showing Glucose as dominant predictor

[![Prediction Probability Distribution](prediction_probability_distribution.png)](prediction_probability_distribution.png)
**Prediction Probabilities** - Distribution of model confidence for diabetic and non-diabetic cases

## Hyperparameter Tuning

[![Hyperparameter Impact](hyperparameter_impact.png)](hyperparameter_impact.png)
**Hyperparameter Impact** - Effect of regularization (C) and class weighting on F1-score

[![Baseline vs Tuned Confusion Matrix](baseline_vs_tuned_confusion_matrix.png)](baseline_vs_tuned_confusion_matrix.png)
**Model Comparison** - Baseline vs optimized model showing +39% recall improvement

## Final Results

[![Diabetes Prediction Summary](diabetes_prediction_summary.png)](diabetes_prediction_summary.png)
**Complete Summary** - Comprehensive view of model selection, features, metrics, and final confusion matrix

---

**All visualizations generated using:** matplotlib 3.7+ and seaborn 0.12+
