# Dataset

This folder contains the PIMA Indians Diabetes Database used for model training and evaluation.

## File

- **diabetes.csv** - 768 observations with 8 diagnostic features and binary diabetes outcome

## Source

National Institute of Diabetes and Digestive and Kidney Diseases  

## Features

| Column | Description |
|--------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (µU/mL) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age (years) |
| Outcome | 0 = Non-diabetic, 1 = Diabetic |

## Data Characteristics

- **Samples:** 768
- **Features:** 8
- **Target:** Binary (0/1)
- **Class Distribution:** 500 non-diabetic (65%), 268 diabetic (35%)
- **Missing Values:** Encoded as zeros in some columns (see main analysis)
