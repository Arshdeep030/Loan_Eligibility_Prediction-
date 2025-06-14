# Loan Eligibility Prediction Project 

## Introduction
Loan disbursement is a core operation in the banking and finance sector. However, determining whether a loan applicant will successfully repay their loan remains a challenging task. This project focuses on building a **predictive machine learning model** to assess **loan eligibility**—whether a loan applicant is likely to **repay** or **default** on a loan.

## Problem Statement
Dream Housing Finance is a company that provides loans across urban, semi-urban, and rural sectors. The company wants to automate its validation process by predicting customer eligibility for a loan based on information like income, age, employment length, credit history, etc.

The objective is to help the company:
- Identify eligible customers.
- Reduce the default rate.
- Improve targeting and customer satisfaction.


## Features Description

| Feature | Description |
|--------|-------------|
| `person_age` | Age of the individual. |
| `person_income` | Annual income of the applicant. |
| `person_home_ownership` | Type of home ownership: `rent`, `mortgage`, `own`, `other`. |
| `person_emp_length` | Employment length in years. |
| `loan_intent` | Reason for taking the loan. |
| `loan_grade` | Creditworthiness score (A to G). |
| `loan_amnt` | Requested loan amount. |
| `loan_int_rate` | Loan interest rate. |
| `loan_status` | Target variable: `0` = Not Defaulted, `1` = Defaulted. |
| `loan_percent_income` | Loan amount as a percentage of income. |
| `cb_person_default_on_file` | Credit bureau default history: `Y` or `N`. |
| `cb_person_cred_hist_length` | Length of credit history (in years). |


## Preprocessing Steps

- **One Hot Encoding**: Applied to categorical features with more than two categories:
  - `loan_grade`, `person_home_ownership`, `loan_intent`
  
- **Ordinal Encoding**: Applied to binary categorical column:
  - `cb_person_default_on_file`
  
- **Feature Scaling**: Standardization applied to numerical columns.

- **SMOTE**: Synthetic Minority Over-sampling Technique used to handle class imbalance.


## Machine Learning Models Used

| Model | Train Accuracy | Test Accuracy | Comments |
|-------|----------------|---------------|----------|
| **Logistic Regression (Before SMOTE)** | 0.848 | 0.846 | Lower recall on defaulters |
| **Logistic Regression (After SMOTE)** | 0.804 | 0.832 | Improved recall for class 1 |
| **Decision Tree (Before SMOTE)** | 1.000 | 0.887 | Overfitting detected |
| **Decision Tree (After SMOTE)** | 1.000 | 0.890 | Slight improvement |
| **Random Forest (Before SMOTE)** | 1.000 | 0.934 | Strong performance |
| **Random Forest (After SMOTE)** | 1.000 | 0.935 | Balanced recall and precision |
| **Random Forest (Hyperparameter Tuned)** | 0.983 | 0.933 | Best balance, reduced overfitting |
| **XGBoost (Before SMOTE)** | 0.957 | 0.932 | High performance |
| **XGBoost (After SMOTE)** | 0.962 | 0.931 | Comparable performance |
| **SVM (Before SMOTE)** | 0.697 | 0.707 | Poor class 1 prediction |
| **SVM (After SMOTE)** | 0.649 | 0.782 | Class 1 still not captured well |

## Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**


## Tech Stack

- Python
- Jupyter Notebook
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost
