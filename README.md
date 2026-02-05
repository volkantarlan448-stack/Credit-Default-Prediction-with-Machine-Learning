# Credit-Default-Prediction-with-Machine-Learning



# Project Overview
This project focuses on predicting **credit default risk** using machine learning techniques on real-world financial data.  
The main objective is to identify high-risk customers and support **data-driven decision-making** in credit lending processes.

---

 Business Problem
Financial institutions need accurate models to:
- Minimize loan default risk
- Improve credit approval decisions
- Balance risk and profitability

This project aims to predict whether a customer is likely to **default on a loan** based on demographic and financial features.

---

 Dataset
The dataset contains customer-level financial and demographic information, including:
- Age
- Income
- Interest Rate
- Credit Score
- Loan Amount & Term
- Employment Duration
- Debt-to-Income Ratio (DTI)

Target variable:
- **Loan Default (0 = No Default, 1 = Default)**

---

Exploratory Data Analysis (EDA)
- Analyzed feature distributions and correlations
- Investigated default vs non-default customer profiles
- Detected missing values and outliers
- Visualized feature importance and class imbalance

---

 Data Preprocessing
- Handled missing values and outliers
- Encoded categorical variables
- Applied feature scaling
- Addressed class imbalance
- Split data into train and test sets

---
 Modeling
The following models were implemented and compared:
- Logistic Regression
- Random Forest
- XGBoost

---

 Model Evaluation
Models were evaluated using:
- ROC-AUC
- Precision
- Recall
- F1-Score

Additionally, **decision threshold tuning** was performed to optimize the Precision–Recall trade-off based on business needs.

---

 Key Insights
- **Interest Rate, Income, Age, and Credit Score** are the most influential features
- Random Forest provided strong performance and interpretability
- Threshold optimization significantly improved business-relevant predictions
- Certain customer segments contribute disproportionately to default risk

---

 Visualizations
- Precision–Recall vs Threshold analysis
- Feature importance ranking using Random Forest

---

 Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost

---

 Conclusion
This project demonstrates an end-to-end machine learning pipeline for **credit risk modeling**, combining technical performance with business interpretability.  
The results highlight how machine learning can effectively support financial decision-making processes.

---

**Volkan Tarlan**  
Management Information Systems (MIS) Student  
Aspiring Data Scientist / Data Analyst
