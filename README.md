# 🛡️ NovaPay Fraud Detection System


## 📌 Project Overview

An end-to-end machine learning fraud detection system for cross-border digital money transfers, designed for real-time transaction risk scoring in a production-style environment. The system integrates SHAP-based explainable AI to deliver transparent, transaction-level insights for fraud analysts and regulatory stakeholders, and includes an interactive web app for live fraud detection, model interpretation, and decision support.

---

## 🎯 Project Objectives: Building Intelligence

- **Supervised Classification:** Build robust supervised classifiers capable of accurately     distinguishing fraudulent transactions from legitimate ones accross diverse transaction patterns.
- **Handle Class Imbalance:** Address severe class imbalance through class-weighted ensemble methods.
- **Explainability Integration:** Integrate SHAP to provide human readable explanations 
- **Performance Target:** Achieve at least 15% improvement in recall compared to rules-based baseline while maintaining acceptable precision levels to minimize false positives.
- **Production Deployment:** Package the solution as a web app capable of real-time scoring for immediate transaction evaluation in production environments.


---

## 🏗️ System Architecture

```
Raw Data
   ↓
Data Cleaning & Validation
   ↓
Exploratory Data Analysis (EDA)
   ↓
Feature Engineering
   ↓
Model Training & Evaluation
   ↓
Explainability (SHAP)
   ↓
Deployment (Streamlit AI Dashboard)
```

---

## 📁 Project Structures

```
NovaPay_Fraud_Detection/
│
├── data/                                          # Datasets and preprocessing artifacts
│                         
│   
├── notebooks/
│   ├── 01_Nova_Data_Cleaning.ipynb                # Data cleaning
│   ├── 02_Nova_EDA.ipynb                          # Exploratory Data Analysis 
│   ├── 03_Nova_Feature_Engineering.ipynb          # Feature Engineering
│   ├── 04_Nova_Modelling.ipynb                    # Modelling
│   └── 05_Best_Model_and_SHAP.ipynb               # SHAP integration into the best model
│
│                    
├── model/
│   └── rf_fraud_pipeline.pkl     
│
│
├── .streamlit/
│   └── config.toml                                # Light AI-themed UI configuration
│
│
├── app.py                                         # Streamlit web app
├── requirements.txt                               # Python dependencies
│
│
└── README.md                                      # Project documentation  This file

```

---

## 🧪 Data Preproocessing 

### 1️⃣ Data Cleaning
- Removed duplicate transaction records  
- Fixed inconsistent categories (channels, currencies, KYC tiers)  
- Handled missing values using **domain-aware logic**  
- Corrected invalid numeric values (negative amounts, scores)  

### 2️⃣ Exploratory Data Analysis (EDA)
Key findings:
- Fraud transactions show **higher transaction velocity**
- Lower **device trust scores** strongly correlate with fraud
- **New accounts** are significantly riskier
- Certain **corridors and channels** carry higher fraud risk
- No data leakage detected

---

## 🧬 Feature Engineering

Engineered features include:
- Transaction velocity ratios
- Account maturity indicators
- Device and IP risk interactions
- Time-based features (hour, day, month)
- Corridor and currency risk indicators

This step significantly improved model performance.

---

## 🤖 Model Training & Evaluation

### Models Tested
- Logistic Regression 
- Random Forest
- XGBoost
- KNN
- SVC
- AdaBoost
- Gradient Boosting
- Decision Tree

### Final Model
✅ **Random Forest Classifier**

Chosen for:
- Strong ROC-AUC and PR-AUC
- Robust handling of non-linear patterns
- Best Precision and Recall
- Interpretability with SHAP

---

## 🔍 Explainable AI (SHAP)

To ensure trust and transparency:
- SHAP values were computed for the Random Forest model  
- Global explanations identify **key fraud drivers**
- Local explanations show *why a specific transaction was flagged*

Top drivers:
- Transaction velocity
- Device trust score
- IP risk score
- Account age 
- Amount-Velocity interaction

---

## 🖥️ Deployment – AI-Assisted System

The system is deployed as a **Streamlit web application** with Real-Time fraudulent transaction detection 

 ---

## 🚀 How to Run the App

1. **Clone the repository**

```bash
git clone https://github.com/Habib-A/NovaPay.git
cd NovaPay
```

2.  **Install dependencies**

```bash
pip install -r requirements.txt
```

3.  **Run the Streamlit app**
```bash
https://fraudulent-transaction-detection.streamlit.app/
```

---

## 📈 Results

- Built and deployed Streamlit web application for Real-Time fraudulent transaction detection
- Risk assessment with human-readable explanations
- Delivered high-confidence fraud detection with 100% precision, capturing 81.9% of fraudulent transactions while minimizing false positives.

---

## 🧰 Tech Stack
| Category           | Tools & Libraries                          |
| -----------------  | ------------------------------------------ |
| Language           | Python                                     |
| Data Analysis      | pandas, numpy                              |
| Visualization      | matplotlib, seaborn, plotly                |
| Machine Learning   | scikit-learn (Supervised Classification)   |
| App Framework      | Streamlit                                  |
| Version Control    | Git & GitHub                               |
| Explainability AI  | SHAP                                       |

---


## 👤 Author

**Habib Pelumi Abdullahi**  
Data Scientist | Machine Learning Engineer  
📧 habibpelumiabdullahi@gmail.com

---
