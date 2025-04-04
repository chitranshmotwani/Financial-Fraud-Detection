# 🚀 Financial Fraud Detection: A Comprehensive Machine Learning Approach with Explainable AI

Financial fraud is a growing concern, affecting businesses and consumers worldwide. This project explores machine learning techniques to develop a robust fraud detection model capable of identifying fraudulent transactions with high accuracy.

---

## 📌 Project Overview

This production-grade fraud detection system combines multiple machine learning approaches with advanced explainability techniques to identify fraudulent financial transactions in real-time. The project delivers:

- **High-performance models** achieving 0.985 AUC-ROC score
- **Comprehensive explainability** through SHAP, LIME, and feature importance analysis
- **Interactive dashboard** for both real-time and batch processing
- **Data valuation** to understand instance-level contributions
- **Full CI/CD-ready** pipeline from raw data to predictions

---

## 🏆 Key Achievements

✅ **State-of-the-art performance**: XGBoost model with 0.872 PR-AUC on highly imbalanced data  
✅ **Multi-model comparison**: 4 distinct algorithms benchmarked systematically  
✅ **Advanced explainability**: Combined SHAP, LIME, and counterfactual explanations  
✅ **Production-ready interface**: Streamlit app with real-time API capabilities  
✅ **Data-centric analysis**: Leave-One-Out and Shapley value data valuation  
✅ **Modular architecture**: Fully reproducible research pipeline  

---

## 🛠️ Technical Stack

### Core Technologies
- **Python 3.8+** (Type-hinted, PEP 8 compliant)
  - **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost, SHAP, LIME, Streamlit  
- **Scikit-learn** (Random Forest, Logistic Regression)
- **XGBoost** (Gradient Boosted Trees)
- **Imbalanced-learn** (SMOTE, ADASYN)
- **SHAP & LIME** (Model explainability)
- **Streamlit** (Interactive web interface)
- - **Data Sources**:  
  - Kaggle Credit Card Fraud Dataset  

---

## 📂 Project Structure  
```
Financial-Fraud-Detection/
financial-fraud-detection/
├── data/ # Data artifacts (raw → processed)
│ ├── raw/ # Original datasets
│ └── processed/ # Cleaned and preprocessed data
├── models/ # Serialized models and metrics
├── notebooks/ # Research notebooks
│ ├── data_exploration.ipynb # Comprehensive data exploration
│ ├── data_valuation.ipynb # Instance importance analysis
│ ├── model_evaluation.ipynb # Model Metrics
│ └── shapley_analysis.ipynb# Explainability research
├── src/ # Production code
│ ├── models/ # Training and evaluation
│ ├── app/ # Streamlit application
│ ├── utils/ # Shared utilities
│ ├── main.py # Main Pipeline Code
├── docs/ # Documentation
│ ├── report.pdf # Comprehensive Report
└── requirements.txt # File listing all requirements
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Kaggle API credentials (for data download)
- Virtual environment (recommended)

### Installation

### Clone repository
```
git clone https://github.com/chitranshmotwani/financial-fraud-detection.git
cd financial-fraud-detection
```

### Create and activate virtual environment
```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### Install dependencies
```
pip install -r requirements.txt
```

### Set up Kaggle API
```
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```
---

## 🚀 Running the Pipeline

### 1. Complete pipeline (data → models)
```
python src/models/main.py --all
```


### 2. Launch Streamlit app
```
streamlit run src/app/streamlit_app.py
```
---

## 📈 Model Performance  
The trained model is evaluated using:  
- Precision, Recall, and F1-score  
- AUC-ROC, PR-AUC Curve  
- Confusion Matrix  

Detailed performance report can be found in the `reports/` directory.

---

## 🔗 Connect with Me  
💼 [LinkedIn](https://www.linkedin.com/in/chitranshmotwani)  
📧 Email: [cma115@sfu.ca](mailto:cma115@sfu.ca)  

---
