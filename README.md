# 🚀 Financial Fraud Detection Using Machine Learning

Financial fraud is a growing concern, affecting businesses and consumers worldwide. This project explores machine learning techniques to develop a robust fraud detection model capable of identifying fraudulent transactions with high accuracy.

---

## 📌 Project Overview  
This project leverages supervised and unsupervised learning techniques to detect fraudulent financial transactions. The goal is to develop a model that balances high precision with recall to minimize false positives and false negatives.

### **Key Features**  
✅ Exploratory Data Analysis (EDA) to understand transaction patterns  
✅ Handling imbalanced data using resampling techniques (SMOTE, undersampling)  
✅ Training multiple ML models (Random Forest, XGBoost, Logistic Regression, Isolation Forest)  
✅ Model evaluation using AUC-ROC, Precision-Recall, and Confusion Matrix  
✅ Explainability tools like SHAP and LIME to interpret model decisions  
✅ Interactive dashboard for fraud detection using Streamlit
✅ Basic deployment with a CLI tool for fraud detection

---

## 🛠️ Technologies Used  
- **Programming Language**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost, SHAP, LIME, Streamlit  
- **Data Sources**:  
  - Kaggle Credit Card Fraud Dataset  
  - PaySim Synthetic Mobile Transactions  

---

## 📂 Project Structure  
```
fraud-detection-ml/
│── data/                 # Raw and preprocessed datasets  
│── notebooks/            # Jupyter notebooks for EDA and model development  
│── src/                  # Source code for data processing and model training  
│   ├── preprocess.py     # Data preprocessing functions  
│   ├── train.py          # Model training and evaluation
│   ├── app.py            # Streamlit app for interactive fraud detection   
│── models/               # Saved trained models  
│── reports/              # Analysis and findings  
│── README.md             # Project documentation  
│── requirements.txt      # Dependencies  
```

---

## 🚀 Installation & Setup  
### **Clone the Repository**  
```sh
git clone https://github.com/chitranshmotwani/Financial-Fraud-Detection.git
cd Financial-Fraud-Detection
```

### **Create a Virtual Environment (Optional but Recommended)**  
```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### **Install Dependencies**  
```sh
pip install -r requirements.txt
```

---

## 📊 Usage  
### **1️⃣ Data Preprocessing**  
Run the preprocessing script to clean and prepare the dataset:  
```sh
python src/preprocess.py
```

### **2️⃣ Train the Model**  
Train the fraud detection model using different algorithms:  
```sh
python src/train.py
```

### **3️⃣ Run the Streamlit App**  
Launch the Streamlit app for an interactive interface:  
```sh
streamlit run src/app.py
```

### **4️⃣ Predict Fraud on New Transactions**  
Use the trained model to classify transactions:  
```sh
python src/predict.py --input new_transactions.csv
```

---

## 📈 Model Performance  
The trained model is evaluated using:  
- Precision, Recall, and F1-score  
- AUC-ROC Curve  
- Confusion Matrix  

Detailed performance reports can be found in the `reports/` directory.

---

## 🎯 Future Improvements  
🔹 Fine-tune models for better accuracy  
🔹 Experiment with deep learning techniques  
🔹 Deploy as a Flask API for real-time fraud detection  
🔹 Enhance data visualization and dashboarding  

---

## 🤝 Contributing  
Contributions are welcome! Feel free to submit pull requests or open issues for suggestions.

---

## 📜 License  
This project is licensed under the **MIT License**.

---

## 🔗 Connect with Me  
💼 [LinkedIn](https://www.linkedin.com/in/chitranshmotwani)  
📧 Email: [cma115@sfu.ca](mailto:cma115@sfu.ca)  

---
