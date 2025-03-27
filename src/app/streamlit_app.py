#src/app/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.explainability import ModelExplainer
from src.utils.preprocessing import DataPreprocessor
import shap
import lime
import lime.lime_tabular

# Set page config
st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

class FraudDetectionApp:
    def __init__(self):
        self.load_models()
        self.load_data()
        self.explainer = ModelExplainer()
        
    def load_models(self):
        """Load trained models"""
        try:
            self.models = {
                'Random Forest': joblib.load('models/random_forest_model.joblib'),
                'Logistic Regression': joblib.load('models/logistic_regression_model.joblib'),
                'XGBoost': joblib.load('models/xgboost_model.joblib'),
                'Isolation Forest': joblib.load('models/isolation_forest_model.joblib')
            }
            self.scaler = joblib.load('models/scaler.joblib')
            self.feature_names = joblib.load('data/processed/feature_names.joblib')
        except FileNotFoundError as e:
            st.error(f"Model files not found. Please train models first. Error: {e}")
            st.stop()
            
    def load_data(self):
        """Load sample data"""
        try:
            _, self.X_test, _, self.y_test = joblib.load('data/processed/processed_data.joblib')
        except FileNotFoundError:
            st.error("Processed data not found. Please run preprocessing first.")
            st.stop()
            
    def show_overview(self):
        """Show project overview"""
        st.title("Financial Fraud Detection System")
        st.markdown("""
        This application detects fraudulent financial transactions using machine learning.
        It supports both single transaction analysis and batch processing.
        """)
        
        # Show data summary
        st.subheader("Data Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", len(self.X_test))
        col2.metric("Fraudulent Transactions", sum(self.y_test))
        col3.metric("Fraud Rate", f"{sum(self.y_test)/len(self.y_test)*100:.2f}%")
        
        # Show model performance
        st.subheader("Model Performance")
        try:
            metrics = joblib.load('data/processed/model_metrics.joblib')
            st.dataframe(metrics.style.highlight_max(axis=0))
        except FileNotFoundError:
            st.warning("Model metrics not available. Please train models first.")
            
    def single_transaction_analysis(self):
        """Analyze a single transaction"""
        st.header("Single Transaction Analysis")
        
        # Create form for manual input
        with st.form("transaction_form"):
            st.subheader("Enter Transaction Details")
            
            # Create input fields for each feature
            inputs = {}
            cols = st.columns(3)
            for i, feature in enumerate(self.feature_names):
                col = cols[i % 3]
                inputs[feature] = col.number_input(
                    feature,
                    value=float(self.X_test.iloc[0][feature]),
                    format="%.6f" if "V" in feature else "%.2f"
                )
                
            submitted = st.form_submit_button("Analyze Transaction")
            
        if submitted:
            # Prepare input data
            input_df = pd.DataFrame([inputs])
            scaled_input = self.scaler.transform(input_df)
            
            # Make predictions with all models
            st.subheader("Prediction Results")
            results = []
            
            for model_name, model in self.models.items():
                if model_name == 'Isolation Forest':
                    pred = model.predict(scaled_input)
                    proba = -model.decision_function(scaled_input)  # anomaly score
                    pred_label = "Fraud" if pred[0] == -1 else "Legitimate"
                else:
                    proba = model.predict_proba(scaled_input)[0][1]
                    pred_label = "Fraud" if proba > 0.5 else "Legitimate"
                    
                results.append({
                    "Model": model_name,
                    "Prediction": pred_label,
                    "Probability/Score": f"{proba:.4f}",
                    "Confidence": f"{max(proba, 1-proba)*100:.1f}%"
                })
                
            # Show results table
            st.table(pd.DataFrame(results))
            
            # Show explainability
            st.subheader("Model Explanations")
            model_choice = st.selectbox(
                "Select model to explain",
                list(self.models.keys())
            )
            
            tab1, tab2, tab3 = st.tabs(["SHAP", "LIME", "Feature Importance"])
            
            with tab1:
                try:
                    shap_values = joblib.load(f'models/{model_choice.lower().replace(" ", "_")}_shap_values.joblib')
                    st_shap(shap.plots.force(shap_values[0]))
                except FileNotFoundError:
                    st.warning("SHAP values not available for this model.")
                    
            with tab2:
                try:
                    st.image(f'models/{model_choice.lower().replace(" ", "_")}_lime_instance_0.png')
                except FileNotFoundError:
                    st.warning("LIME explanation not available for this model.")
                    
            with tab3:
                try:
                    st.image(f'models/{model_choice.lower().replace(" ", "_")}_feature_importance.png')
                except FileNotFoundError:
                    st.warning("Feature importance not available for this model.")
                    
    def batch_analysis(self):
        """Analyze a batch of transactions"""
        st.header("Batch Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with transactions",
            type=["csv"]
        )
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                batch_df = pd.read_csv(uploaded_file)
                
                # Check if required columns are present
                missing_cols = set(self.feature_names) - set(batch_df.columns)
                if missing_cols:
                    st.error(f"Missing columns in uploaded file: {missing_cols}")
                    return
                    
                # Preprocess and scale
                preprocessor = DataPreprocessor()
                batch_processed = preprocessor.basic_preprocessing(batch_df)
                batch_scaled = self.scaler.transform(batch_processed[self.feature_names])
                
                # Select model
                model_choice = st.selectbox(
                    "Select model for prediction",
                    list(self.models.keys())
                )
                model = self.models[model_choice]
                
                # Make predictions
                if model_choice == 'Isolation Forest':
                    predictions = model.predict(batch_scaled)
                    scores = -model.decision_function(batch_scaled)
                    batch_df['Prediction'] = np.where(predictions == -1, 'Fraud', 'Legitimate')
                    batch_df['Anomaly_Score'] = scores
                else:
                    probabilities = model.predict_proba(batch_scaled)[:, 1]
                    batch_df['Prediction'] = np.where(probabilities > 0.5, 'Fraud', 'Legitimate')
                    batch_df['Fraud_Probability'] = probabilities
                    
                # Show results
                st.subheader("Batch Prediction Results")
                st.dataframe(batch_df)
                
                # Show summary stats
                fraud_count = sum(batch_df['Prediction'] == 'Fraud')
                st.metric("Predicted Fraudulent Transactions", fraud_count)
                
                # Download results
                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Predictions",
                    csv,
                    "fraud_predictions.csv",
                    "text/csv",
                    key='download-csv'
                )
                
            except Exception as e:
                st.error(f"Error processing file: {e}")
                
    def data_exploration(self):
        """Show data exploration visualizations"""
        st.header("Data Exploration")
        
        # Load full processed data
        try:
            X_train, X_test, y_train, y_test = joblib.load('data/processed/processed_data.joblib')
            X_test['Class'] = y_test
        except FileNotFoundError:
            st.error("Processed data not found.")
            return
            
        # Show feature distributions
        st.subheader("Feature Distributions")
        feature_choice = st.selectbox(
            "Select feature to visualize",
            self.feature_names
        )
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(
            X_test[feature_choice],
            kde=True,
            ax=ax[0]
        )
        ax[0].set_title(f"Distribution of {feature_choice}")
        
        sns.boxplot(
            x='Class',
            y=feature_choice,
            data=X_test,
            ax=ax[1]
        )
        ax[1].set_title(f"{feature_choice} by Class")
        st.pyplot(fig)
        
        # Show correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_cols = X_test.select_dtypes(include=np.number).columns
        corr = X_test[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        st.pyplot(plt)
        
def st_shap(plot, height=None):
    """Display SHAP plot in Streamlit"""
    import streamlit.components.v1 as components
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def main():
    app = FraudDetectionApp()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Go to",
        ["Overview", "Single Transaction", "Batch Analysis", "Data Exploration"]
    )
    
    # Show selected page
    if app_mode == "Overview":
        app.show_overview()
    elif app_mode == "Single Transaction":
        app.single_transaction_analysis()
    elif app_mode == "Batch Analysis":
        app.batch_analysis()
    elif app_mode == "Data Exploration":
        app.data_exploration()
        
if __name__ == "__main__":
    main()