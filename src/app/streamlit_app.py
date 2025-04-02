import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from typing import Dict, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.explainability import ModelExplainer
from src.utils.preprocessing import DataPreprocessor
import shap
import lime
import lime.lime_tabular

# Initialize SHAP JS - this is crucial for visualizations
shap.initjs()

# Set page config
st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FraudDetectionApp:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = None
        self.feature_names = []
        self.X_test = pd.DataFrame()
        self.y_test = pd.Series()
        self.explainer = ModelExplainer()
        self.load_models()
        self.load_data()
        
    def load_models(self) -> None:
        """Safely load trained models with error handling"""
        model_files = {
            'XGBoost': 'xgboost_model.joblib',
            'Random Forest': 'random_forest_model.joblib',
            'Logistic Regression': 'logistic_regression_model.joblib',
            'Isolation Forest': 'isolation_forest_model.joblib'
        }
        
        for model_name, filename in model_files.items():
            try:
                self.models[model_name] = joblib.load(f'models/{filename}')
            except Exception as e:
                st.error(f"Failed to load {model_name}: {str(e)}")
        
        try:
            self.scaler = joblib.load('models/scaler.joblib')
            self.feature_names = joblib.load('data/processed/feature_names.joblib')
        except Exception as e:
            st.error(f"Failed to load preprocessing artifacts: {str(e)}")
            st.stop()
            
    def load_data(self) -> None:
        """Load and validate processed data"""
        try:
            _, self.X_test, _, self.y_test = joblib.load('data/processed/processed_data.joblib')
            if len(self.X_test) == 0 or len(self.y_test) == 0:
                raise ValueError("Loaded empty test data")
        except Exception as e:
            st.error(f"Data loading failed: {str(e)}")
            st.stop()
            
    def show_overview(self) -> None:
        """Display project overview and model performance"""
        st.title("Financial Fraud Detection System")
        st.markdown("""
        This application detects fraudulent financial transactions using machine learning.
        It supports both single transaction analysis and batch processing.
        """)
        
        # Data summary
        st.subheader("Data Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", len(self.X_test))
        fraud_count = int(sum(self.y_test))  # Ensure integer for display
        col2.metric("Fraudulent Transactions", fraud_count)
        col3.metric("Fraud Rate", f"{fraud_count/len(self.y_test)*100:.2f}%")
        
        # Model performance
        st.subheader("Model Performance")
        try:
            metrics = joblib.load('data/processed/model_metrics.joblib')
            
            # Display metrics without confusion matrix
            st.dataframe(
                metrics.drop(columns=['confusion_matrix'])
                .style.highlight_max(axis=0, props='font-weight:bold;background-color:yellow')
            )
            
            # Confusion matrices with heatmaps
            st.subheader("Confusion Matrices")
            for model_name in metrics.index:
                with st.expander(f"{model_name} Details"):
                    cm = metrics.loc[model_name, 'confusion_matrix']
                    
                    # Table view
                    st.write(pd.DataFrame(
                        cm,
                        columns=['Predicted Legitimate', 'Predicted Fraud'],
                        index=['Actual Legitimate', 'Actual Fraud']
                    ))
                    
                    # Heatmap visualization
                    fig, ax = plt.subplots(figsize=(6,4))
                    sns.heatmap(
                        cm,
                        annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Legitimate', 'Fraud'],
                        yticklabels=['Legitimate', 'Fraud']
                    )
                    st.pyplot(fig)
                    plt.close()
                    
        except Exception as e:
            st.warning(f"Could not load model metrics: {str(e)}")
            
    def _get_prediction_results(self, scaled_input: np.ndarray) -> pd.DataFrame:
        """Generate predictions for all models"""
        results = []
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'Isolation Forest':
                    pred = model.predict(scaled_input)
                    proba = float(-model.decision_function(scaled_input)[0])  # Convert to scalar
                    pred_label = "Fraud" if pred[0] == -1 else "Legitimate"
                else:
                    proba = float(model.predict_proba(scaled_input)[0][1])  # Convert to scalar
                    pred_label = "Fraud" if proba > 0.5 else "Legitimate"
                    
                results.append({
                    "Model": model_name,
                    "Prediction": pred_label,
                    "Probability/Score": f"{proba:.4f}",
                    "Confidence": f"{max(proba, 1-proba)*100:.1f}%"
                })
            except Exception as e:
                st.error(f"Prediction failed for {model_name}: {str(e)}")
                continue
                
        return pd.DataFrame(results)
            
    def single_transaction_analysis(self) -> None:
        """Analyze a single transaction with robust error handling"""
        st.header("Single Transaction Analysis")
        
        # Initialize session state variables
        if 'form_submitted' not in st.session_state:
            st.session_state.form_submitted = False
        if 'current_transaction' not in st.session_state:
            self._get_new_random_transaction()
        
        # Check if we need to get a new random transaction
        if 'new_random_requested' in st.session_state and st.session_state.new_random_requested:
            self._get_new_random_transaction()
            st.session_state.new_random_requested = False
        
        with st.form("transaction_form"):
            st.subheader("Enter Transaction Details")
            
            inputs = {}
            cols = st.columns(3)
            
            # Create input fields using the current transaction as default
            for i, feature in enumerate(self.feature_names):
                default_val = float(st.session_state.current_transaction.get(feature, 0.0))
                inputs[feature] = cols[i % 3].number_input(
                    feature,
                    value=default_val,
                    format="%.6f" if "V" in feature else "%.2f"
                )
                
            # Form buttons
            col1, col2, col3 = st.columns(3)
            submitted = col1.form_submit_button("Analyze Transaction")
            new_random = col3.form_submit_button("Get New Random Transaction")
            
            if submitted:
                st.session_state.form_submitted = True
                st.session_state.current_inputs = inputs
            
            if new_random:
                st.session_state.new_random_requested = True
                st.rerun()
        
        # Only show results if the form was submitted (either now or previously)
        if st.session_state.get('form_submitted', False):
            try:
                # Use the stored inputs
                current_inputs = st.session_state.current_inputs
                
                # Prepare and scale input
                input_df = pd.DataFrame([current_inputs])
                scaled_input = self.scaler.transform(input_df)
                
                # Get and display predictions
                st.subheader("Prediction Results")
                results = self._get_prediction_results(scaled_input)
                if not results.empty:
                    st.dataframe(results)
                    
                    # Store the scaled input in session state
                    st.session_state.scaled_input = scaled_input
                    st.session_state.input_df = input_df
                    
                    # Model explanation section
                    self._show_model_explanations(scaled_input, input_df)
                else:
                    st.warning("No predictions could be generated")
                    
            except Exception as e:
                st.error(f"Transaction analysis failed: {str(e)}")

    def _get_new_random_transaction(self) -> None:
        """Select a new random transaction and update session state"""
        random_idx = np.random.choice(self.X_test.index)
        st.session_state.current_transaction = self.X_test.loc[random_idx].to_dict()
        st.session_state.current_transaction_class = self.y_test.loc[random_idx]
        st.session_state.form_submitted = False
        if 'current_inputs' in st.session_state:
            del st.session_state.current_inputs

    def _show_model_explanations(self, scaled_input: np.ndarray, input_df: pd.DataFrame) -> None:
        """Display model explanation tabs with robust error handling"""
        st.subheader("Model Explanations")
        
        # Get the selected model from session state if it exists, otherwise default to first model
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = list(self.models.keys())[0]
        
        # Update the selected model based on user input
        model_choice = st.selectbox(
            "Select model to explain",
            list(self.models.keys()),
            key='model_selector',
            index=list(self.models.keys()).index(st.session_state.selected_model)
        )
        
        # Update session state with the new selection
        st.session_state.selected_model = model_choice
        
        tab1, tab2, tab3 = st.tabs(["SHAP", "LIME", "Feature Importance"])
        
        with tab1:
            try:
                # Initialize SHAP for the current model
                model = self.models[model_choice]

                if model_choice == 'XGBoost':
                    # Special handling for XGBoost
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(input_df)
                    
                    # XGBoost returns a different format - we need to handle the array properly
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # Take the fraud class values
                    
                    st.subheader("Local Explanation")
                    plt.figure()
                    shap.force_plot(
                        explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
                        shap_values[0],
                        input_df.iloc[0],
                        feature_names=self.feature_names,
                        matplotlib=True,
                        show=False
                    )
                    st.pyplot(plt.gcf(), bbox_inches='tight')
                    plt.close()
                    
                    st.subheader("Global Feature Importance")
                    plt.figure()
                    shap.summary_plot(
                        shap_values,
                        input_df,
                        feature_names=self.feature_names,
                        show=False
                    )
                    st.pyplot(plt.gcf(), bbox_inches='tight')
                    plt.close()
                
                elif model_choice == 'Isolation Forest':
                    st.info("""
                    **Isolation Forest Note**: Global feature importance isn't available for unsupervised models. 
                    Below shows how each feature contributed to this specific anomaly score.
                    """)
                    
                    # Create explainer for Isolation Forest
                    def predict_fn(x):
                        return -model.decision_function(x)
                    
                    explainer = shap.KernelExplainer(
                        predict_fn,
                        shap.sample(self.X_test, 100)
                    )
                    
                    # Calculate SHAP values
                    shap_values = explainer.shap_values(input_df)
                    
                    # Force plot
                    st.subheader("Local Explanation")
                    plt.figure()
                    shap.force_plot(
                        explainer.expected_value,
                        shap_values[0],
                        input_df.iloc[0],
                        matplotlib=True,
                        show=False
                    )
                    st.pyplot(plt.gcf(), bbox_inches='tight')
                    plt.close()
                    
                elif model_choice == 'Random Forest':
                    try:
                        model = self.models[model_choice]
                        
                        # Try the standard SHAP approach first
                        try:
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(input_df)
                            
                            # For binary classification, get values for positive class (fraud)
                            if isinstance(shap_values, list):
                                shap_values = shap_values[1]  # Index 1 for fraud class
                                expected_value = explainer.expected_value[1]
                            else:
                                expected_value = explainer.expected_value
                            
                            # Create simple bar chart of SHAP values
                            st.subheader("Feature Contributions")
                            importance_df = pd.DataFrame({
                                'Feature': self.feature_names,
                                'SHAP Value': shap_values[0]  # First (only) prediction
                            }).sort_values('SHAP Value', key=abs, ascending=False)
                            
                            # Plot top contributing features
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors = ['red' if x < 0 else 'green' for x in importance_df['SHAP Value'].head(10)]
                            ax.barh(importance_df['Feature'].head(10)[::-1], 
                                    importance_df['SHAP Value'].head(10)[::-1],
                                    color=colors[::-1])
                            ax.set_title('Top Feature Contributions (SHAP Values)')
                            ax.set_xlabel('Impact on Fraud Probability')
                            ax.axvline(0, color='black', linestyle='--')
                            st.pyplot(fig)
                            plt.close()
                            
                        except Exception as e:
                            # Fallback to feature importances if SHAP fails
                            st.info("""
                                **SHAP Explanation Note**: 
                                SHAP explanations are currently not available for Random Forest models 
                                due to technical limitations in the visualization library. 
                                Please use the LIME or Feature Importance tabs for model explanations.
                                """)
                    
                    except Exception as e:
                        st.error(f"SHAP explanation failed: {str(e)}")
                        st.info("Please try the LIME explanations instead.")
                    
                elif model_choice == 'Logistic Regression':
                    # Use LinearExplainer for linear models
                    explainer = shap.LinearExplainer(model, self.X_test)
                    shap_values = explainer.shap_values(input_df)
                    
                    # Force plot
                    st.subheader("Local Explanation")
                    plt.figure()
                    shap.force_plot(
                        explainer.expected_value,
                        shap_values[0],
                        input_df.iloc[0],
                        feature_names=self.feature_names,
                        matplotlib=True,
                        show=False
                    )
                    st.pyplot(plt.gcf(), bbox_inches='tight')
                    plt.close()
                    
                    # Summary plot
                    st.subheader("Global Feature Importance")
                    plt.figure()
                    shap.summary_plot(
                        shap_values,
                        input_df,
                        feature_names=self.feature_names,
                        plot_type="bar",
                        show=False
                    )
                    st.pyplot(plt.gcf(), bbox_inches='tight')
                    plt.close()
                    
            except Exception as e:
                st.warning(f"SHAP explanation failed: {str(e)}")
                st.write("Using alternative SHAP visualization method...")
                self._show_alternative_shap(model_choice, input_df)
                
        with tab2:
            try:
                # Always generate LIME explanation on the fly for the current transaction
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=self.X_test.values,
                    feature_names=self.feature_names,
                    class_names=['Legitimate', 'Fraud'],
                    mode='classification',
                    random_state=42
                )
                
                if model_choice == 'Isolation Forest':
                    def predict_fn(x):
                        preds = self.models[model_choice].predict(x)
                        return np.vstack([(preds == 1).astype(int), (preds == -1).astype(int)]).T
                else:
                    def predict_fn(x):
                        return self.models[model_choice].predict_proba(x)
                
                # Generate explanation for the current input
                exp = explainer.explain_instance(
                    input_df.values[0], 
                    predict_fn,
                    num_features=10
                )
                
                # Show the visualization
                fig = exp.as_pyplot_figure()
                plt.title(f'LIME Explanation for Current Transaction - {model_choice}')
                st.pyplot(fig)
                plt.close()
                    
            except Exception as e:
                st.warning(f"LIME explanation failed: {str(e)}")
                
        with tab3:
            try:
                if model_choice == 'Isolation Forest':
                    # Isolation Forest doesn't have feature importances
                    st.info("Feature importance not available for Isolation Forest")
                else:
                    st.image(f'models/{model_choice.lower().replace(" ", "_")}_feature_importance.png')
            except Exception as e:
                st.warning(f"Feature importance not available: {str(e)}")

    def _show_alternative_shap(self, model_choice: str, input_df: pd.DataFrame) -> None:
        """Simpler fallback visualization when SHAP fails"""
        try:
            model = self.models[model_choice]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            
            # Handle binary classification case
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use fraud class values
            
            # Create simple bar chart of absolute SHAP values
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': np.abs(shap_values).mean(0)
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance_df['Feature'].head(15), 
                    importance_df['Importance'].head(15),
                    color='skyblue')
            ax.set_title('Feature Importance (Absolute SHAP Values)')
            ax.set_xlabel('Mean Absolute SHAP Value')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.error(f"SHAP visualization failed: {str(e)}")
            st.info("""
            SHAP explanations could not be generated for this model/transaction.
            Please try the LIME explanations instead.
            """)
                
    def batch_analysis(self) -> None:
        """Process batch transactions with comprehensive validation"""
        st.header("Batch Analysis")
        uploaded_file = st.file_uploader(
            "Upload CSV file with transactions",
            type=["csv"]
        )
        
        if uploaded_file:
            try:
                # Load the raw data
                batch_df = pd.read_csv(uploaded_file)
                
                # Check for minimum required columns (original features before engineering)
                required_original_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)] + ['Class']
                missing_original_cols = set(required_original_cols) - set(batch_df.columns)
                
                if missing_original_cols:
                    st.error(f"Missing required original columns: {list(missing_original_cols)}")
                    return
                    
                # Perform the same feature engineering as in preprocessing
                batch_df = self._engineer_features(batch_df)
                
                # Now check if we have all the expected features
                missing_cols = set(self.feature_names) - set(batch_df.columns)
                
                if missing_cols:
                    st.error(f"After feature engineering, still missing columns: {list(missing_cols)}")
                    return
                    
                # Preprocess and predict
                preprocessor = DataPreprocessor()
                batch_processed = preprocessor.basic_preprocessing(batch_df)
                batch_scaled = self.scaler.transform(batch_processed[self.feature_names])
                
                model_choice = st.selectbox(
                    "Select prediction model",
                    list(self.models.keys())
                )
                
                batch_df = self._generate_batch_predictions(
                    batch_df, batch_scaled, model_choice
                )
                
                # Display and export results
                self._display_batch_results(batch_df)
                
            except Exception as e:
                st.error(f"Batch processing failed: {str(e)}")
                st.exception(e)  # This will show the full traceback for debugging

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform the same feature engineering as in preprocessing"""
        df = df.copy()
        
        # Time features
        if 'Time' in df.columns:
            df['Time_hour'] = df['Time'] % (24 * 3600) // 3600
            df['Time_day'] = df['Time'] // (24 * 3600)
        
        # Amount features
        if 'Amount' in df.columns:
            df['Amount_log'] = np.log1p(df['Amount'])
        
        return df
                
    def _generate_batch_predictions(self, 
                                batch_df: pd.DataFrame, 
                                batch_scaled: np.ndarray,
                                model_name: str) -> pd.DataFrame:
        """Generate predictions for batch data"""
        model = self.models[model_name]
        
        try:
            if model_name == 'Isolation Forest':
                preds = model.predict(batch_scaled)
                scores = -model.decision_function(batch_scaled)
                batch_df['Prediction'] = np.where(preds == -1, 'Fraud', 'Legitimate')
                batch_df['Anomaly_Score'] = scores
            else:
                probs = model.predict_proba(batch_scaled)[:, 1]
                batch_df['Prediction'] = np.where(probs > 0.5, 'Fraud', 'Legitimate')
                batch_df['Fraud_Probability'] = probs
            
            # Add confidence score
            batch_df['Confidence'] = batch_df.apply(
                lambda row: f"{max(row['Fraud_Probability'], 1-row['Fraud_Probability'])*100:.1f}%" 
                if 'Fraud_Probability' in row else "N/A",
                axis=1
            )
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            batch_df['Prediction'] = "Error"
            batch_df['Error'] = str(e)
            
        return batch_df
        
    def _display_batch_results(self, batch_df: pd.DataFrame) -> None:
        """Display and export batch results"""
        st.subheader("Batch Prediction Results")
        
        # Show summary stats first
        if 'Prediction' in batch_df.columns:
            fraud_count = int(sum(batch_df['Prediction'] == 'Fraud'))
            total_count = len(batch_df)
            st.metric("Predicted Fraudulent Transactions", 
                    f"{fraud_count} ({fraud_count/total_count*100:.1f}%)")
        
        # Display the dataframe with important columns first
        display_cols = ['Prediction']
        if 'Fraud_Probability' in batch_df.columns:
            display_cols.append('Fraud_Probability')
        if 'Confidence' in batch_df.columns:
            display_cols.append('Confidence')
        if 'Anomaly_Score' in batch_df.columns:
            display_cols.append('Anomaly_Score')
        
        # Add all other columns except prediction-related ones
        other_cols = [col for col in batch_df.columns if col not in display_cols]
        display_cols.extend(other_cols)
        
        st.dataframe(batch_df[display_cols])
        
        # Download button
        csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Predictions",
            csv,
            "fraud_predictions.csv",
            "text/csv",
            key='download-csv'
        )
        
    def data_exploration(self) -> None:
        """Interactive data visualization"""
        st.header("Data Exploration")
        
        try:
            X_train, X_test, y_train, y_test = joblib.load('data/processed/processed_data.joblib')
            X_test['Class'] = y_test
            
            # Feature distribution analysis
            self._show_feature_distributions(X_test)
            
            # Correlation analysis
            self._show_correlation_heatmap(X_test)
            
        except Exception as e:
            st.error(f"Data exploration failed: {str(e)}")
            
    def _show_feature_distributions(self, data: pd.DataFrame) -> None:
        """Show feature distributions by class"""
        st.subheader("Feature Distributions")
        feature = st.selectbox("Select feature", self.feature_names)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(data[feature], kde=True, ax=ax[0])
        ax[0].set_title(f"Distribution of {feature}")
        
        sns.boxplot(x='Class', y=feature, data=data, ax=ax[1])
        ax[1].set_title(f"{feature} by Class")
        st.pyplot(fig)
        plt.close()
        
    def _show_correlation_heatmap(self, data: pd.DataFrame) -> None:
        """Show feature correlation matrix"""
        st.subheader("Correlation Heatmap")
        numeric_cols = data.select_dtypes(include=np.number).columns
        corr = data[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        st.pyplot(plt)
        plt.close()

def st_shap(plot, height: Optional[int] = None) -> None:
    """Render SHAP plots in Streamlit"""
    import streamlit.components.v1 as components
    # Get both JS and HTML parts
    shap_html = f"""
    <head>
        {shap.getjs()}
    </head>
    <body>
        {plot.html()}
    </body>
    """
    components.html(shap_html, height=height, scrolling=True)

def main() -> None:
    app = FraudDetectionApp()
    
    # Navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Go to",
        ["Overview", "Single Transaction", "Batch Analysis", "Data Exploration"]
    )
    
    # Page routing
    page_map = {
        "Overview": app.show_overview,
        "Single Transaction": app.single_transaction_analysis,
        "Batch Analysis": app.batch_analysis,
        "Data Exploration": app.data_exploration
    }
    
    try:
        page_map[app_mode]()
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()