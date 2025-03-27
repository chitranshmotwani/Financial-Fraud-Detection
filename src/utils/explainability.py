import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from typing import Any

class ModelExplainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.explainer = None
        
    def shap_analysis(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        model_name: str,
        sample_size: int = 100
    ):
        """
        Perform SHAP analysis on the model
        
        Args:
            model: Trained model
            X_train (pd.DataFrame): Training data (for explainer background)
            X_test (pd.DataFrame): Test data to explain
            model_name (str): Name of the model
            sample_size (int): Number of samples to use for explanation
        """
        # Sample data for faster computation
        background = shap.sample(X_train, 100, random_state=self.random_state)
        test_sample = shap.sample(X_test, sample_size, random_state=self.random_state)
        
        # Create explainer based on model type
        if model_name == 'isolation_forest':
            # For Isolation Forest, we explain the anomaly scores
            def model_predict(x):
                return -model.decision_function(x)
                
            self.explainer = shap.KernelExplainer(
                model_predict,
                background
            )
        else:
            # For supervised models, explain class probabilities
            self.explainer = shap.Explainer(
                model,
                background,
                algorithm='auto'
            )
            
        # Calculate SHAP values
        shap_values = self.explainer(test_sample)
        
        # Plot summary
        plt.figure()
        shap.summary_plot(shap_values, test_sample, show=False)
        plt.title(f'SHAP Summary - {model_name}')
        plt.tight_layout()
        plt.savefig(f'models/{model_name}_shap_summary.png')
        plt.close()
        
        # Save SHAP values
        joblib.dump(shap_values, f'models/{model_name}_shap_values.joblib')
        
        return shap_values
        
    def lime_analysis(
        self,
        model: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: list,
        class_names: list,
        model_name: str,
        num_samples: int = 5000,
        num_features: int = 10
    ):
        """
        Perform LIME analysis on the model
        
        Args:
            model: Trained model
            X_train (np.ndarray): Training data (for statistics)
            X_test (np.ndarray): Test data to explain
            feature_names (list): List of feature names
            class_names (list): List of class names
            model_name (str): Name of the model
            num_samples (int): Number of samples for LIME
            num_features (int): Number of features to show
        """
        # Create explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            random_state=self.random_state
        )
        
        # Explain a few instances
        explanations = []
        for i in range(min(5, len(X_test))):  # Explain first 5 instances
            if model_name == 'isolation_forest':
                # For Isolation Forest, we need to map predictions
                def model_predict(x):
                    preds = model.predict(x)
                    # Convert to probability-like output (0 for inlier, 1 for outlier)
                    return np.vstack([(preds == 1).astype(int), (preds == -1).astype(int)]).T
                    
                exp = explainer.explain_instance(
                    X_test[i], 
                    model_predict,
                    num_features=num_features,
                    num_samples=num_samples
                )
            else:
                exp = explainer.explain_instance(
                    X_test[i], 
                    model.predict_proba,
                    num_features=num_features,
                    num_samples=num_samples
                )
                
            explanations.append(exp)
            
            # Save explanation plot
            fig = exp.as_pyplot_figure()
            plt.title(f'LIME Explanation - Instance {i} - {model_name}')
            plt.tight_layout()
            plt.savefig(f'models/{model_name}_lime_instance_{i}.png')
            plt.close()
            
        return explanations
        
    def global_feature_importance(
        self,
        model: Any,
        feature_names: list,
        model_name: str
    ):
        """
        Plot global feature importance for tree-based models
        
        Args:
            model: Trained model
            feature_names (list): List of feature names
            model_name (str): Name of the model
        """
        if hasattr(model, 'feature_importances_'):
            # For models with feature_importances_
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title(f'Feature Importances - {model_name}')
            plt.bar(range(len(importances)), importances[indices], align='center')
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(f'models/{model_name}_feature_importance.png')
            plt.close()
            
        elif hasattr(model, 'coef_'):
            # For linear models
            coef = model.coef_[0]
            indices = np.argsort(np.abs(coef))[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title(f'Feature Coefficients - {model_name}')
            plt.bar(range(len(coef)), coef[indices], align='center')
            plt.xticks(range(len(coef)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(f'models/{model_name}_coefficients.png')
            plt.close()