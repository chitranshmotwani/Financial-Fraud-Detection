#src/models/train.py

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple

class FraudDetectionModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            'random_forest': RandomForestClassifier(random_state=random_state),
            'logistic_regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'xgboost': XGBClassifier(random_state=random_state, eval_metric='logloss'),
            'isolation_forest': IsolationForest(random_state=random_state, contamination='auto')
        }
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}
        
    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Dict[str, Any] = None,
        cv: int = 5
    ) -> Any:
        """
        Train a specified model with optional hyperparameter tuning
        
        Args:
            model_name (str): Name of the model to train
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            param_grid (dict): Hyperparameter grid for tuning
            cv (int): Number of cross-validation folds
            
        Returns:
            Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.models.keys())}")
            
        model = self.models[model_name]
        
        # Handle Isolation Forest separately (unsupervised)
        if model_name == 'isolation_forest':
            print(f"Training {model_name}...")
            model.fit(X_train)
            self.best_model = model
            self.best_model_name = model_name
            return model
            
        # For supervised models
        if param_grid:
            print(f"Performing grid search for {model_name}...")
            cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best ROC-AUC: {grid_search.best_score_:.4f}")
            
            self.best_model = grid_search.best_estimator_
            self.best_model_name = model_name
            return grid_search.best_estimator_
        else:
            print(f"Training {model_name} with default parameters...")
            model.fit(X_train, y_train)
            self.best_model = model
            self.best_model_name = model_name
            return model
            
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = None,
        compute_shap: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test set
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            model_name (str): Name of the model
            compute_shap (bool): Whether to compute SHAP values
                
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if model_name is None:
            model_name = self.best_model_name
            
        # Handle Isolation Forest separately
        if model_name == 'isolation_forest':
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred == 1, 0, 1)
            y_scores = -model.decision_function(X_test)
        else:
            y_pred = model.predict(X_test)
            y_scores = model.predict_proba(X_test)[:, 1]
            
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_scores)
        pr_auc = average_precision_score(y_test, y_scores)
        
        # Store metrics
        self.metrics[model_name] = {
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score'],
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm.tolist()
        }
        
        # Generate and save SHAP values if requested and not Isolation Forest
        if compute_shap and model_name != 'isolation_forest':
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                
                # For binary classification, get values for positive class
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]
                
                # Create proper Explanation object
                explanation = shap.Explanation(
                    values=shap_values,
                    base_values=explainer.expected_value,
                    data=X_test.values,
                    feature_names=X_test.columns.tolist()
                )
                
                # Save the explanation object
                joblib.dump(explanation, f'models/{model_name}_shap.joblib')
                
            except Exception as e:
                print(f"Could not generate SHAP values: {str(e)}")
        
        # Plotting functions
        self.plot_confusion_matrix(cm, model_name)
        self.plot_roc_pr_curves(y_test, y_scores, model_name)
        self.save_model(model, model_name)
        
        return self.metrics[model_name]
        
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'models/{model_name}_confusion_matrix.png')
        plt.close()
        
    def plot_roc_pr_curves(self, y_true: pd.Series, y_scores: np.ndarray, model_name: str):
        """Plot and save ROC and PR curves"""
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color='blue', lw=2, label='PR curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        
        plt.tight_layout()
        plt.savefig(f'models/{model_name}_curves.png')
        plt.close()
        
    def save_model(self, model: Any, model_name: str):
        """Save trained model to disk"""
        joblib.dump(model, f'models/{model_name}_model.joblib')
        print(f"Model saved to models/{model_name}_model.joblib")
        
    def compare_models(self) -> pd.DataFrame:
        """Compare performance of all trained models"""
        return pd.DataFrame(self.metrics).T