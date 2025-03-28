#src/main.py

import argparse
import joblib
from utils.data_loader import DataLoader
from utils.preprocessing import DataPreprocessor
from models.train import FraudDetectionModel
from utils.explainability import ModelExplainer

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Financial Fraud Detection System")
    parser.add_argument('--download', action='store_true', help='Download dataset from Kaggle')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the data')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--explain', action='store_true', help='Generate model explanations')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    args = parser.parse_args()
    
    if args.download or args.all:
        print("=== Downloading data ===")
        loader = DataLoader()
        df = loader.load_creditcard_data()
        
    if args.preprocess or args.all:
        print("\n=== Preprocessing data ===")
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.full_pipeline(
            df,
            target_col='Class',
            sampling_strategy='smote',
            scaler_type='robust'
        )
        
        # Save feature names for later use
        joblib.dump(list(X_train.columns), 'data/processed/feature_names.joblib')
        
    if args.train or args.all:
        print("\n=== Training models ===")
        model = FraudDetectionModel()
        
        # Define parameter grids for grid search
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2']
            },
            'xgboost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1]
            }
        }
        
        # Train and evaluate each model
        for model_name in ['random_forest', 'logistic_regression', 'xgboost', 'isolation_forest']:
            print(f"\nTraining {model_name}...")
            if model_name in param_grids:
                trained_model = model.train_model(
                    model_name,
                    X_train,
                    y_train,
                    param_grid=param_grids[model_name]
                )
            else:
                trained_model = model.train_model(
                    model_name,
                    X_train,
                    y_train
                )
                
            # Evaluate model
            metrics = model.evaluate_model(
                trained_model,
                X_test,
                y_test,
                model_name
            )
            print(f"Evaluation metrics for {model_name}:")
            print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"PR-AUC: {metrics['pr_auc']:.4f}")
            
        # Compare all models
        comparison_df = model.compare_models()
        print("\nModel Comparison:")
        print(comparison_df)
        
        # Save metrics
        joblib.dump(comparison_df, 'data/processed/model_metrics.joblib')
        
    if args.explain or args.all:
        print("\n=== Generating model explanations ===")
        explainer = ModelExplainer()
        
        # Load processed data
        X_train, X_test, y_train, y_test = joblib.load('data/processed/processed_data.joblib')
        feature_names = joblib.load('data/processed/feature_names.joblib')
        
        for model_name in ['random_forest', 'logistic_regression', 'xgboost', 'isolation_forest']:
            print(f"\nExplaining {model_name}...")
            model = joblib.load(f'models/{model_name}_model.joblib')
            
            # SHAP analysis
            print("Running SHAP analysis...")
            shap_values = explainer.shap_analysis(
                model,
                X_train,
                X_test,
                model_name
            )
            
            # LIME analysis
            print("Running LIME analysis...")
            explainer.lime_analysis(
                model,
                X_train.values,
                X_test.values,
                feature_names,
                ['Legitimate', 'Fraud'],
                model_name
            )
            
            # Feature importance
            print("Generating feature importance...")
            explainer.global_feature_importance(
                model,
                feature_names,
                model_name
            )
            
    print("\n=== All steps completed ===")

if __name__ == "__main__":
    main()