#src/utils/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import joblib
from typing import Tuple, Union

class DataPreprocessor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = None
        self.preprocessor = None
        
    def basic_preprocessing(self, df: pd.DataFrame, target_col: str = 'Class') -> pd.DataFrame:
        """
        Perform basic preprocessing:
        - Handle missing values
        - Convert data types if needed
        - Basic feature engineering
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Name of the target column
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Handle missing values if any
        if df.isnull().sum().sum() > 0:
            print("Handling missing values...")
            # For numerical columns, fill with median
            num_cols = df.select_dtypes(include=np.number).columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
            
            # For categorical columns, fill with mode
            cat_cols = df.select_dtypes(exclude=np.number).columns
            for col in cat_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Basic feature engineering for credit card data
        if 'Time' in df.columns:
            # Convert Time to hours of the day
            df['Time_hour'] = df['Time'] % (24 * 3600) // 3600
            df['Time_day'] = df['Time'] // (24 * 3600)
            
        if 'Amount' in df.columns:
            # Log transform of Amount to handle skewness
            df['Amount_log'] = np.log1p(df['Amount'])
            
        return df
        
    def train_test_split_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'Class',
        test_size: float = 0.2,
        sampling_strategy: Union[str, dict] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets with optional resampling
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Name of the target column
            test_size (float): Proportion for test split
            sampling_strategy (str/dict): Resampling strategy if needed
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Split into train and test first to avoid data leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        # Apply resampling if specified
        if sampling_strategy:
            print(f"Applying {sampling_strategy} resampling...")
            if sampling_strategy == 'smote':
                sampler = SMOTE(random_state=self.random_state)
            elif sampling_strategy == 'adasyn':
                sampler = ADASYN(random_state=self.random_state)
            elif sampling_strategy == 'undersample':
                sampler = RandomUnderSampler(random_state=self.random_state)
            else:
                raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
                
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            
        return X_train, X_test, y_train, y_test
        
    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        scaler_type: str = 'standard'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using specified scaler
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            scaler_type (str): Type of scaler ('standard' or 'robust')
            
        Returns:
            Tuple: Scaled X_train and X_test
        """
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
            
        # Fit on training data only to avoid data leakage
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(
            X_train_scaled, 
            columns=X_train.columns, 
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled, 
            columns=X_test.columns, 
            index=X_test.index
        )
        
        # Save the scaler for later use
        joblib.dump(self.scaler, 'models/scaler.joblib')
        
        return X_train_scaled, X_test_scaled
        
    def full_pipeline(
        self,
        df: pd.DataFrame,
        target_col: str = 'Class',
        test_size: float = 0.2,
        sampling_strategy: str = None,
        scaler_type: str = 'standard'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Complete preprocessing pipeline:
        1. Basic preprocessing
        2. Train-test split
        3. Feature scaling
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            test_size (float): Test set proportion
            sampling_strategy (str): Resampling strategy
            scaler_type (str): Type of feature scaler
            
        Returns:
            Tuple: Processed X_train, X_test, y_train, y_test
        """
        # Step 1: Basic preprocessing
        df_processed = self.basic_preprocessing(df, target_col)
        
        # Step 2: Train-test split with optional resampling
        X_train, X_test, y_train, y_test = self.train_test_split_data(
            df_processed, 
            target_col, 
            test_size, 
            sampling_strategy
        )
        
        # Step 3: Feature scaling
        X_train_scaled, X_test_scaled = self.scale_features(
            X_train, 
            X_test, 
            scaler_type
        )
        
        # Save processed data
        joblib.dump(
            (X_train_scaled, X_test_scaled, y_train, y_test),
            'data/processed/processed_data.joblib'
        )
        
        return X_train_scaled, X_test_scaled, y_train, y_test