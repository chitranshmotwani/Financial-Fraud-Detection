#src/utils/data_loader.py

import os
import zipfile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm
import joblib

class DataLoader:
    def __init__(self):
        self.api = KaggleApi()
        self.api.authenticate()
        
    def download_dataset(self, dataset_name, save_path='data/raw'):
        """
        Download dataset from Kaggle
        
        Args:
            dataset_name (str): Kaggle dataset name in format 'username/dataset'
            save_path (str): Path to save the downloaded data
        """
        os.makedirs(save_path, exist_ok=True)
        
        print(f"Downloading dataset: {dataset_name}")
        self.api.dataset_download_files(dataset_name, path=save_path, unzip=True)
        print("Download completed!")
        
        # Find the downloaded CSV file
        csv_files = [f for f in os.listdir(save_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the downloaded dataset")
            
        return os.path.join(save_path, csv_files[0])
    
    def load_creditcard_data(self, file_path=None):
        """
        Load credit card fraud dataset
        If file_path is not provided, it will download from Kaggle
        
        Args:
            file_path (str): Path to the dataset file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        if file_path is None:
            file_path = self.download_dataset(
                dataset_name='mlg-ulb/creditcardfraud',
                save_path='data/raw'
            )
            
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # Save the raw data as a joblib file for faster loading
        processed_path = 'data/processed/creditcard_raw.joblib'
        joblib.dump(df, processed_path)
        print(f"Raw data saved to {processed_path}")
        
        return df
        
    def load_paysim_data(self, file_path=None):
        """
        Load PaySim mobile transactions dataset
        If file_path is not provided, it will download from Kaggle
        
        Args:
            file_path (str): Path to the dataset file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        if file_path is None:
            file_path = self.download_dataset(
                dataset_name='ealtman2019/ibm-transactions-for-fraud-detection',
                save_path='data/raw'
            )
            
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # Save the raw data as a joblib file for faster loading
        processed_path = 'data/processed/paysim_raw.joblib'
        joblib.dump(df, processed_path)
        print(f"Raw data saved to {processed_path}")
        
        return df