import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass
from src.components.data_transformation import DataTrans

from sklearn.model_selection import train_test_split

@dataclass
class DataInjConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataInj:
    def __init__(self):
        self.injection_config = DataInjConfig()

    def initiate_data_injection(self):
        logging.info("Reading dataset.")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            os.makedirs(os.path.dirname(self.injection_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.injection_config.raw_data_path, index=False, header=True)

            # Split data into train and test
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            train_df.to_csv(self.injection_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.injection_config.test_data_path, index=False, header=True)

            logging.info("Data Injection Complete.")
            return (self.injection_config.train_data_path, self.injection_config.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataInj()
    train_path, test_path = obj.initiate_data_injection()

    # Initiate data transformation after injection
    data_transformation = DataTrans()
    train_arr, test_arr, _ = data_transformation.initiate_data_trans(train_path, test_path)
    print(f"Transformed Train Data Shape: {train_arr.shape}")
    print(f"Transformed Test Data Shape: {test_arr.shape}")
