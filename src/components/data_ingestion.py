import os
import sys
from logger import logging
from exception import CustomException
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


from components.data_transfornation import DataTransformation
from components.data_transfornation import DataTransformationConfig

#from src.components.model_trainer import ModelTrainerConfig
#from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(r'C:\Users\pj\Desktop\End to End Project\Laptop_price_predictor\Notebook\data\laptop_data.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.15, random_state=2)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            # Check if all required columns are present in both DataFrames
            required_columns = [
                'Company', 'TypeName', 'Inches', 'ScreenResolution', 'Cpu', 'Ram', 'Memory',
                'Gpu', 'OpSys', 'Weight', 'Price'
            ]
            missing_train_columns = set(required_columns) - set(train_set.columns)
            missing_test_columns = set(required_columns) - set(test_set.columns)

            if missing_train_columns or missing_test_columns:
                logging.info("Required columns are missing in the training or testing DataFrame.")

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()  # Separate the paths

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    #train_arr,test_arr,_=

    #modeltrainer=ModelTrainer()
    #print(modeltrainer.initiate_model_trainer(train_arr,test_arr))