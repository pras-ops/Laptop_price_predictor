import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd

from sklearn.compose import ColumnTransformer
from exception import CustomException
from logger import logging
from utils.utils import save_object
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logging.info("DataTransformation: Starting data transformation steps...")
        # Logging the DataFrame before any transformations
        logging.info(f"Before transformation:\n{X}")
        X 
        logging.info(f"Column names present in the data: {X.columns.tolist()}")
        # Extract column names from the DataFrame
        existing_columns = set(X.columns)

        # Check if all the required columns are present in the DataFrame
        required_columns = [
            'Company', 'TypeName', 'Inches', 'ScreenResolution', 'Cpu', 'Ram', 'Memory',
            'Gpu', 'OpSys', 'Weight', 'Price'
        ]
        missing_columns = set(required_columns) - existing_columns
        if missing_columns:
            logging.warning(f"Missing columns in the DataFrame: {missing_columns}")

        # Selecting and reordering columns directly
        X = X[required_columns]

        # Apply other transformations only if certain columns are present
        if 'Unnamed: 0' in X.columns:
            X.drop(columns=['Unnamed: 0'], inplace=True)
            logging.info("remove_unnamed_column")

        if 'ScreenResolution' in X.columns:
            if 'ScreenResolution' in X.columns:
                X.rename(columns={'ScreenResolution': 'Resolution'}, inplace=True)
            X.rename(columns={
                'Company': 'Brand',
                'TypeName': 'Type',
                'Inches': 'ScreenSize',
                'Ram': 'RAM',
                'Memory': 'Storage',
                'Gpu': 'GPU',
                'OpSys': 'OperatingSystem',
                'Weight': 'Weight',
                'Price_euros': 'Price'
            }, inplace=True)
            logging.info("rename_columns")

        if 'RAM' in X.columns:
            X['RAM'] = X['RAM'].str.replace('GB', '')
            X['RAM'] = X['RAM'].astype('int32')
            logging.info("remove_units and convert_ram_to_int")

        if 'Weight_kg' in X.columns:
            X['Weight_kg'] = X['Weight_kg'].str.replace('kg', '')
            X['Weight_kg'] = X['Weight_kg'].astype('float32')
            logging.info("convert_weight_to_float")

        if 'Resolution' in X.columns:
            X['TouchScreen'] = X['Resolution'].apply(lambda element: 1 if 'Touchscreen' in element else 0)
            X['IPS'] = X['Resolution'].apply(lambda element: 1 if 'IPS' in element else 0)
            logging.info("extract_touchscreen_feature and extract_ips_feature")

        if 'CPU' in X.columns:
            print("Columns before extracting CPU feature:", X.columns)
            X['Processor'] = X['CPU'].apply(lambda text: " ".join(text.split()[:3]))
            print("Columns after extracting CPU feature:", X.columns)
            logging.info("extract_cpu_feature")

        if 'Resolution' in X.columns:
            split_df = X['Resolution'].str.split('x', n=1, expand=True)
            X['X_res'] = split_df[0].str.replace(',', '').str.findall(r'(\d+\.?\d+)').apply(lambda x: x[0]).astype('int')
            X['Y_res'] = split_df[1].astype('int')
            X['PPI'] = (((X['X_res'] ** 2 + X['Y_res'] ** 2) ** 0.5) / X['ScreenSize']).astype('float')
            # Drop only the columns that are no longer needed for further processing
            X.drop(columns=['X_res', 'Y_res'], inplace=True)
            logging.info("extract_resolution_features")

        if 'CPU_name' in X.columns:
            listtoapply = ['HDD', 'SSD', 'Hybrid', 'FlashStorage']
            for value in listtoapply:
                X['Layer1'+value] = X['first'].apply(lambda x: 1 if value in x else 0)

            X['first'] = X['first'].str.replace(r'\D','')
            X['first'] = X['first'].astype('int')

            listtoapply1 = ['HDD', 'SSD', 'Hybrid', 'FlashStorage']
            X['Second'] = X['Second'].fillna("0")
            for value in listtoapply1:
                X['Layer2'+value] = X['Second'].apply(lambda x: 1 if value in x else 0)

            X['Second'] = X['Second'].str.replace(r'\D','')
            X['Second'] = X['Second'].astype('int')

            # Multiplying the elements and storing the result in subsequent columns
            X["HDD"] = (X["first"] * X["Layer1HDD"] + X["Second"] * X["Layer2HDD"])
            X["SSD"] = (X["first"] * X["Layer1SSD"] + X["Second"] * X["Layer2SSD"])

            # ... (add similar lines for Hybrid and Flash_Storage if needed)

            # Dropping unnecessary columns
            X.drop(columns=['first', 'Second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
                            'Layer1FlashStorage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
                            'Layer2FlashStorage'], inplace=True)
            X.drop(columns=['Memory'], inplace=True)
            X.drop(columns=['Hybrid', 'Flash_Storage'], inplace=True)

            # ... (add any additional processing steps if needed)

            # Processing GPU column
            X['Gpu brand'] = X['Gpu'].apply(lambda x: x.split()[0])
            X.drop(columns=['Gpu'], inplace=True)

            # Setting OpSys category
            X['OpSys'] = X['OpSys'].apply(lambda x: self.setcategory(x))

        return X

    def data_transformation_pipeline(self, data):
        # Automatically identify numerical and categorical columns
        numerical_columns = data.select_dtypes(include=['int32', 'int64', 'float32', 'float64']).columns
        categorical_columns = data.select_dtypes(include=['object']).columns

        data_transformer = DataTransformation()

        # Create a pipeline to encapsulate the data transformation steps
        data_pipeline = Pipeline(steps=[
            ('data_transform', data_transformer)
        ])

        # Create the ColumnTransformer
        preprocessor = ColumnTransformer(transformers=[
            ('num', data_pipeline, numerical_columns),
            ('cat', data_pipeline , categorical_columns)
        ])

        logging.info("data_transformation_pipeline: Starting data transformation pipeline...")

        try:
            processed_data = preprocessor.fit_transform(data)
        except KeyError as e:
            logging.error(f"data_transformation_pipeline: KeyError - {e}")
            raise  # Re-raise the exception to see the full traceback

        logging.info("data_transformation_pipeline: Data transformation pipeline completed.")

        return processed_data

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            logging.info("Train DataFrame info:")
            logging.info(f"Column names present in the train_df: {train_df.columns.tolist()}")
            test_df = pd.read_csv(test_path)
            logging.info("Test DataFrame info:")
            logging.info(f"Column names present in the test_df: {test_df.columns.tolist()}")

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.data_transformation_pipeline(train_df)

            target_column_name = "Price"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Use the fitted preprocessing object to transform both train and test data
            input_feature_train_arr = preprocessing_obj.transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Log the columns in the DataFrame after transformation
            logging.info("Columns in the transformed train DataFrame:")
            logging.info(pd.DataFrame(input_feature_train_arr, columns=input_feature_train_df.columns))

            logging.info("Columns in the transformed test DataFrame:")
            logging.info(pd.DataFrame(input_feature_test_arr, columns=input_feature_test_df.columns))

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
