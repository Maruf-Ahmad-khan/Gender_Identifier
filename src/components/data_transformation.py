import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation initiated")

            categorical_cols = ['Favorite Color', 'Favorite Music Genre', 'Favorite Beverage', 'Favorite Soft Drink']

            Color_categories = ['Neutral', 'Warm', 'Cool']
            Music_categories = ['Jazz/Blues', 'Folk/Traditional', 'R&B and soul', 'Electronic', 'Hip hop', 'Rock', 'Pop']
            Beverage_categories = ['Vodka', 'Whiskey', 'Wine', 'Other', 'Beer', "Doesn't drink"]
            Drink_categories = ['Coca Cola/Pepsi', 'Fanta', '7UP/Sprite', 'other']

            logging.info("Pipeline Initiated")

            # Only Categorical Pipeline since no numerical features
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encoder', OrdinalEncoder(
                        categories=[Color_categories, Music_categories, Beverage_categories, Drink_categories],
                        handle_unknown='use_encoded_value',
                        unknown_value=-1)),
                    
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            return preprocessor

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Gender'
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Label Encode the target — not saving it
            label_encoder = LabelEncoder()
            target_feature_train_df = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_df = label_encoder.transform(target_feature_test_df)

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applied preprocessing to train and test")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Saved preprocessor to pkl")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.info("Exception occurred in the initiate_data_transformation")
            raise CustomException(e, sys)
