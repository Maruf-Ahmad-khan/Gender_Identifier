import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
     def __init__(self):
          pass
     
     def predict(self, features):
          try:
               preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
               model_path = os.path.join('artifacts', 'model.pkl')
               
               preprocessor = load_object(preprocessor_path)
               model = load_object(model_path)
               
               data_scaled = preprocessor.transform(features)
               
               pred = model.predict(data_scaled)
               return pred
               
          except Exception as e:
               logging.info("Exception occured in prediction")
               raise CustomException(e, sys)
          
class CustomData:
    def __init__(self, Favorite_Color, Favorite_Music_Genre, Favorite_Beverage, Favorite_Soft_Drink):
        self.Favorite_Color = Favorite_Color
        self.Favorite_Music_Genre = Favorite_Music_Genre
        self.Favorite_Beverage = Favorite_Beverage
        self.Favorite_Soft_Drink = Favorite_Soft_Drink

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Favorite Color': [self.Favorite_Color],
                'Favorite Music Genre': [self.Favorite_Music_Genre],
                'Favorite Beverage': [self.Favorite_Beverage],
                'Favorite Soft Drink': [self.Favorite_Soft_Drink],
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame created from user input")
            return df

        except Exception as e:
            logging.info("Exception in get_data_as_dataframe")
            raise CustomException(e, sys)

               
          
          
               