import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def save_object(file_path, obj):
     try:
          dir_path = os.path.dirname(file_path)
          
          os.makedirs(dir_path, exist_ok=True)
          
          with open(file_path, "wb") as file_obj:
               pickle.dump(obj, file_obj)
               
     except Exception as e:
          raise CustomException(e, sys)
     
     
def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Trains and evaluates classification models.
    Returns a dictionary with model names and F1 scores (can be changed to any metric).
    """
    try:
        report = {}

        for name, model in models.items():
            logging.info(f"Training model: {name}")
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_test_pred) * 100
            precision = precision_score(y_test, y_test_pred, zero_division=0) * 100
            recall = recall_score(y_test, y_test_pred, zero_division=0) * 100
            f1 = f1_score(y_test, y_test_pred, zero_division=0) * 100

            # Log model performance
            logging.info(f"{name} - Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1 Score: {f1:.2f}%")

            # Save F1 score in report (can change to accuracy if preferred)
            report[name] = f1

        return report

    except Exception as e:
        logging.info("Exception occurred during model evaluation")
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a serialized object from a pickle file.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info("Exception occurred in load_object function")
        raise CustomException(e, sys)