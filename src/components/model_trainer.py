import numpy as np
import pandas as pd
import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from src.utils import save_object
from src.logger import logging
from src.exception import CustomException


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_model(self, true, predicted):
        accuracy = accuracy_score(true, predicted) * 100
        precision = precision_score(true, predicted) * 100
        recall = recall_score(true, predicted) * 100
        f1 = f1_score(true, predicted) * 100
        return accuracy, precision, recall, f1

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define classifiers
            models = {
                'LogisticRegression': LogisticRegression(),
                'RandomForestClassifier': RandomForestClassifier(),
                'XGBClassifier': XGBClassifier(eval_metric='logloss')
            }

            results = {}
            print("\n📊 Model Training and Evaluation Started...\n")

            for model_name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    accuracy, precision, recall, f1 = self.evaluate_model(y_test, y_pred)

                    results[model_name] = f1  # You can change this to accuracy if needed

                    print(f"\n✅ {model_name} Results:")
                    print(f"Accuracy: {accuracy:.2f}%")
                    print(f"Precision: {precision:.2f}%")
                    print(f"Recall: {recall:.2f}%")
                    print(f"F1 Score: {f1:.2f}%")
                    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
                    print("Classification Report:\n", classification_report(y_test, y_pred))

                except Exception as e:
                    print(f"❌ Error training model {model_name}: {e}")

            # Find best model
            best_model_name = max(results, key=results.get)
            best_model = models[best_model_name]
            best_score = results[best_model_name]

            logging.info(f"Best Model: {best_model_name} with F1 Score: {best_score:.2f}%")
            print(f"\n🏆 Best Model: {best_model_name} with F1 Score: {best_score:.2f}%")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Saved the best model successfully.")

            return best_model_name, best_score

        except Exception as e:
            logging.info("Exception occurred during model training")
            raise CustomException(e, sys)
