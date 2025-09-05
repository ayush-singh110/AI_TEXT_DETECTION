import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.utils import evaluate_models, save_object

@dataclass
class ModelTrainerConfig:
    trainer_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__ (self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,transformed_data_path):
        logging.info("Model Trainer has started")
        try:
            df=pd.read_csv(transformed_data_path)
            X=df.drop(columns=['generated'],axis=1)
            y=df['generated'].astype(int)
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
            models={
                "LinearRegression":LogisticRegression(),
                "SVC":SVC(),
                "DecisionTreeClassifier":DecisionTreeClassifier(),
                "RandomForestClassifier":RandomForestClassifier(),
                "GradientBoostingClassifier":GradientBoostingClassifier()
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on both training and testing data")
            save_object(
                file_path=self.model_trainer_config.trainer_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            score=accuracy_score(y_test,predicted)
            return score,best_model_name
        except Exception as e:
            raise CustomException(e,sys)

