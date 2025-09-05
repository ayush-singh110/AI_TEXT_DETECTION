import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.utils import preprocess_text,stemming

@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('artifacts','raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df=pd.read_csv("notebook\\balanced_ai_human_prompts.csv")
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True,inplace=True)
            logging.info("Read the dataset as dataframe")
            df['text']=df['text'].apply(lambda x: preprocess_text(x))
            df['text']=df['text'].apply(lambda x:stemming(x))
            logging.info("Ingestion is complete")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            return self.ingestion_config.raw_data_path
        except Exception as e:
            raise CustomException(e,sys)
        
