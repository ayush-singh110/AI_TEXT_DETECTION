import os 
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.components.data_ingestion import DataIngestion
import pandas as pd
from src.utils import simp_preprocess, avg_word2vec
import gensim
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split

@dataclass
class DataTransformationConfig:
    transformed_data_path: str=os.path.join('artifacts','transformed_data.csv')

class DataTransformation:
    def __init__ (self):
        self.data_transformation_config=DataTransformationConfig()

    def word2vec(self,words):
        model=gensim.models.Word2Vec(words)
        model.save("Word2Vec.model")

    def transformation(self,raw_data_path):
        logging.info("Data Transformation has started")
        try:
            df=pd.read_csv(raw_data_path)
            words=simp_preprocess(raw_data_path)
            self.word2vec(words)
            X=[]
            for i in tqdm.tqdm(range(len(words))):
                X.append(avg_word2vec(words[i]))
            X=np.array(X)
            df1=pd.DataFrame(X)
            df1['generated']=df['generated'].reset_index(drop=True)
            logging.info("Data transformation is completed")
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_data_path),exist_ok=True)
            df1.to_csv(self.data_transformation_config.transformed_data_path,index=False,header=True)
            return self.data_transformation_config.transformed_data_path
        except Exception as e:
            raise CustomException(e,sys)