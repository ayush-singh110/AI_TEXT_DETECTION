import pandas as pd
from src.exception import CustomException
from src.logger import logging
import sys
from src.utils import load_object, preprocess_text, stemming, avg_word2vec
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from gensim.utils import simple_preprocess


class PredictPipeline:
    def __init__ (self):
        pass
    def predict(self,text):
        try:
            model_path="artifacts\\model.pkl"
            model=load_object(model_path)
            text=preprocess_text(text)
            text=stemming(text)
            words=[]
            sent_token=sent_tokenize(text)
            for i in sent_tokenize:
                words.append(simple_preprocess(i,deacc=True))
            vector=avg_word2vec(words)
            vector=vector.reshape(1,-1)
            pred=model.predict(vector)
            prediction=pred[0]
            return prediction
        except Exception as e:
            raise CustomException(e,sys)