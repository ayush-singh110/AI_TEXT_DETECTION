import os
import sys
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import re  
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from bs4 import BeautifulSoup
from src.exception import CustomException
from gensim.utils import simple_preprocess
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')
import gensim
from sklearn.metrics import accuracy_score
import pickle

def preprocess_text(text):
    try:
        text=text.lower()
        text=re.sub("[^a-z A-Z 0-9]+","",text)
        text=" ".join([y for y in text.split() if y not in set(stopwords.words('english'))])
        text=re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&/~+#-]*[\w@?^=%&/~+#])?','',str(text))
        text=" ".join([y for y in text.split()])
        return text
    except Exception as e:
        raise CustomException(e,sys)
    
stemmer=SnowballStemmer("english")



def stemming(text):
    text=" ".join([stemmer.stem(y) for y in text.split()])
    return text

def simp_preprocess(file_path):
    try:
        words=[]
        df=pd.read_csv(file_path)
        for i in df.index:
            sent_token=sent_tokenize(df['text'][i])
            for i in sent_token:
                words.append(simple_preprocess(i,deacc=True))
        return words
    except Exception as e:
        raise CustomException(e,sys)

def avg_word2vec(doc):
    model=gensim.models.Word2Vec.load("Word2Vec.model")
    words_in_vocab=[word for word in doc if word in model.wv.index_to_key]
    if not words_in_vocab:
        return np.zeros(model.wv.vector_size)
    return np.mean([model.wv[word] for word in words_in_vocab],axis=0)

def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            model.fit(X_train,y_train)
            y_test_pred=model.predict(X_test)
            test_model_score=accuracy_score(y_test,y_test_pred)
            report[list(models.keys())[i]]=test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)