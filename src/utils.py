import os
import sys
import pandas as pd
import numpy as np
import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from bs4 import BeautifulSoup
from src.exception import CustomException
from nltk.corpus import wordnet
from gensim.utils import simple_preprocess
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')
import gensim

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