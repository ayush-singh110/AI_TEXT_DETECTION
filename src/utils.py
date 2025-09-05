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
from nltk import pos_tag, word_tokenize

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
    print("stemming:",text)
    text=" ".join([stemmer.stem(y) for y in text.split()])
    return text