#==============================================================#
#  Basic Project - Natural Language Processing                 #
#   Author: María Sauras & Oscar Llorente                      #
#   Date: 19/12/2022                                           #                                    
#==============================================================#

######## Required libraries ########
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from gensim.utils import tokenize
import re  # For preprocessing
import pandas as pd  # For data handling
from collections import defaultdict  # For word frequency
import spacy  # For preprocessing
import logging  # Setting up the loggings to monitor gensim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


def load_df(df):
    return pd.read_csv(df)

#==============================================================#
#                STEP 1. Text Preprocessing                    #
#--------------------------------------------------------------#
#                                                              #
# Text preprocessing is an important step in NLP tasks, as it  #
# helps to get the text into a suitable format                 #
# for further analysis.                                        #
#                                                              #
#==============================================================#
class Text_preprocessing():

    def __init__(self, df):
        self.df = df
    
    # Processing dataframe.
    def reviews(self,column, column_label):
        df = self.df[[column, column_label]]
        return df.dropna().reset_index(drop=True)
    
    def homogenization(self, doc):
        # Lemmatizes and removes stopwords
        txt = [token.lemma_ for token in doc if not token.is_stop]
        # Word2Vec uses contextual words to learn the vector representation of a target word, 
        # if a sentence has only one or two words, the training benefit is very small so we eliminate it.
        if len(txt) > 2: return ' '.join(txt)

    # We are lemmatizing and removing the stopwords and non-alphabetic characters for each line of dialogue.
    def cleaning_data(self,data):
        nlp = spacy.load("en_core_web_sm") # para acelerar el proceso de limpieza
        cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in data)
        return [self.homogenization(doc) for doc in nlp.pipe(cleaning, batch_size=5000)]
    
def extract_data(projects_path, codes_path):
    
    codes_path = 'data/SciVocCodes.xlsx'
    projects_path = 'data/projects.xlsx'

    codes = pd.read_excel(codes_path, usecols=['code', 'full_code'], dtype={'code': 'string', 'full_code': 'string'})
    codes = codes.dropna().reset_index()
    codes['full_code'] = codes['full_code'].apply(lambda x: x[1:].split('/')[0])
    full_codes = codes['full_code'].tolist()
    codes = codes['code'].tolist()
    codes = dict(zip(codes, full_codes))

    classes_dict = {'21': '0', '23': '1', '25': '2', '27': '3', '29': '4', '31': '5'}

    projects = pd.read_excel(projects_path, usecols=['title', 'summary', 'euroSciVocCode'], 
                                        dtype={'title': 'string', 'summary': 'string','euroSciVocCode': 'string'})
    projects = projects.dropna().reset_index()
    projects['euroSciVocCode'] = \
        projects['euroSciVocCode'].apply(lambda code: classes_dict[codes[code[1:-1].split(',')[0]]])
    
    projects['summary'] = projects['title'] + projects['summary']
    
    corpus = projects['summary']
    labels = projects['euroSciVocCode']

    return corpus, labels
    
    
def preprocessing(texts):
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    texts = [word_tokenize(text) for text in texts]
    texts = [re.sub("[^A-Za-z']+", ' ', ' '.join(text)).lower() for text in texts]
    texts = [[word for word in text.split(' ') if word not in stop_words] for text in texts]
    texts = [[lemmatizer.lemmatize(word) for word in text] for text in texts]
    texts = [' '.join(text) for text in texts]
    
    return texts
    
    


#==============================================================#
#                STEP 2. Text Vectorization                    #
#--------------------------------------------------------------#
#                                                              #
# Text vectorization is the process of converting              #
# text data into numerical vectors that can be used            #
# as input to machine learning models.                         #
#                                                              #
#==============================================================#

class Text_Vectorization():

    def __init__(self, df):
        self.df = df
    

class Classificatio():
    def __init__(self, df):
        self.df = df


class Semantic_Analysis():
    def __init__(self, df):
        self.df = df


