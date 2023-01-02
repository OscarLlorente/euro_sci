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
    def reviews(self,column):
        df = self.df[column]
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


