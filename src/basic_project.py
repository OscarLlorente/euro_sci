#==============================================================#
#  Basic Project - Natural Language Processing                 #
#   Author: María Sauras & Oscar Llorente                      #
#   Date: 19/12/2022                                           #                                    
#==============================================================#

######## Required libraries ########
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


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
        return self.df[column]
    
    ## STEP 1. TOKENIZATION
    def tokenization(self):
        rvw = self.reviews('Review Text')
        return [wordpunct_tokenize(t) for t in sent_tokenize(rvw)]
    
    ## STEP 2. HOMOGENIZATION
    def homogenization(self):
        rvw = self.tokenization()
        '''
        review_tokens_filtered = []
        for token in rvw:
            if token.isalnum(): review_tokens_filtered.append(token.lower())
        return review_tokens_filtered
        '''
        # Lower-cased of the tokens and elimination of non-alphanumeric characters
        review_tokens_filtered =  [token.lower() for token in rvw if token.isalnum() ]
        # Word normalization --> Lemmatization
        wnl = WordNetLemmatizer()
        return [wnl.lemmatize(el) for el in review_tokens_filtered]

    ## STEP 3. Removing those words that are very common in language and 
    # do not carry out useful semantic content --> Removing stopwords
    def cleaning_data(self):
        sw = stopwords.words('english')
        l = self.homogenization()
        return [word for word in l if word not in sw]


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


