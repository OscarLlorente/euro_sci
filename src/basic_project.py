#==============================================================#
#  Basic Project - Natural Language Processing                 #
#   Author: María Sauras & Oscar Llorente                      #
#   Date: 19/12/2022                                           #                                    
#==============================================================#

# machine learning and NLP libraries
import re 
import numpy as np
import pandas as pd  
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report

# other libraries
from typing import List, Tuple, Union

    
def extract_data(projects_path: str, codes_path: str) -> Tuple[List[List[str]], np.ndarray, np.ndarray, np.ndarray]:

    # create variable to get correct classes
    classes_dict = {'21': '0', '23': '1', '25': '2', '27': '3', '29': '4', '31': '5'}
    
    # extract codes dictionary
    codes = pd.read_excel(codes_path, usecols=['code', 'full_code'], dtype={'code': 'string', 'full_code': 'string'})
    codes = codes.dropna().reset_index()
    codes['full_code'] = codes['full_code'].apply(lambda x: x[1:].split('/')[0])
    full_codes = codes['full_code'].tolist()
    codes = codes['code'].tolist()
    codes = dict(zip(codes, full_codes))

    # extract projects text, variables and codes
    projects = pd.read_excel(projects_path, usecols=['title', 'summary', 'euroSciVocCode', 'totalCost', 
                                                     'ecMaxContribution'], 
                                        dtype={'title': 'string', 'summary': 'string','euroSciVocCode': 'string', 
                                               'totalCost': float, 'ecMaxContribution': float})
    projects = projects.dropna().reset_index()
    projects['euroSciVocCode'] = \
        projects['euroSciVocCode'].apply(lambda code: classes_dict[codes[code[1:-1].split(',')[0]]])
    
    # concat title and summary
    projects['summary'] = projects['title'] + projects['summary']
    
    # get final variables to return them
    corpus = projects['summary']
    labels = projects['euroSciVocCode'].to_numpy()
    costs = projects['totalCost'].to_numpy()
    contributions = projects['ecMaxContribution'].to_numpy()

    return corpus, labels, costs, contributions
    
    
def preprocessing(texts: List[str]) -> List[str]:
    
    # set needed objects
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # tokenize
    texts = [word_tokenize(text) for text in texts]
    
    # eliminate non char tokens and remove stop-words
    texts = [re.sub("[^A-Za-z']+", ' ', ' '.join(text)).lower() for text in texts]
    texts = [[word for word in text.split(' ') if word not in stop_words] for text in texts]
    
    # lemmantize and return list of strings
    texts = [[lemmatizer.lemmatize(word) for word in text] for text in texts]
    texts = [' '.join(text) for text in texts]
    
    return texts


class BasicModel:
    
    def __init__(self, corpus: Union[np.ndarray, csr_matrix], labels: np.ndarray) -> None:
        
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(corpus, labels, test_size=0.1, random_state=42)
            
        # train model
        grid_param = {'kernel':('linear', 'rbf'), 'C':[0.1 ,1, 10]}
        svc = svm.SVC(class_weight='balanced')
        model_grid = GridSearchCV(svc, grid_param, refit=True, cv = 2, n_jobs=-1)
        model_grid.fit(self.x_train, self.y_train)
        
        # get predictions
        self.y_pred = model_grid.predict(self.x_test)
    
    @property
    def accuracy(self) -> float:
        return balanced_accuracy_score(self.y_test, self.y_pred)
    
    @property
    def classification_report(self) -> pd.DataFrame:
        return classification_report(self.y_test, self.y_pred)
    
    