# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 15:55:56 2019

@author: home
"""

import numpy as np
import pandas as pd 
import nltk
from nltk.util import ngrams
import os
from sklearn.feature_extraction.text import TfidfVectorizer#,CountVectorizer
path = "C://Users//home//Desktop//txt_sentoken//pos"
path_neg = "C://Users//home//Desktop//txt_sentoken//neg"

file_list = os.listdir(path)
file_list_neg = os.listdir(path_neg)
print (len(file_list))
print (len(file_list_neg))

df = pd.DataFrame(columns=['file_name', 'text', 'target'])

df.head()

for name in file_list :
    df = df.append({'file_name': name ,
                    'text': open(path+'//'+name).read(),
                    'target' : 'pos'} ,ignore_index=True)
                    
print (df['text'][1])

for name in file_list_neg :
    df = df.append({'file_name': name ,
                    'text': open(path_neg+'//'+name).read(),
                    'target' : 'neg'} ,ignore_index=True)
 
df.shape
df['text'][0:2]
df_shuffle= df.sample(frac=1, random_state=1).reset_index(drop=True)
df_shuffle['text'][0:2]
corpus = [ ]

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    sent = [word for word,pos in sent if (pos == 'JJ' or pos =='JJR' or pos == 'JJS' or  pos =='VB' or pos =='VBD' or pos =='VBG' or pos =='VBN' or pos =='VBP' or pos =='VBZ' or  pos =='RB' or pos =='RBR' or pos =='RBS' or pos =='WRB')] 
    return sent

for word in df_shuffle['text'][0:1500]:
    corpus += (preprocess(word))
df_shuffle['text'][500:1500].shape
len(corpus)

corpus = list(dict.fromkeys(corpus))

f1 = list(ngrams(corpus,2))
f1 =  [" ".join(ngram) for ngram in f1]
len(f1)
f1 = list(dict.fromkeys(f1))

f2 = list(ngrams(corpus,3))
f2 =  [" ".join(ngram) for ngram in f2]
len(f2)
f2 = list(dict.fromkeys(f2))

features = corpus + f1 +f2
len(features)

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w+',vocabulary = features)
tfidf__count_vector = tfidf_vect.fit_transform(df_shuffle['text'])
feature_names_tfidf = tfidf_vect.get_feature_names()
len(feature_names_tfidf)
tfidf__count_vector.shape

from sklearn import preprocessing, metrics
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(df_shuffle['target'])
df_shuffle['target'].unique()
train_y

df_shuffle['target'][500:1500].shape

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
dual=[True,False]
max_iter=[100,110,120,130,140]
C = [0.001,0.01,0.1,1,10,100,1000]
param_grid = dict(dual=dual,max_iter=max_iter,C=C)
import time
lr = LogisticRegression(penalty='l2')
grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv = 3, n_jobs=-1)
start_time = time.time()
grid_result = grid.fit(tfidf__count_vector[0:1500],train_y[0:1500])
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')
y_predict = grid.predict(tfidf__count_vector[1500:2000])
accuracy = metrics.accuracy_score( y_predict, train_y[1500:2000])
print (accuracy)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis(solver='svd',store_covariance=False)
X = tfidf__count_vector[0:1500].toarray()
X.shape
clf.fit(X,train_y[0:1500]) 
Y = tfidf__count_vector[1500:2000].toarray()
y_pred=clf.predict(Y)
accuracy = metrics.accuracy_score( y_pred,train_y[1500:2000] )
print (accuracy)


from sklearn import svm
gamma= [0.01,0.1,1,10,100]
C=[0.01,0.1,1,10,100]
svm_cv = svm.SVC(kernel = 'rbf')
param_grid = dict(gamma = gamma, C= C)
import time
grid = GridSearchCV(estimator=svm_cv, param_grid=param_grid, cv = 3, n_jobs=-1)
start_time = time.time()
grid_result = grid.fit(tfidf__count_vector[0:1500],train_y[0:1500])
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')
y_pred=grid.predict(tfidf__count_vector[1500:2000])
accuracy = metrics.accuracy_score( y_pred, train_y[1500:2000])
print (accuracy)


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components = 500)
svd.get_params
tfidf_modif_train = svd.fit_transform(tfidf__count_vector[0:1500])
tfidf_modif_test = svd.fit_transform(tfidf__count_vector[1500:2000])
tfidf_modif_train.shape
tfidf_modif_test.shape


from sklearn.neighbors import KNeighborsClassifier
n_neighbors=[1,3,5]
p = [1,2,np.inf]
param_grid = dict(n_neighbors = n_neighbors, p= p)
neigh = KNeighborsClassifier(metric = 'minkowski')
import time
grid = GridSearchCV(estimator=neigh, param_grid=param_grid, cv = 3, n_jobs=-1)
start_time = time.time()
grid_result = grid.fit(tfidf_modif_train,train_y[0:1500])
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')
y_pred=grid.predict(tfidf_modif_test)
accuracy = metrics.accuracy_score( y_pred, train_y[1500:2000])
print (accuracy)



from sklearn.neighbors.kde import KernelDensity
bandwidth=[0.1,1,10]
#kernel = ['gaussian','epanechnikov','cosine']
param_grid = dict( bandwidth=bandwidth)
kde = KernelDensity(kernel='cosine')
import time
grid = GridSearchCV(estimator=kde, param_grid=param_grid, cv = 3, n_jobs=-1)
start_time = time.time()
grid_result = grid.fit(tfidf_modif_train,train_y[0:1500])
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')
grid.get_params

dimmu = pd.DataFrame(tfidf_modif_train)
dimmu.shape
dummy = pd.DataFrame(tfidf_modif_test)

from sklearn.base import BaseEstimator, ClassifierMixin

class KDEClassifier(BaseEstimator, ClassifierMixin):
    """Bayesian generative classification based on KDE
    
    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    """
    def __init__(self, bandwidth=0.1, kernel='cosine'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        
    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel).fit(Xi)
                        for Xi in training_sets]
       # self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
 #                          for Xi in training_sets]
        self.logpriors_ = [np.log(0.492),np.log(0.508)]
        return self
        
    def predict_proba(self, X):
        logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)
        
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]

model = KDEClassifier()
label = model.fit(dimmu, train_y[0:1500]).predict(dummy)
#label
accuracy = metrics.accuracy_score( y_pred, train_y[1500:2000])
print (accuracy)





