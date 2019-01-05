CORPUS =['the sky is blue','sky is blue and sky is beautiful','the beautiful sky is so blue','i love blue cheese']
new_doc = ['loving this blue sky today']

#function to implement bag of stopwords
from sklearn.feature_extraction.text import CountVectorizer

def bow_extractor(corpus,ngram_range=(1,1)):

    vectorizer = CountVectorizer(min_df=1,ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer,features

#build bow vectorizer and get features
bow_vectorizer,bow_features = bow_extractor(CORPUS)
features = bow_features.todense()
print(features)

#extract features from new document usinf built bow_vectorizer
new_doc_features = bow_vectorizer.transform(new_doc)
new_doc_features = new_doc_features.todense()
print(new_doc_features)

#print the feature name
feature_names = bow_vectorizer.get_feature_names()
print(new_doc_features)

#feature vector
import pandas as pd

def display_features(features,feature_names):
    df = pd.DataFrame(data=features,columns=feature_names)
    print(df)

display_features(features,feature_names)
display_features(new_doc_features,feature_names)

#implementation of tfidf feature vector
from sklearn.feature_extraction.text import TfidfTransformer

def tfidf_transformer(bow_matrix):
    transformer = TfidfTransformer(norm='l2',smooth_idf=True,use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer,tfidf_matrix

import numpy as np
#from feature_extractors import tfidf_transformer
features_names = bow_vectorizer.get_feature_names()

#build tfidf transformer and show train corpus tfidf features
tfidf_trans,tfidf_features = tfidf_transformer(bow_features)
features = np.round(tfidf_features.todense(),2)
display_features(features,feature_names)

#show tfidf features for new_docusing built tfidf transformer
nd_tfidf = tfidf_trans.transform(new_doc_features)
nd_features = np.round(nd_tfidf.todense(),2)
display_features(nd_features,feature_names)

#bihind the scene
import scipy.sparse as sp
from numpy.linalg import norm
feature_names = bow_vectorizer.get_feature_names()

#compute term frequency
tf = bow_features.todense()
tf=np.array(tf,dtype='float64')

#show tern frequencies
print('scene')
display_features(tf,feature_names)

#bag of words feature matrix
df = np.diff(sp.csc_matrix(bow_features,copy=True).indptr)
df = 1+df #to smooth

#show document frequencies
display_features([df],feature_names)

#compute inverse document frequencies
total_docs = 1+len(CORPUS)
idf = 1.0 + np.log(float(total_docs)/df)

#show idf
display_features([np.round(idf,2)],feature_names)

#compute idf diagonal matrix
total_features = bow_features.shape[1]
idf_diag = sp.spdiags(idf,diags=0,m=total_features,n=total_features)
idf = idf_diag.todense()

#print the idf matrix
print(np.round(idf,2))

#compute tfidf features matrix
tfidf = tf * idf

#show tfidf matrix
display_features(np.round(tfidf,2),feature_names)

#compute l2 norms
norms = norm(tfidf,axis=1)

print(np.round(norms,2))

#compute normalized tfidf
norm_tfidf = tfidf / norms[:,None]

#show final
display_features(np.round(norm_tfidf,2),feature_names)

#compute new doc term freqs from bow freqs
nd_tf = new_doc_features
nd_tf = np.array(nd_tf,dtype='float64')

#compute tfidf using idf matrix from train corpus
nd_tfidf = nd_tf*idf
nd_norms = norm(nd_tfidf,axis=1)
norm_nd_tfidf = nd_tfidf / nd_norms[:,None]

#show new_doc tfidf feature vector
display_features(np.round(norm_nd_tfidf,2),feature_names)

#generic function
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_extractor(corpus,ngram_range=(1,1)):
    vectorizer = TfidfVectorizer(min_df=1,norm='l2',smooth_idf=True,use_idf=True,ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer,features

#build tfidf vector and get training corpus feature vectors
tfidf_vectorizer,tfidf_features = tfidf_extractor(CORPUS)
display_features(np.round(tfidf_features.todense(),2),feature_names)

#get tfidf feature vector for the new doc
nd_tfidf = tfidf_vectorizer.transform(new_doc)
display_features(np.round(nd_tfidf.todense(),2),feature_names)

#advance
print('-------------------------------------------------------------')

#advanced word vectorization models
import gensim
import nltk
#tokenize corpora
TOKENIZED_CORPUS =[nltk.word_tokenenize(sentence) for sentence in CORPUS]
tokenized_new_doc = [nltk.word_tokenenize(sentence) for sentence in new_doc]

#build the word2vec model on our training corpus
model = gensim.models.word2vec(TOKENIZED_CORPUS,size=10,window=10,min_count=2,sample=1e-3)
def average_word_vector(words,model,vocabulary,num_features):
    feature_vector = np.Zeros((num_features,),dtype='float64')
    nwords=0

    for word in words:
        if word in vocabulary:
            nwords=nwords+1.
            feature_vector=np.add(feature_vector,model[word])

    if nwords:
        feature_vector=np.divide(feature_vector,nwords)

    return feature_vector

#generalize above function for a corpus of document
