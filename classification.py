import tensorflow as tf
import numpy as np
import pandas as pd

dataset = pd.read_csv('/home/DEADPOOL/Downloads/ag_news.csv')
dataset.info()

#train,test
headlines=dataset['headline'].values
labels=dataset['label'].values

train_headlines=headlines[:75000]
train_labels=labels[:75000]

test_headlines=headlines[75000:]
test_labels=labels[75000:]

#print(train_headlines.shape,test_headlines.shape)

#normalize the dataset
import contractions
from bs4 import BeautifulSoup
import unicodedata
import re

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def expand_contractions(text):
    return contractions.fix(text)

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text

def pre_process_document(document):
    # strip HTML
    document = strip_html_tags(document)
    # lower case
    document = document.lower()
    # remove extra newlines (often might be present in really noisy text)
    document = document.translate(document.maketrans("\n\t\r", "   "))
    # remove accented characters
    document = remove_accented_chars(document)
    # expand contractions
    document = expand_contractions(document)
    # remove special characters and\or digits
    # insert spaces between special characters to isolate them
    special_char_pattern = re.compile(r'([{.(-)!}])')
    document = special_char_pattern.sub(" \\1 ", document)
    document = remove_special_characters(document, remove_digits=True)
    # remove extra whitespace
    document = re.sub(' +', ' ', document)
    document = document.strip()

    return document


pre_process_corpus = np.vectorize(pre_process_document)

train_headlines = pre_process_corpus(train_headlines)
#val_reviews = pre_process_corpus(val_reviews)
test_headlines = pre_process_corpus(test_headlines)

#feature_extraction
from feature_extractors import bow_extractor, tfidf_extractor
from feature_extractors import averaged_word_vectorizer
from feature_extractors import tfidf_weighted_averaged_word_vectorizer
import nltk
import gensim

# bag of words features
bow_vectorizer, bow_train_features = bow_extractor(train_headlines)
bow_test_features = bow_vectorizer.transform(test_headlines)

# tfidf features
tfidf_vectorizer, tfidf_train_features = tfidf_extractor(train_headlines)
tfidf_test_features = tfidf_vectorizer.transform(test_headlines)

# tokenize documents
tokenized_train = [nltk.word_tokenize(text)
                   for text in train_headlines]
tokenized_test = [nltk.word_tokenize(text)
                   for text in test_headlines]

# build word2vec model
model = gensim.models.Word2Vec(tokenized_train,
                               size=500,
                               window=100,
                               min_count=30,
                               sample=1e-3)

# averaged word vector features
avg_wv_train_features = averaged_word_vectorizer(corpus=tokenized_train,
                                                 model=model,
                                                 num_features=500)
avg_wv_test_features = averaged_word_vectorizer(corpus=tokenized_test,
                                                model=model,
                                                num_features=500)

# tfidf weighted averaged word vector features
vocab=tfidf_vectorizer.vocabulary_
tfidf_wv_train_features=tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_train,tfidf_vectors=tfidf_train_features,tfidf_vocabulary=vocab, model=model,num_features=500)
tfidf_wv_test_features =tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_test,tfidf_vectors=tfidf_test_features,tfidf_vocabulary=vocab, model=model,num_features=500)

#models
from sklearn import metrics
import numpy as np

def get_metrics(true_labels, predicted_labels):
    print ('Accuracy:', np.round(
                        metrics.accuracy_score(true_labels,
                                               predicted_labels),
                        2))
    print ('Precision:', np.round(
                        metrics.precision_score(true_labels,
                                               predicted_labels,
                                               average='weighted'),
                        2))
    print ('Recall:', np.round(
                        metrics.recall_score(true_labels,
                                               predicted_labels,
                                               average='weighted'),
                        2))
    print ('F1 Score:', np.round(
                        metrics.f1_score(true_labels,
                                               predicted_labels,
                                               average='weighted'),
                        2))

#ml algo
def train_predict_evaluate_model(classifier,
                                 train_features, train_labels,
                                 test_features, test_labels):
    # build model
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features)
    # evaluate model prediction performance
    get_metrics(true_labels=test_labels,
                predicted_labels=predictions)
    return predictions

#models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

mnb = MultinomialNB()
svm = SGDClassifier(loss='hinge', n_iter=100)

#svm with tfidf features
svm_tfidf_predictions = train_predict_evaluate_model(classifier=svm,train_features=tfidf_train_features,train_labels=train_labels,test_features=tfidf_test_features,test_labels=test_labels)   
