#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import re
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

def max_length_text(df):
    max_length = 0
    for i in range(df.shape[0]):
        length = np.size(word_tokenize(df.at[i, 'text']))
        if length > max_length: max_length = length
    return max_length


def sequence_length_histogram(df):
    lengths = []
    for i in range(df.shape[0]):
        length = np.size(word_tokenize(df.at[i, 'text']))
        lengths.append(length)
    
    plt.hist(lengths, bins = 20)
    plt.show()
    return lengths


def text_normalization(data):
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply((lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x)))

    
def remove_stop_words(data, language, get_tokenize):
    stopwords = nltk.corpus.stopwords.words(language)
    if get_tokenize:
        for i in range(data.shape[0]):
            data.at[i, 'text'] = [word for word in nltk.word_tokenize(data.at[i, 'text']) if word not in stopwords]
    else:
        for i in range(data.shape[0]):
            data.at[i, 'text'] = [word for word in nltk.word_tokenize(data.at[i, 'text']) if word not in stopwords]
            data.at[i, 'text'] = ' '.join(data.at[i, 'text'])
            

def apply_stemming(data, language):
    stemmer = SnowballStemmer(language)
    for i in range(data.shape[0]):
         data.at[i, 'text'] = (' '.join([stemmer.stem(word) for word in data.at[i, 'text'].split()]))


##### get_matrix representation | Tf-idf for Classic ML

def get_matrix(data, vocabulary_length, stemming, remove_stopwords, language):

    df = data.copy(deep = True)
    
    text_normalization(df) # Text normalization
    
    # Stop_words
    if remove_stopwords:
        remove_stop_words(df, language, False)
    
    # Stemming
    if stemming:
        apply_stemming(df, language)
    
    # Word representation
    tfidf_vectorizer = TfidfVectorizer(max_df = 0.9, max_features = vocabulary_length, min_df = 0, use_idf = True)
    matrix = tfidf_vectorizer.fit_transform(df.text)
    
    return matrix, df
