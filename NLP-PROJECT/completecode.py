

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from ast import literal_eval
import pandas as pd
import numpy as np

import re

# Dictionary of all tags from train corpus with their counts.
from collections import Counter

from scipy import sparse as sp_sparse
    
from sklearn.feature_extraction.text import TfidfVectorizer
########################################################################

train = pd.read_csv("/home/rana/mypython/NLP-PROJECT/week1/train.csv")
test = pd.read_csv('/home/rana/mypython/NLP-PROJECT/week1/test.csv')
X_train, y_train = train['text'].values, train['author'].values


y_train_hot=np.zeros(y_train.shape,dtype='int')
for i in range(0 ,len(y_train)):
    if y_train[i]=="EAP":
        y_train_hot[i]=0
    elif y_train[i]=="HPL":
        y_train_hot[i]=1
    elif y_train[i]=="MWS":
        y_train_hot[i]=2
#X_val, y_val = validation['title'].values, validation['tags'].values
X_test = test['text'].values


#######################################################################
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    #print(text)
    text = text.lower() # lowercase text
    #print(text)
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    # print(text)
    text = re.sub(BAD_SYMBOLS_RE, "", text) # delete symbols which are in BAD_SYMBOLS_RE from text
    # print(text)
    text = " " + text + " "
    for sw in STOPWORDS:
        text = text.replace(" "+sw+" ", " ") # delete stopwors from text
    #print(text)
    text = re.sub('[ ][ ]+', " ", text)
    length=len(text)-1
    #print(length)
    
    if text[0] == ' ':
        text = text[1:]
    if length == 0:
        return text
    if text[length-1] == ' ':
        text = text[:-1]
        
    # text = text[1:-1]
    # print(text)
    return text

X_train = [text_prepare(x) for x in X_train]
#X_val = [text_prepare(x) for x in X_val]
X_test = [text_prepare(x) for x in X_test]
#print(stopwords.words('english'))
########################################################################


# Dictionary of all tags from train corpus with their counts.
tags_counts = Counter() #{}
# Dictionary of all words from train corpus with their counts.
words_counts = Counter() #{}

######################################
######### YOUR CODE HERE #############
######################################
# print(X_train[:3], y_train[:3])
for sentence in X_train:
    for word in sentence.split():
        # print(word)
        words_counts[word] += 1


for l in y_train:
        tags_counts[l] += 1

most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:]
most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:]

DICT_SIZE = 5000
WORDS_TO_INDEX = {p[0]:i for i,p in enumerate(most_common_words[:DICT_SIZE])} ####### YOUR CODE HERE #######
INDEX_TO_WORDS = {WORDS_TO_INDEX[k]:k for k in WORDS_TO_INDEX}####### YOUR CODE HERE #######
ALL_WORDS = WORDS_TO_INDEX.keys()

def my_bag_of_words(text, words_to_index, dict_size):
    """
        text: a string
        dict_size: size of the dictionary
        
        return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    for word in text.split():
        if word in words_to_index:
            result_vector[words_to_index[word]] += 1
    return result_vector


X_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
#X_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
X_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])
print('X_train shape ', X_train_mybag.shape)
#print('X_val shape ', X_val_mybag.shape)
print('X_test shape ', X_test_mybag.shape)
x_train_vsm=np.array(X_train_mybag.toarray())
x_test_vsm=np.array(X_test_mybag.toarray())

#################################################################################

def tfidf_features(X_train, X_test):
    """
        X_train, X_val, X_test — samples        
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result
     
    tfidf_vectorizer = TfidfVectorizer(token_pattern='(\S+)', min_df=5, max_df=0.9, ngram_range=(1,2)) ####### YOUR CODE HERE #######
    
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    tfidf_vectorizer.fit(X_train)
    X_train = tfidf_vectorizer.transform(X_train)
    X_test = tfidf_vectorizer.transform(X_test)
    print(tfidf_vectorizer.vocabulary_)
    return X_train , X_test, tfidf_vectorizer.vocabulary_

X_train_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_test)
tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}

x_train_vsm=np.array(X_train_tfidf.toarray())
x_test_vsm=np.array(X_test_tfidf.toarray())

################################################################################
