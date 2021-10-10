import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
import string
import re
import time
start1 = time.time()
ps = PorterStemmer()
pd.set_option('display.max_colwidth', 100)
stopwords = nltk.corpus.stopwords.words('english')
dataset = pd.read_csv('C:/Users/Bcorpus.csv')
dataset.columns = ['ID', 'address', 'Code', 'Comment']
print(dataset.head(5))

def clean_text(text):
    text = "".join([c for c in text if c not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

dataset['Code_nostop'] = dataset['Code'].apply(lambda x: clean_text(x.lower()))
dataset['Comment_nostop'] = dataset['Comment'].apply(lambda x: clean_text(x.lower()))
print(dataset.head(5))

def stemming(tokenized_text):
    text = [ps.stem(word) for word in tokenized_text]
    return text

dataset['Code_stemmed'] = dataset['Code_nostop'].apply(lambda x: stemming(x))
dataset['Comment_stemmed'] = dataset['Comment_nostop'].apply(lambda x: stemming(x))
print(dataset.head(5))

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

print("Time taken to clean data: %s seconds" % (time.time() - start1))

start2 = time.time()
##Code Comparison

Code_corpus= dataset.iloc[0:100, 6]
taggedcode = [TaggedDocument(d, [i]) for i, d in enumerate(Code_corpus)]
taggedcode

model1 = Doc2Vec(taggedcode, vector_size = 20, window = 2, min_count = 1, epochs = 100)

code = input("Enter Code to be searched")

test_doc = word_tokenize(code.lower())
test_doc_vector = model1.infer_vector(test_doc)
temp = np.empty((0,0))
temp = model1.dv.most_similar(positive = [test_doc_vector])
print (temp)

for x in range(10):
    a = temp[x][0]
    print(dataset.iloc[a, 1])

print("Time taken Doc2vec Model: %s seconds" % (time.time() - start2))