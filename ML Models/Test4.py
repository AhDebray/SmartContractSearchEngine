import nltk
from sklearn.metrics.pairwise import euclidean_distances
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
import string
import re
import gensim
import numpy
import time
start1 = time.time()
raw_data = pd.read_csv('C:/Users/Bcorpus.csv')
print(raw_data[0:500])

code_list = raw_data['function'].tolist()
comment_list = raw_data['docstring'].tolist()
print(code_list[0:5])
print(comment_list[0:5])

combined_df = pd.DataFrame({
    'Code': code_list,
    'Comment': comment_list
})
print(combined_df.head(5))

raw_data.columns = ['ID', 'address', 'Code', 'Comment']
print(raw_data.head(5))

data = pd.read_csv('C:/Users/Bcorpus.csv')
data.columns = ['ID', 'address', 'Code', 'Comment']
pd.set_option('display.max_colwidth', 100)
print(string.punctuation)

def remove_punctuation(txt):
    txt_nopunct = "".join([c for c in txt if c not in string.punctuation])
    return txt_nopunct

data['Code_clean'] = data['Code'].apply(lambda x: remove_punctuation(x))
data['Comment_clean'] = data['Comment'].apply(lambda x: remove_punctuation(x))
print(data.head())

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

print("Time taken to clean data: %s seconds" % (time.time() - start1))

##Bag Of Words
##codeDoc = dataset['Code_stemmed']
##commentDoc = dataset['Code_stemmed']
##print(codeDoc[0:5])
##print(commentDoc[0:5])


start2 = time.time()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
temp = data.iloc[0:10, 4]
print(temp)
temp = tfidf.fit_transform(data.iloc[0:10, 4])
print(temp)

from sklearn.metrics.pairwise import cosine_similarity
cosine = cosine_similarity(temp, temp)
print(cosine)

print("Time taken for TF-IDF cosine distance: %s seconds" % (time.time() - start2))
