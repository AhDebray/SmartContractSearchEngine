import nltk
from sklearn.metrics.pairwise import euclidean_distances
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
import string
import re

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

def tokenize(txt):
    tokens = re.split('\W+', txt)
    return tokens

data['Code_clean_tokenized'] = data['Code_clean'].apply(lambda x: tokenize(x.lower()))
data['Comment_clean_tokenized'] = data['Comment_clean'].apply(lambda x: tokenize(x.lower()))
print(data.head(5))

stopwords = nltk.corpus.stopwords.words('english')
print(stopwords[0:10])

def remove_stopwords(txt_tokenized):
    txt_clean = [word for word in txt_tokenized if word not in stopwords]
    return txt_clean

data['Code_no_sw'] = data['Code_clean_tokenized'].apply(lambda x: remove_stopwords(x))
data['Comment_no_sw'] = data['Comment_clean_tokenized'].apply(lambda x: remove_stopwords(x))
print(data.head(5))


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

##Bag Of Words
##codeDoc = dataset['Code_stemmed']
##commentDoc = dataset['Code_stemmed']
##print(codeDoc[0:5])
##print(commentDoc[0:5])

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

cv1 = CountVectorizer(analyzer = clean_text)
x = cv1.fit_transform(dataset['Code_stemmed'])
print(x)
data_sample = dataset[0:10]
cv2 = CountVectorizer(analyzer = clean_text)
x = cv2.fit_transform(data_sample['Code_stemmed'])
print(x.shape)

df = pd.DataFrame(x.toarray(), columns=cv2.get_feature_names())
print(df.head(10))

##1
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(analyzer = clean_text)

data_sample = dataset
x = tfidf.fit_transform(dataset['Code_stemmed'])
df = pd.DataFrame(x.toarray(), columns=tfidf.get_feature_names())
print(df.head(10))



