import nltk
from sklearn.metrics.pairwise import euclidean_distances
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
import string
import re

input_txt = "I am learning NPL and using NLTK"
word_tokens = word_tokenize(input_txt)

print(input_txt)
print(word_tokens)

raw_data = open('C:/Users/SMSSpamCollection').read()
print(raw_data[0:500])

parsed_data = raw_data.replace('\t', '\n').split('\n')
print(parsed_data[0:10])

label_list = parsed_data[0::2]
msg_list = parsed_data[1::2]
print(label_list[0:5])
print(msg_list[0:5])

combined_df = pd.DataFrame({
    'label': label_list[:-1],
    'sms': msg_list
})
print(combined_df.head(5))

dataset = pd.read_csv('C:/Users/SMSSpamCollection', sep="\t", header=None)
dataset.head()
dataset.columns = ['label', 'sms']
print(dataset.head(5))

print(f'Input data has {len(dataset)} rows, {len(dataset.columns)} columns')
print(f'ham = {len(dataset[dataset["label"] == "ham"])}')
print(f'spam = {len(dataset[dataset["label"] == "spam"])}')
print(f"Number of missing label = {dataset['label'].isnull().sum()}")
print(f"Number of missing label = {dataset['sms'].isnull().sum()}")

pd.set_option('display.max_colwidth', 100)
data = pd.read_csv('C:/Users/SMSSpamCollection', sep="\t", header=None)
data.columns = ['label', 'msg']
print(data.head())
print(string.punctuation)

def remove_punctuation(txt):
    txt_nopunct = "".join([c for c in txt if c not in string.punctuation])
    return txt_nopunct

data['msg_clean'] = data['msg'].apply(lambda x: remove_punctuation(x))
print(data.head())

def tokenize(txt):
    tokens = re.split('\W+', txt)
    return tokens

data['msg_clean_tokenized'] = data['msg_clean'].apply(lambda x: tokenize(x.lower()))
print(data.head(5))

stopwords = nltk.corpus.stopwords.words('english')
print(stopwords[0:10])

def remove_stopwords(txt_tokenized):
    txt_clean = [word for word in txt_tokenized if word not in stopwords]
    return txt_clean

data['msg_no_sw'] = data['msg_clean_tokenized'].apply(lambda x: remove_stopwords(x))
print(data.head(5))

ps = PorterStemmer()
pd.set_option('display.max_colwidth', 100)
stopwords = nltk.corpus.stopwords.words('english')
data = pd.read_csv('C:/Users/SMSSpamCollection', sep="\t", header=None)
data.columns = ['label', 'msg']
print(data.head(5))

def clean_text(text):
    text = "".join([c for c in text if c not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

data['msg_nostop'] = data['msg'].apply(lambda x: clean_text(x.lower()))
print(data.head(5))

def stemming(tokenized_text):
    text = [ps.stem(word) for word in tokenized_text]
    return text

data['msg_stemmed'] = data['msg_nostop'].apply(lambda x: stemming(x))
print(data.head(5))

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
corpus = ["This is a sentence is",
          "This is another sentence",
          "third document is here"]

x = cv.fit(corpus)
print(x.vocabulary_)
print(cv.get_feature_names())

x = cv.transform(corpus)
#x = cv.fit_transform(corpus)
print(x.shape)
print(x.toarray())

df = pd.DataFrame(x.toarray(), columns = cv.get_feature_names())
print(df)

cv1 = CountVectorizer(analyzer = clean_text)
x = cv1.fit_transform(data['msg'])
print(x.shape)

data_sample = data[0:10]
cv2 = CountVectorizer(analyzer = clean_text)
x = cv2.fit_transform(data_sample['msg'])
print(x.shape)

df = pd.DataFrame(x.toarray(), columns=cv2.get_feature_names())
print(df.head(10))

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer()
corpus = ["This is a sentence is",
          "This is another sentence",
          "third document is here"]

x = tfidf_vect.fit(corpus)
print(x.vocabulary_)
print(tfidf_vect.get_feature_names())

x = tfidf_vect.transform(corpus)
#x = cv.fit_transform(corpus)
print(x.shape)
print(x.toarray())

df = pd.DataFrame(x.toarray(), columns = tfidf_vect.get_feature_names())
print(df)

data_sample = data[0:10]
tfidf2 = TfidfVectorizer(analyzer = clean_text)
x = tfidf2.fit_transform(data_sample['msg'])
print(x.shape)

df = pd.DataFrame(x.toarray(), columns=tfidf2.get_feature_names())
print(df.head(10))

data = pd.read_csv('C:/Users/SMSSpamCollection', sep="\t", header=None)
data.columns = ['label', 'msg']
data['msg_len'] = data['msg'].apply(lambda x: len(x))
print(data.head())

def punctuation_count(txt):
    count = sum([1 for c in txt if c in string.punctuation])
    return 100*count/(len(txt))

data['punctuatuion_%'] = data['msg'].apply(lambda x: punctuation_count(x))
print(data.head(5))

features = cv1.fit_transform(data['msg'])
print(features.shape)

for f in features:
    print(euclidean_distances(features[0], f))





#from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer()

#cv1 = CountVectorizer(analyzer = clean_text)
#x = cv1.fit_transform(dataset['Code_stemmed'])
#print(x)
#data_sample = dataset[0:10]
#cv2 = CountVectorizer(analyzer = clean_text)
#x = cv2.fit_transform(data_sample['Code_stemmed'])
#print(x.shape)

#df = pd.DataFrame(x.toarray(), columns=cv2.get_feature_names())
#print(df.head(10))

