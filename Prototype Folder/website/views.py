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

from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
from sqlalchemy.sql.expression import null
from .models import Note
from .models import Search
from . import db
import json

views = Blueprint('views', __name__)

code = ""

@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST':
        note = request.form.get('note')

        if len(note) < 1:
            flash('Note is too short!', category='error')
        else:
            new_note = Note(data=note, user_id=current_user.id)
            db.session.add(new_note)
            db.session.commit()
            flash('Note added!', category='success')

    return render_template("home.html", user=current_user)

@views.route('/delete-note', methods=['POST'])
def delete_note():
    note = json.loads(request.data)
    noteId = note['noteId']
    note = Note.query.get(noteId)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()

    return jsonify({})


@views.route('/search', methods=['GET'])
@login_required
def search_page():
    return render_template("search.html", user=current_user)

@views.route('/search', methods=['GET', 'POST'])
@login_required
def search_engine():
    if request.method == 'POST':
        code = request.form.get('code')
    i = 0
    adata = ["" for x in range(10)]
    if code != "":  
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

        Code_corpus= dataset.iloc[0:50, 6]
        taggedcode = [TaggedDocument(d, [i]) for i, d in enumerate(Code_corpus)]
        taggedcode

        model1 = Doc2Vec(taggedcode, vector_size = 20, window = 2, min_count = 1, epochs = 100)

        test_doc = word_tokenize(code.lower())
        test_doc_vector = model1.infer_vector(test_doc)
        temp = np.empty((0,0))
        temp = model1.dv.most_similar(positive = [test_doc_vector])
        print('')
        print (temp)
        print('')
        print("Top 10 most similar contract addessses in descending order:-")
        print('')
        for x in range(10):
            a = temp[x][0]
            adata[x] = dataset.iloc[a, 1]
            print(dataset.iloc[a, 1])
        i = i + 1
        print('')
        print("Time taken Doc2vec Model: %s seconds" % (time.time() - start2))
    print('')
    heading = ("Addresses in Descending Order of Similarity")

    if i == 1:
        for x in range(10):
            a = temp[x][0]
            new_address = Search(address=a, user_id=current_user.id)
            db.session.add(new_address)
            db.session.commit()
        flash('Searching Contracts !', category='success')

    return render_template("search.html", user=current_user, heading=heading, adata=adata)


