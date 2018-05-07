# encoding: utf-8
import fileinput
import json
import pprint

import re
import string

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from collections import Counter

import numpy as np
import pandas as pd

from preprocess import preprocess
from clean_text import get_text_sanitized


def remove_characters_before_tokenization(sentence, keep_apostrophes=False):
    sentence = sentence.strip()
    if keep_apostrophes:
        PATTERN = r'[?|$|&|*|%|@|(|)|~]' # add other characters here to remove them
        filtered_sentence = re.sub(PATTERN, r'', sentence).lower()
    else:
        PATTERN = r'[^a-zA-Z0-9 ]' # only extract alpha-numeric characters
        filtered_sentence = re.sub(PATTERN, r'', sentence).lower()
    return filtered_sentence


def sanitize_content(text):
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
    return text


def nostop_content(text):
    text = ' '.join([word for word in text.split() if word not in stopword_list])
    return text.lower()


def lemmatize(word):
    lemma = lemmatizer.lemmatize(word, 'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word, 'n')
    return lemma


def lemmatize_content(text):
    tmp = []
    for word in text.split():
        tmp.append(lemmatize(word))
    return ' '.join(tmp)

def int64toint32(x):
    x = x.astype(np.int32)
    return x



def read_tweets(filenames):
    # https://stackoverflow.com/questions/24754861/unicode-file-with-python-and-fileinput
    # fileinput.input(filename, openhook=fileinput.hook_encoded("utf-8")).
    # raw = url.read().decode('windows-1252')
    tweets = []
    with fileinput.input((filenames)) as f:
        for line in f:
            tweet = json.loads(line)
            tweet['text_tokenized'] = preprocess(tweet['text'])
            tweets.append(tweet)
        return tweets
    

if __name__ == '__main__':
    tweets_emotion_file = './data/text_emotion.csv'
    features = ['tweet_id', 'sentiment', 'text']
    tweets_emotions = pd.read_csv(tweets_emotion_file)

    lemmatizer = WordNetLemmatizer()   
    stopword_list = nltk.corpus.stopwords.words("english")

    content_sanitizer = lambda x: sanitize_content(x)
    content_stopwordsremover = lambda x: nostop_content(x)
    content_lemmatizer = lambda x: lemmatize_content(x)

    tweets_emotions['sanitized_content'] = tweets_emotions['content'].apply(content_sanitizer)
    tweets_emotions['sanitized_content'] = tweets_emotions['sanitized_content'].apply(content_stopwordsremover)
    tweets_emotions['lemmatized_content'] = tweets_emotions['sanitized_content'].apply(content_lemmatizer)

    X = tweets_emotions.lemmatized_content
    y = tweets_emotions.sentiment

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

    count_vect = CountVectorizer()
    X_train_bow = count_vect.fit(X_train)
    X_train_bow = count_vect.transform(X_train)
    X_test_bow = count_vect.transform(X_test)

    tfidf_transformer = TfidfTransformer(use_idf=False).fit(X_train_bow)
    X_train_tfidf = tfidf_transformer.transform(X_train_bow)

    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    y_predict_class = clf.predict(X_test_bow)

    filenames = ['data/MelbourneTweets0.txt', 
                 'data/MelbourneTweets2.txt']
    tweets = read_tweets(filenames)

    # handles @mention, make lowercase
    tknzr = nltk.tokenize.casual.TweetTokenizer(preserve_case=False, 
                                                strip_handles=True, 
                                                reduce_len=True)
    regexp_hashtag = re.compile(r'(?:\A|\s)#([a-z]{1,})(?:\Z|\s)')
    regexp_url = re.compile(r"http\S+")

    for t in tweets:
        tokenized = tknzr.tokenize(t['text'])
        not_a_stopword = []
        for word in tokenized:
            word = lemmatize(word)
            if word not in stopword_list:
                not_a_stopword.append(word)
        t['text_tokens'] = ' '.join(not_a_stopword)
        predict_new = clf.predict(count_vect.transform([t['text_tokens']]))
        t.pop('text_tokenized', None)
        t.pop('text_tokens', None)
        t['text_sentiment'] = predict_new.tolist()[0]


    tweets_json = json.dumps(tweets)

    f = open("output.json", "w")
    f.write(tweets_json)
    f.close()
