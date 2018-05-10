# encoding: utf-8
import sys
import getopt
import fileinput
import json
import pprint
import glob
import os
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


def read_arguments(argv):
    input_dir = argv[0]
    output_dir = argv[1]

    # try:
    #     opts, args = getopt.getopt(argv[1:], 'io:')
    # except getopt.GetoptError as error:
    #     print(error)
    #     sys.exit()
        
    # for opt, arg in opts:
    #     if opt in ("-i"):
    #         input_dir = arg
    #     if opt in ("-o"):
    #         output_dir = arg
    #     else:
    #         print('Invalid option')
    #         sys.exit()
    return input_dir, output_dir

    
def read_tweets(input_dir):
    # https://stackoverflow.com/questions/24754861/unicode-file-with-python-and-fileinput
    # fileinput.input(filename, openhook=fileinput.hook_encoded("utf-8")).
    # raw = url.read().decode('windows-1252')
    all_tweets = []
    txt_names = os.listdir(input_dir)
    txt_files = glob.glob(input_dir + '/*.txt')
    for file in txt_files:
        with open(file, 'r', encoding='utf-8') as tweet_data:
            tweets = []
            for line in tweet_data:
                t = json.loads(line)
                t['text_tokenized'] = preprocess(t['text'])
                tweets.append(t)
        all_tweets.extend(tweets)
    return all_tweets


if __name__ == '__main__':
    input_dir, output_dir = read_arguments(sys.argv[1:])
    tweets_emotion_file = './reference/text_emotion.csv'
    features = ['tweet_id', 'sentiment', 'text']
    stopword_list = nltk.corpus.stopwords.words("english")
    tweets_emotions = pd.read_csv(tweets_emotion_file)

    lemmatizer = WordNetLemmatizer()   

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

    # file_path = 'data'
    # tweets = read_tweets(file_path)
    tweets = read_tweets(input_dir)

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

    print(input_dir)
    print(output_dir)
    output_dir = os.path.join(os.getcwd(), output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    f = open(output_dir + '/output.json', "w")
    f.write(tweets_json)
    f.close()
