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
from nltk.tokenize.casual import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

import numpy as np
import pandas as pd

from extract_data import prep_training
from split_train_classifier import train_model
from preprocess import preprocess
from classifier import classify
from write_data import write_tweets


def read_arguments(argv):
    input_dir = argv[0]
    output_dir = argv[1]
    return input_dir, output_dir


def get_filenames(input_dir, show_folder=True):
    if show_folder:
        txt_files = glob.glob(input_dir + '/*.txt')
    else:
        txt_files = os.listdir(os.getcwd() + '/' + input_dir)
    return txt_files


def attach_filenames(filenames, tag):
    files = []
    for f in filenames:
        f_split = list(os.path.splitext(f))
        if f_split[1] in ['.txt']:
            f_split[0] += tag
            f_split = ''.join(f_split)
            files.append(f_split)
    return files
            

def align_files(input_dir, output_dir):
    input_files = get_filenames(input_dir, show_folder=True)
    output_files = attach_filenames(input_files, tag='_output')
    zipped_files = zip(input_files, output_files)
    return zipped_files


def read_tweets(filename):
    # https://stackoverflow.com/questions/24754861/unicode-file-with-python-and-fileinput
    # fileinput.input(filename, openhook=fileinput.hook_encoded("utf-8")).
    # raw = url.read().decode('windows-1252')
    tweets = []
    with open(filename, 'r', encoding='utf-8') as tweet_data:
        for line in tweet_data:
            t = json.loads(line)
            tweets.append(t)
        return tweets

    
if __name__ == '__main__':
    # parse commandline arguments
    input_dir, output_dir = read_arguments(sys.argv[1:])
    zipped_io = align_files(input_dir, output_dir)

    # conduct supervised learning
    emotions_labelled_data = prep_training('./reference/text_emotion.csv')
    emotions_vect, emotions_classifier = train_model(emotions_labelled_data)

    for i, o in list(zipped_io):
        tweets = read_tweets(i)
        print(len(tweets))
        # t['text_tokenized'] = preprocess(t['text'])
        tweets = preprocess(tweets)
        tweets_classified = classify(tweets, emotions_vect, emotions_classifier)
        write_tweets(tweets, output_dir, o)    
