import fileinput
import json
from clean_text import get_text_cleaned
import pprint
import nltk
from nltk.stem import WordNetLemmatizer


def read_file(filenames):
    with fileinput.input((filenames)) as f:
        for line in f:
            tweet = json.loads(line)
            tweets.append(tweet)
        return tweets


def extract_geo():
    for tweet in tweets:
        if(tweet['geo'] is not None):
            t = {
                "coordinates": tweet['coordinates']['coordinates'],
                "date": tweet['created_at'],
                "text": tweet['text']
                }
            geo_tweets.append(t)
    return geo_tweets


def extract_text(tweets):
    texts = [tweet['text'] for tweet in tweets if tweet['text'] is not None]
    return texts


def tokenize_text(text):
    sentences = nltk.sent_tokenize(text)
    word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    return word_tokens


def remove_stopwords(tokens):
    stopword_list = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens


# pick out the most common words per coordinate
#
if __name__ == '__main__':
    tweets = []
    tweet_texts = []
    wordnet_lemmatizer = WordNetLemmatizer()
    
    filenames = ['data/MelbourneTweets0.txt', 'data/MelbourneTweets2.txt']
               
    tweets = read_file(filenames)
    tweet_texts = extract_text(tweets)
    tweet_cleaned = [get_text_cleaned(tweet) for tweet in tweets]
  