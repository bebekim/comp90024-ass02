import fileinput
import json
import pprint
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# https://docs.python.org/3.6/library/fileinput.html
def read_file():
  with fileinput.input(files=('Tweets2.txt')) as f:
    for line in f:
      tweet = json.loads(line)
      tweets.append(tweet)

def extract_geo():
  for tweet in tweets:
    if(tweet['geo'] is not None):
      t = {
        "coordinates": tweet['coordinates']['coordinates'],
        "date": tweet['created_at'],
        "text": tweet['text']
        }
      geo_tweets.append(t)
  
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
  geo_tweets = []
  
  
  wordnet_lemmatizer = WordNetLemmatizer()

