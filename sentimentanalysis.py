import fileinput
import json
import pprint
import re, string
import nltk
from nltk.stem import WordNetLemmatizer

# https://docs.python.org/3.6/library/fileinput.htmld
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
    
# for cleansing part of tweet, I have used the script from following source
# https://gist.github.com/timothyrenner/dd487b9fd8081530509c
# -----------------------------------------------------------------------------

#Gets the tweet time.
def get_time(tweet):
  return datetime.strptime(tweet['created_at'], "%a %b %d %H:%M:%S +0000 %Y")

#Gets all hashtags.
def get_hashtags(tweet):
  return [tag['text'] for tag in tweet['entities']['hashtags']]

#Gets the screen names of any user mentions.
def get_user_mentions(tweet):
  return [m['screen_name'] for m in tweet['entities']['user_mentions']]

#Gets the text, sans links, hashtags, mentions, media, and symbols.
def get_text_cleaned(tweet):
  text = tweet['text']
  slices = []
  #Strip out the urls.
  if 'urls' in tweet['entities']:
    for url in tweet['entities']['urls']:
      slices += [{'start': url['indices'][0], 'stop': url['indices'][1]}]
    
  #Strip out the hashtags.
  if 'hashtags' in tweet['entities']:
    for tag in tweet['entities']['hashtags']:
      slices += [{'start': tag['indices'][0], 'stop': tag['indices'][1]}]
    
    #Strip out the user mentions.
  if 'user_mentions' in tweet['entities']:
    for men in tweet['entities']['user_mentions']:
      slices += [{'start': men['indices'][0], 'stop': men['indices'][1]}]
    
    #Strip out the media.
  if 'media' in tweet['entities']:
    for med in tweet['entities']['media']:
      slices += [{'start': med['indices'][0], 'stop': med['indices'][1]}]
    
    #Strip out the symbols.
  if 'symbols' in tweet['entities']:
    for sym in tweet['entities']['symbols']:
      slices += [{'start': sym['indices'][0], 'stop': sym['indices'][1]}]
    
    # Sort the slices from highest start to lowest.
  slices = sorted(slices, key=lambda x: -x['start'])
    
    #No offsets, since we're sorted from highest to lowest.
  for s in slices:
    text = text[:s['start']] + text[s['stop']:]
  return text
# -----------------------------------------------------------------------------


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

  filenames = ['data/MelbourneTweets0.txt',
               'data/MelbourneTweets2.txt']
  tweets = read_file(filenames)
  tweet_texts = extract_text(tweets)
  tweet_cleaned = [get_text_cleaned(tweet) for tweet in tweets]
  
