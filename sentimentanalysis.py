import fileinput
import json
import nltk
from nltk.stem import WordNetLemmatizer
import re
import string
from collections import Counter
from contraction import CONTRACTION_MAP


def read_file(filenames):
    with fileinput.input((filenames)) as f:
        for line in f:
            tweet = json.loads(line)
            tweets.append(tweet)
        return tweets


# def extract_geo():
#     for tweet in tweets:
#         if(tweet['geo'] is not None):
#             t = {
#                 "coordinates": tweet['coordinates']['coordinates'],
#                 "date": tweet['created_at'],
#                 "text": tweet['text']
#                 }
#             geo_tweets.append(t)a
#     return geo_tweets


def lemmatize(word):
    lemma = lemmatizer.lemmatize(word, 'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word, 'n')
    return lemma

# pick out the most common words per coordinate
#
if __name__ == '__main__':
    tweets = []
    filenames = ['data/MelbourneTweets0.txt', 'data/MelbourneTweets2.txt']

    tweets = read_file(filenames)
    tweets_minus_url = [re.sub(r'http\S+', '', tweet['text']) for tweet in tweets]
    tweets_minus_rt = [re.sub(r'RT', '', tweet) for tweet in tweets_minus_url]
    tweets_minus_escape = [re.sub(r'\n', '', tweet) for tweet in tweets_minus_rt]

    # handles @mention, make lowercase
    tknzr = nltk.tokenize.casual.TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweets_tokenized = [tknzr.tokenize(tweet) for tweet in tweets_minus_escape]

    lemmatizer = WordNetLemmatizer()
    tweets_minus_stop = []
    stopword_list = nltk.corpus.stopwords.words('english')
    for tweet in tweets_tokenized:
        for word in tweet:
            word = lemmatize(word)
        x = [x for x in tweet if x not in stopword_list]
        tweets_minus_stop.append(x)
    
    # remove punctuations
    tweets_final = []
    for tweet in tweets_minus_stop:
        tweet = [''.join(c for c in s if c not in string.punctuation) for s in tweet]
        tweet = [t for t in tweet if t]
        tweets_final.append(tweet)

    token_counter = Counter(x for xs in tweets_final for x in set(xs))
    print(token_counter.most_common(30))


    # don't throw away the hashtags they might be of value later on
    # tweets_cleaned = [get_text_cleaned(tweet) for tweet in tweets]
    # tweets_removed = get_text_removed(tweets_cleaned)
    # contraction
    # tweets_tokenized = [tokenize_text(tweet) for tweet in tweets_removed]
    # tweets_tokenized = tokenize_text(tweets_removed)
    # nltk.download('wordnet')
    # tweets_lemmatized = [lemmatize_text(tweet) for tweet in tweets_tokenized]
    
    # nltk.download('stopwords')
    # tweet_stopremoved = remove_stopwords(tweet_tokenized)

