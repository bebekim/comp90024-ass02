import csv
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from random import shuffle



def csv2listdict(file):
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        ls = []
        for row in reader:
            obj = {}
            obj['tweet_id'] = row['tweet_id']
            obj['sentiment'] = row['sentiment']
            obj['text'] = row['content']
            ls.append(obj)
        return ls

# current_path = os.getcwd()
# tweets_emotion_file = current_path + '/data/text_emotion.csv'
# tweets_emotion = csv2listdict(tweets_emotion_file)

# regex_hashtag = re.compile(r'(?:\A|\s)#([a-z]{1,})(?:\Z|\s)')
# remove_hashtag(tweets_emotion, regex_hashtag)
# print(tweets_emotion[32694]['text']
                        
# tknzr = nltk.tokenize.casual.TweetTokenizer(preserve_case=False,
#                                             strip_handles=True, reduce_len=True)
#     regex_hashtag = re.compile(r'(?:\A|\s)#([a-z]{1,})(?:\Z|\s)')

#     remove_hashtag(tweets_emotion, regex_hashtag)
