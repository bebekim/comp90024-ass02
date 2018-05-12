import csv
import os

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
