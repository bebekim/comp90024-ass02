import json
import pprint
import nltk
from nltk.stem import WordNetLemmatizer

tweets = []
geo_tweets = []


with open("Tweets2.txt") as f:
  for line in f:
    tweet = json.loads(line)
    tweets.append(tweet)

for tweet in tweets:
  if(tweet['geo'] is not None):
    t = {
      "coordinates": tweet['coordinates']['coordinates'],
      "date": tweet['created_at'],
      "text": tweet['text']
      }
    geo_tweets.append(t)
    

for i in geo_tweets:
  pprint.pprint(i)
  
  
def tokenize_text(text):
  sentences = nltk.sent_tokenize(text)
  word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
  return word_tokens



def remove_stopwords(tokens):
  stopword_list = nltk.corpus.stopwords.words('english')
  filtered_tokens = [token for token in tokens if token not in stopword_list]
  return filtered_tokens

wordnet_lemmatizer = WordNetLemmatizer()

# pick out the most common words per coordinate
#