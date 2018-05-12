import re
import nltk
from nltk.tokenize.casual import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

lemmatizer = WordNetLemmatizer()   

def lemmatize(word):
    lemma = lemmatizer.lemmatize(word, 'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word, 'n')
    return lemma


# list of dictionaries coming in
def preprocess(tweets):
    stopwords_list = nltk.corpus.stopwords.words("english")
    # stopwords_list = stopwords.words('english')
    tknzr = nltk.tokenize.casual.TweetTokenizer(preserve_case=False, 
                                            strip_handles=True, 
                                            reduce_len=True)
    for t in tweets:
        not_a_stopword = []
        tokenized = tknzr.tokenize(t['text'].lower())
        for word in tokenized:
            word = lemmatize(word)
            if word not in stopwords_list:
                not_a_stopword.append(word)
        t['text_tokens'] = ' '.join(not_a_stopword)

    return tweets

    #     tokens = tokenize(s)
    # if lowercase:
    #     tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    # return tokens


# for t in tweets:
#     tokenized = tknzr.tokenize(t['text'])
#     not_a_stopword = []
#     for word in tokenized:
#         word = lemmatize(word)
#         if word not in stopword_list:
#             not_a_stopword.append(word)
#     t['text_tokens'] = ' '.join(not_a_stopword)
#     predict_new = clf.predict(count_vect.transform([t['text_tokens']]))
#     t.pop('text_tokenized', None)
#     t.pop('text_tokens', None)
#     t['text_sentiment'] = predict_new.tolist()[0]

# tweets_json = json.dumps(tweets)


    
# tweet = 'RT @marcobonzanini: just an example! :D http://example.com #NLP'
# print(preprocess(tweet))
# ['RT', '@marcobonzanini', ':', 'just', 'an', 'example', '!', ':D', 'http://example.com', '#NLP']
