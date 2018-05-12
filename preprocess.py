import re
from nltk.tokenize.casual import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# emoticons_str = r"""
#     (?:
#         [:=;] # Eyes
#         [oO\-]? # Nose (optional)
#         [D\)\]\(\]/\\OpP] # Mouth
#     )"""
 
# regex_str = [
#     emoticons_str,
#     r'<[^>]+>', # HTML tags
#     r'(?:@[\w_]+)', # @-mentions
#     r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
#     r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
#     r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
#     r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
#     r'(?:[\w_]+)', # other words
#     r'(?:\S)' # anything else
# ]

# tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
# emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

# def tokenize(s):
#     return tokens_re.findall(s)
 

# def remove_characters_before_tokenization(sentence, keep_apostrophes=False):
#     sentence = sentence.strip()
#     if keep_apostrophes:
#         PATTERN = r'[?|$|&|*|%|@|(|)|~]' # add other characters here to remove them
#         filtered_sentence = re.sub(PATTERN, r'', sentence).lower()
#     else:
#         PATTERN = r'[^a-zA-Z0-9 ]' # only extract alpha-numeric characters
#         filtered_sentence = re.sub(PATTERN, r'', sentence).lower()
#     return filtered_sentence


# def sanitize_content(text):
#     text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
#     return text


# def nostop_content(text):
#     text = ' '.join([word for word in text.split() if word not in stopword_list])
#     return text.lower()




# def lemmatize_content(text):
#     tmp = []
#     for word in text.split():
#         tmp.append(lemmatize(word))
#     return ' '.join(tmp)

# def preprocess(s, lowercase=False):
#     tokens = tokenize(s)
#     if lowercase:
#         tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
#     return tokens


# preprocessing and tagging is mixed together
# untangle it.
# handles @mention, make lowercase
tknzr = nltk.tokenize.casual.TweetTokenizer(preserve_case=False, 
                                            strip_handles=True, 
                                            reduce_len=True)

    # regexp_hashtag = re.compile(r'(?:\A|\s)#([a-z]{1,})(?:\Z|\s)')
# regexp_url = re.compile(r"http\S+")

def lemmatize(word):
    lemma = lemmatizer.lemmatize(word, 'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word, 'n')
    return lemma


# list of dictionaries coming in
def preprocess(tokenizer, tweets, lowercase, stopword_list):
    lemmatizer = WordNetLemmatizer()
    for t in tweets:
        not_a_stopword = []
        tokenized = tokenizer.tokenize(t['text'])
        for word in tokenized:
            word = lemmatize(word)
            if word not in stopword_list:
                not_a_stopword.append(word)
        t['text_tokens'] = ' '.join(not_a_stopword)

        tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


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
