import csv
import os
from nltk.corpus import stopwords
from random import shuffle

def ReadCSVasDict(csv_file):
    try:
        with open(csv_file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                print row['Row'], row['Name'], row['Country']
    except IOError as (errno, strerror):
            print("I/O error({0}): {1}".format(errno, strerror))    
    return

currentPath = os.getcwd()
csv_file = currentPath + "/data/text_emotion.csv"

ReadCSVasDict(csv_file)


# stops = set(stopwords.words('english'))

# positive_tweets = nltk.corpus.twitter_samples.tokenized("positive_tweets.json")
# negative_tweets = nltk.corpus.twitter_samples.tokenized("negative_tweets.json")

# def removestopwords(tokens, stops):
#     r = [i for i in tokens if i.lower() not in stops and 
#                                         not re.search(r'[^a-zA-Z]', i)]
# nn    return r

# positive_tweets = [removestopwords(tweet, stops) for tweet in positive_tweets]
# negative_tweets = [removestopwords(tweet, stops) for tweet in negative_tweets]

# shuffle(positive_tweets)
# shuffle(negative_tweets)

# p1 = int(len(positive_tweets)*0.8)
# n1 = int(len(negative_tweets)*0.8)
# p2 = p1 + int(len(positive_tweets)*0.1)
# n2 = n1 + int(len(negative_tweets)*0.1)
             
# training = positive_tweets[:p1] + negative_tweets[:n1]
# development = positive_tweets[p1:p2] + negative_tweets[n1:n2]
# testing = positive_tweets[p2:] + negative_tweets[n2:]

# shuffle(training)
# shuffle(development)
# shuffle(testing)


# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.metrics import accuracy_score
# from sklearn import cross_validation 

# def get_BOW(text):
#     BOW = {}
#     for word in text:
#         BOW[word] = BOW.get(word,0) + 1
#     return BOW
    
# def prepare_tweets_data(tweets, feature_extractor):
#     feature_matrix = []
#     classifications = []
#     for tweet in tweets:
#         feature_dict = feature_extractor(tweet)
#         feature_matrix.append(feature_dict)
#         if tweet in positive_tweets:
#             classifications.append('positive')
#         else:
#             classifications.append('negative')
    
#     vectorizer = DictVectorizer()
#     dataset = vectorizer.fit_transform(feature_matrix)
#     return dataset, classifications

# def check_accuracy(model, predictions, classifications):
#     print "\n"+model+" accuracy"
#     print accuracy_score(classifications,predictions)

# dataset, classifications = prepare_tweets_data(development, get_BOW)
# n_to_test = range(5,25)
# clfs = [MultinomialNB(n, True, [0.52, 0.5]) for n in n_to_test]
# for clf in clfs:
#     predictions = cross_validation.cross_val_predict(clf, dataset, classifications, cv=10)
#     check_accuracy("naive bayes", predictions, classifications)
#     print(clf.get_params())

# n_to_test = range(1,10)
# clfs = [LogisticRegression(C=n/float(10)) for n in n_to_test]
# for clf in clfs:
#     predictions = cross_validation.cross_val_predict(clf, dataset, classifications, cv=10)
#     check_accuracy("logistic regression", predictions, classifications)
#     print(clf.get_params())
