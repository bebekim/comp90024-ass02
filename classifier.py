def classify(tweets, count_vect, clf):
    for t in tweets:
        predict_new = clf.predict(count_vect.transform([t['text_tokens']]))
        t['text_sentiment'] = predict_new.tolist()[0]
    return tweets
