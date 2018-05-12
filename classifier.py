def classify(tweets, count_vect, clf):
    neutral_cnt = 0

    for t in tweets:
        predict_new = clf.predict(count_vect.transform([t['text_tokens']]))
        t['text_sentiment'] = predict_new.tolist()[0]
        if (t['text_sentiment'] == 'neutral'):
            neutral_cnt += 1

    print(neutral_cnt)
    
    return tweets
