import os
import json

def write_tweets(tweets, output_dir, output_file):
    output_path = os.path.join(os.getcwd(), output_dir)
    tweets_json = json.dumps(tweets)

    f = open(os.path.join(os.getcwd(), output_dir, output_file), "w")
    f.write(tweets_json)
    f.close()

