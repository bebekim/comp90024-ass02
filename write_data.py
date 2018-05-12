import os
import json

def write_tweets(tweets, output_dir, output_file):
    output_path = os.path.join(os.getcwd(), output_dir)
    tweets_json = json.dumps(tweets)

    if not os.path.exists(output_dir):
       os.makedirs(output_dir)

    f = open(output_file, "w")
    f.write(tweets_json)
    f.close()

