# Twitter sentiment bot
# This program uses NLP (Natural Langauge Processing) and tokenisation to determine
# whether the topic/text is happy or sad

import tweepy
import os
from textblob import TextBlob



def extract_api_keys(file_path):
    keys = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip() # remove whitespace/newlines

            # skip empty lines
            if not line:
                continue

            if '=' in line:
                key, value = line.split('=', 1)
                keys[key.strip()] = value.strip()
    return keys

# Authentication
api_keys = extract_api_keys(os.path.join('Tutorials', 'twitter key.txt')) # 'twitter key.txt' contains all of my API keys. Replace the directory with where you store your keys
API_KEY = api_keys['API']
SECRET_API_KEY = api_keys['SECRET_API']
ACCESS_TOKEN = api_keys['ACCESS_TOKEN']
SECRET_ACCESS_TOKEN = api_keys['SECRET_ACCESS_TOKEN']

auth = tweepy.OAuthHandler(API_KEY, SECRET_API_KEY)
auth.set_access_token(ACCESS_TOKEN, SECRET_ACCESS_TOKEN)

# Start connecting to the API

api = tweepy.API(auth)

tweets = api.search_tweets('Trump')

for tweet in tweets:
    text = tweet.text
    print(text)
    analysis = TextBlob(text)
    print(analysis.sentiment)
