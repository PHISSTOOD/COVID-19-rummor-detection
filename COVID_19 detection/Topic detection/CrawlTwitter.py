import tweepy


consumer_key = '************'
consumer_secret = '************'
access_token = '************'
access_token_secret = '************'

# authorize twitter, initialize tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

search_words = ["coronavirus", "corona virus", "covid-19", "covid19", "Chinese virus", "Wuhan virus"]
search_rumors = ["Sipping Water Every 15 Minutes","Holding Your Breath","Salt Water or Vinegar","Garlic Water","transmitted with hot and humid climates","Cold weather","Taking a hot bath","mosquito","antibiotics"]
date_since = "2020-01-01"  # YYYY-MM-DD



status = ""
def get_all_tweets(St):
    status = St
    users_locs = []
    for x in search_words:
        for tweet in tweepy.Cursor(api.search, q=St, lang='en', since=date_since,tweet_mode='extended').items(10000):
            try:
                if 'retweeted_status' in tweet._json:
                    continue
                else:
                    if 'RT @' not in tweet.full_text and x in tweet.full_text:
                        text = str(tweet.full_text)
                        print(text)
                        users_locs.append(tweet)
            except tweepy.TweepError as e:
                print(e.reason)
                continue
            except StopIteration:  # stop iteration when last tweet is reached
                break
    return users_locs


if __name__ == '__main__':
	get_all_tweets()
