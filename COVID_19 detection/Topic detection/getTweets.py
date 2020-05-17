import pandas as pd

# Getting Twitter data using the API

from CrawlTwitter import *

'''
status:
['contributors', 'truncated', 'text', 'is_quote_status', 'in_reply_to_status_id', 'id', 'favorite_count', '_api', 
'author', '_json', 'coordinates', 'entities', 'in_reply_to_screen_name', 'id_str', 'retweet_count', 
'in_reply_to_user_id', 'favorited', 'source_url', 'user', 'geo', 'in_reply_to_user_id_str', 'possibly_sensitive', 
'lang', 'created_at', 'in_reply_to_status_id_str', 'place', 'source', 'retweeted']
'''

search_words = ["coronavirus", "corona virus", "covid-19", "covid19", "Chinese virus", "Wuhan virus","Covid-19"]
search_rumors = ["Sipping Water","Hold breath","Holding Your Breath","Salt Water or Vinegar","Garlic Water","transmitted with hot and humid climates","Cold weather","Taking a hot bath","mosquito","antibiotics"]


def tweet2df(tweets):
    columns = ['id', 'text', 'favorite_count', 'retweet_count', 'lang', 'source', 'created_y', 'created_m', 'created_d', 'created_h', 'created_min', 'coordinates']
    data = [
        [tweet.id, tweet.full_text, tweet.favorite_count, tweet.retweet_count, tweet.lang, tweet.source,
         tweet.created_at.year, tweet.created_at.month, tweet.created_at.day, tweet.created_at.hour, tweet.created_at.minute, tweet.coordinates]
        for tweet in tweets]
    df = pd.DataFrame(data, columns=columns)
    return df


if __name__ == '__main__':

    output_path = "C:\\Users\\PHISSTOOD\\Desktop\\Machine Learning\\antibiotics_raw.csv"
    all_tweets = get_all_tweets("antibiotics")
    all_df = tweet2df(all_tweets)

    # transform the tweepy tweets into a 2D array that will populate the csv
    all_df.to_csv(output_path, index=False)

