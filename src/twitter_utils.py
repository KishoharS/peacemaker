import asyncio
from twikit import Client
import os
import json

# Initialize Twikit Client
client = Client('en-US')

async def get_twitter_feed(username, max_tweets=5):
    """
    Extracts tweets and media URLs from a specific X profile for Peacemaker analysis.
    """
    # 1. Credentials (BEST PRACTICE: Use environment variables)
    TW_USER = "@kishohar126521"
    TW_PASS = "bigseq-memxi2-saznUf"
    TW_EMAIL = "your_burner_email" 
    
    COOKIE_PATH = 'twitter_cookies.json'

    try:
        # 2. Login Logic
        if os.path.exists(COOKIE_PATH):
            client.load_cookies(COOKIE_PATH)
        else:
            await client.login(
                auth_info_1=TW_USER,
                auth_info_2=TW_EMAIL,
                password=TW_PASS
            )
            client.save_cookies(COOKIE_PATH)

        # 3. Fetch User and Tweets
        user = await client.get_user_by_screen_name(username)
        tweets = await client.get_user_tweets(user.id, 'Tweets', count=max_tweets)

        results = []
        for tweet in tweets:
            # Extract text and all media (images/videos)
            media_urls = []
            if tweet.media:
                media_urls = [m['media_url_https'] for m in tweet.media]

            results.append({
                'text': tweet.text,
                'image': media_urls[0] if media_urls else None, # First image for Peacemaker
                'all_media': media_urls,
                'is_mock': False
            })
        
        return results

    except Exception as e:
        print(f"Twitter Extraction Error: {e}")
        return []

# Helper to run async function in a sync environment like Streamlit
def get_twitter_feed_sync(username, max_tweets=5):
    return asyncio.run(get_twitter_feed(username, max_tweets))