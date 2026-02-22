import instaloader

def get_instagram_captions(username, max_posts=30):
    """
    Fetches captions and image URLs from a public Instagram profile.
    Returns a list of dicts: {'text': str, 'image': str, 'shortcode': str}
    """
    L = instaloader.Instaloader()
    posts_data = []
    
    try:
        print(f"Attempting to fetch posts for {username}...")
        try:
            profile = instaloader.Profile.from_username(L.context, username)
        except instaloader.ProfileNotExistsException:
             print(f"Profile {username} does not exist.")
             return []
        except instaloader.ConnectionException:
             print("Connection error. Instagram might be blocking requests.")
             return []
        
        count = 0
        for post in profile.get_posts():
            # We want both text (caption) and image (url)
            caption = post.caption if post.caption else ""
            image_url = post.url
            
            posts_data.append({
                'text': caption,
                'image': image_url,
                'shortcode': post.shortcode
            })
            
            count += 1
            if count >= max_posts:
                break
        
        if not posts_data:
            print("No posts found or profile is private.")
            
        print(f"Successfully fetched {len(posts_data)} posts.")
        return posts_data

    except Exception as e:
        print(f"Error fetching Instagram data: {e}")
        return []