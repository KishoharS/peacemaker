import instaloader
import os
import requests
from PIL import Image
from io import BytesIO

SESSION_DIR = "sessions"


def get_loader(username: str, password: str = None):
    """
    Load existing session or create new one.
    Only needs login once - session saved to disk after that.
    """
    L = instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        save_metadata=False,
        quiet=True
    )

    session_file = os.path.join(SESSION_DIR, username)
    os.makedirs(SESSION_DIR, exist_ok=True)

    if os.path.exists(session_file):
        L.load_session_from_file(username, session_file)
    elif password:
        L.login(username, password)
        L.save_session_to_file(session_file)
    else:
        raise Exception("No session found and no password provided.")

    return L


def get_instagram_posts(target_username: str, loader: instaloader.Instaloader, max_posts: int = 30):
    """
    Fetches captions and images from an Instagram profile.
    Requires an authenticated loader from get_loader().
    Returns list of dicts: {'text': str, 'image': PIL.Image or None, 'shortcode': str}
    """
    posts_data = []

    try:
        print(f"Attempting to fetch posts for {target_username}...")
        profile = instaloader.Profile.from_username(loader.context, target_username)
    except instaloader.exceptions.ProfileNotExistsException:
        print(f"Profile '{target_username}' does not exist.")
        return []
    except instaloader.exceptions.LoginRequiredException:
        print("Login required to view this profile.")
        return []
    except instaloader.exceptions.ConnectionException:
        print("Connection error. Instagram might be blocking requests.")
        return []

    count = 0
    for post in profile.get_posts():
        caption = post.caption if post.caption else ""
        image = None

        try:
            response = requests.get(post.url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            print(f"Could not fetch image for post {post.shortcode}: {e}")

        posts_data.append({
            "text": caption,
            "image": image,
            "shortcode": post.shortcode
        })

        count += 1
        if count >= max_posts:
            break

    if not posts_data:
        print("No posts found or profile is private.")
    else:
        print(f"Successfully fetched {len(posts_data)} posts.")

    return posts_data