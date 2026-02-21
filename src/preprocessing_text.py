import re
import string

def clean_text(text):
    """
    Cleans text by removing special characters, punctuation, and converting to lowercase.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLS
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user @ references and '#' from tweet text
    text = re.sub(r'\@\w+|\#','', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra spaces
    text = " ".join(text.split())
    
    return text