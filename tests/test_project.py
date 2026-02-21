import os
import sys
import pytest
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import clean_text
from train import train_model

def test_clean_text():
    raw_text = "Check out this link: http://example.com @user #hashtag"
    expected = "check out this link hashtag"
    assert clean_text(raw_text) == expected

def test_clean_text_punctuation():
    raw_text = "Hello!!! How are you?"
    expected = "hello how are you"
    assert clean_text(raw_text) == expected


