import os
import sys
import tempfile
from io import BytesIO
import asyncio
import re
import random
import importlib

import streamlit as st
import pandas as pd
import requests

import torch
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification, 
    ViTImageProcessor, 
    ViTForImageClassification
)
import whisper
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing_text import clean_text

from instagram_utils import get_instagram_captions
from twitter_utils import get_twitter_feed
from scrapping import TelegramClient, ChannelManager, join_channel, scrape_messages



# Load all models
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'bert_model')
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        return None, None

@st.cache_resource
def load_image_model():
    # Try custom trained model first
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'vit_model')
    if not os.path.exists(model_path):
        return None, None
        
    try:
        processor = ViTImageProcessor.from_pretrained(model_path)
        model = ViTForImageClassification.from_pretrained(model_path)
        return processor, model
    except Exception as e:
        st.error(f"Error loading image model: {e}")
        return None, None

@st.cache_resource
def load_audio_model():
    try:
        # Load Whisper model (base English model is good balance)
        model = whisper.load_model("base")
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

tokenizer, text_model = load_model()
img_processor, img_model = load_image_model()
audio_model = load_audio_model()

# helper function for prediction
def predict_toxicity(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = text_model(**inputs).logits
    probabilities = torch.softmax(logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    return prediction, probabilities[0]

def predict_image(image):
    inputs = img_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = img_model(**inputs).logits
    probabilities = torch.softmax(logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    return prediction, probabilities[0]

st.title("🕊️ Peacemaker: Cyberbullying Detection System")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Text Analyzer", "Instagram", "Twitter", "Telegram", "Image Analyzer", "Audio Analyzer"])

with tab1:
    st.header("Analyze Text/Comments")
    st.write("Paste your comments below to see whether you're being cyberbullied or not!!.")

    text_input = st.text_area("Input Text", height=150)

    if st.button("Analyze Text"):
        if not tokenizer or not text_model:
            st.error("Model artifacts not found. Please train the model and download artifacts to models/bert_model.")
        elif text_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            # Preprocess
            cleaned_text = clean_text(text_input)
            
            # Predict
            try:
                prediction, probabilities = predict_toxicity(cleaned_text)
                
                if prediction == 1:
                    st.error("⚠️ Application Detected: **Cyberbullying**")
                    st.write(f"Confidence: {probabilities[1]*100:.2f}%")
                else:
                    st.success("✅ Application Detected: **Non-Cyberbullying**")
                    st.write(f"Confidence: {probabilities[0]*100:.2f}%")
            except Exception as e:
                st.error(f"Error during prediction: {e}")


with tab2:
    st.header("Instagram Profile Analyzer")
    st.markdown("""
    **Note:** This feature analyzes the public posts, captions of a user to determine if their content is toxic.
    """)
    
    username = st.text_input("Enter Instagram Username (e.g. ig_kishohar)")
    
    if st.button("Analyze Profile"):
        if not username:
            st.warning("Please enter a username.")
        elif not tokenizer or not text_model:
            st.error("Model not loaded.")
        else:
            with st.spinner(f"Fetching posts for {username}..."):
                captions = get_instagram_captions(username)
                
            if not captions:
                st.info("No captions found to analyze.")
            else:
                st.write(f"Analyzed {len(captions)} recent posts.")
                
                toxic_count = 0
                for post in captions:
                    # Handle new dict structure
                    if isinstance(post, dict):
                        text_content = post.get('text', '')
                        image_url = post.get('image')
                    else:
                        text_content = str(post)
                        image_url = None

                    # Text Analysis
                    cleaned = clean_text(text_content)
                    pred, _ = predict_toxicity(cleaned)
                    
                    # Image Analysis (if available and model loaded)
                    img_pred = 0
                    if image_url and img_processor and img_model:
                        try:
                            # Verify if it's a valid URL or skip
                            if image_url.startswith('http'):
                                response = requests.get(image_url, timeout=5)
                                if response.status_code == 200:
                                    img = Image.open(BytesIO(response.content)).convert("RGB")
                                    img_pred, _ = predict_image(img)
                        except Exception as e:
                            # print(f"Image analysis failed: {e}")
                            pass

                    # Combined toxicity (if either is toxic)
                    if pred == 1 or img_pred == 1:
                        toxic_count += 1
                        
                toxicity_score = (toxic_count / len(captions)) * 100
                
                st.metric("Toxicity Score", f"{toxicity_score:.1f}%")
                
                if toxicity_score > 50:
                    st.error(f"⚠️ High Risk: This profile has high indications of toxic behavior ({toxic_count}/{len(captions)} toxic posts).")
                elif toxicity_score > 0:
                    st.warning(f"⚠️ Moderate Risk: Some toxic content detected ({toxic_count}/{len(captions)} posts).")
                else:
                    st.success("✅ Safe Profile: No toxic content detected in recent posts.")
                    
                with st.expander("View Analyzed Posts"):
                    for post in captions:
                        if isinstance(post, dict):
                            text = post.get('text', '')
                            img_url = post.get('image')
                        else:
                            text = str(post)
                            img_url = None
                            
                        st.text(f"{text[:100]}...")
                        if img_url:
                            st.write(f"[Image]({img_url})")

with tab3:
    st.header("X Profile Analyzer")
    st.markdown("""
    **Note:** This feature analyzes recent tweets of a user to determine if their content is toxic.
    """)
    
    tw_username = st.text_input("Enter Twitter Handle (e.g. elonmusk)")
    
    if st.button("Analyze Tweets"):
        if not tw_username:
            st.warning("Please enter a handle.")
        elif not tokenizer or not text_model:
            st.error("Model not loaded.")
        else:
            with st.spinner(f"Fetching tweets for @{tw_username}..."):
                tweets = get_twitter_feed(tw_username)
            
            if not tweets:
                st.info("No tweets found.")
            else:
                if tweets[0].get('is_mock'):
                    st.warning("⚠️ Live Twitter extraction failed (Nitter instances down). Showing **Mock Data** for demonstration.")
                else:
                    st.success("✅ Fetched real tweets from Nitter.")
                    
                st.write(f"Analyzed {len(tweets)} recent tweets.")
                
                toxic_count = 0
                processed_tweets = []
                for tweet in tweets:
                    if isinstance(tweet, dict):
                        text = tweet.get('text', '')
                        image_url = tweet.get('image')
                    else:
                        text = str(tweet)
                        image_url = None

                    cleaned = clean_text(text)
                    pred, _ = predict_toxicity(cleaned)
                    
                    # Image Analysis
                    img_pred = 0
                    if image_url and img_processor and img_model:
                        try:
                            if image_url.startswith('http'):
                                response = requests.get(image_url, timeout=5)
                                if response.status_code == 200:
                                    img = Image.open(BytesIO(response.content)).convert("RGB")
                                    img_pred, _ = predict_image(img)
                        except:
                            pass

                    is_toxic = (pred == 1 or img_pred == 1)
                    if is_toxic:
                        toxic_count += 1
                    
                    processed_tweets.append({'text': text, 'is_toxic': is_toxic, 'image': image_url})

                toxicity_score = (toxic_count / len(tweets)) * 100
                
                st.metric("Toxicity Score", f"{toxicity_score:.1f}%")
                
                if toxicity_score > 50:
                    st.error(f"⚠️ High Risk: High indications of toxic behavior ({toxic_count}/{len(tweets)} toxic tweets).")
                elif toxicity_score > 0:
                    st.warning(f"⚠️ Moderate Risk: Some toxic content detected ({toxic_count}/{len(tweets)} tweets).")
                else:
                    st.success("✅ Safe Profile: No toxic content detected.")
                    
                with st.expander("View Analyzed Tweets"):
                    for item in processed_tweets:
                        status = "🔴" if item['is_toxic'] else "🟢"
                        st.write(f"{status} {item['text']}")
                        if item['image']:
                            st.image(item['image'], width=200)

with tab4:
    st.header("Telegram Channel Analyzer")
    st.markdown("""
    **Note:** This feature analyzes recent messages of a Telegram channel using Telethon modules from `scrapping.py`.
    """)
    tg_input = st.text_input("Channel Username (e.g., news_channel)")
    message_depth = st.number_input("Number of messages to analyze", min_value=1, max_value=100, value=30)
    
    if st.button("Analyze Telegram") and tg_input:
        API_ID = "36191698"
        API_HASH = "fe0b2dbe66ea20681ce48b3b1ba4d95b"
        
        # Async runner for telethon inside streamlit
        async def run_telethon_scrape(channel_link, depth):
            session_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'session_name')
            client = TelegramClient(session_path, API_ID, API_HASH)
            await client.start()
            
            channel_manager = ChannelManager()
            channel_manager.add_channel(channel_link)
            
            messages = []
            try:
                success = await join_channel(client, channel_manager, channel_link)
                if success:
                    entity = await client.get_entity(channel_link)
                    entity_messages, channel_name = await scrape_messages(
                        client, entity, depth, [], channel_manager
                    )
                    messages = entity_messages
            finally:
                await client.disconnect()
            return messages

        with st.spinner(f"Scraping `{tg_input}` using telethon modules..."):
            try:
                # Need to create a new event loop for asyncio.run in Streamlit sometimes, 
                # but asyncio.run is often sufficient if no other loop is running.
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                messages = loop.run_until_complete(run_telethon_scrape(tg_input, message_depth))
                loop.close()
            except Exception as e:
                st.error(f"Error scraping telegram: {e}")
                messages = []
        
        if messages:
            st.success(f"✅ Fetched {len(messages)} messages.")
            toxic_count = 0
            processed_msgs = []
            
            for msg in messages:
                # msg = [sender_id, date, text_content, image_bytes]
                text_content = msg[2]
                image_bytes = msg[3]
                
                # Check text toxicity
                t_pred = 0
                if text_content:
                    cleaned = clean_text(text_content)
                    if cleaned:
                        t_pred, _ = predict_toxicity(cleaned)
                
                # Check image toxicity
                i_pred = 0
                if image_bytes and img_processor and img_model:
                    try:
                        img = Image.open(BytesIO(image_bytes)).convert("RGB")
                        i_pred, _ = predict_image(img)
                    except:
                        pass
                
                is_toxic = (t_pred == 1 or i_pred == 1)
                if is_toxic:
                    toxic_count += 1
                
                processed_msgs.append({'text': text_content, 'is_toxic': is_toxic, 'has_image': bool(image_bytes)})
            
            toxicity_score = (toxic_count / len(messages)) * 100
            st.metric("Toxicity Score", f"{toxicity_score:.1f}%")
            
            if toxicity_score > 50:
                st.error(f"⚠️ High Risk: High indications of toxic behavior ({toxic_count}/{len(messages)} toxic messages).")
            elif toxicity_score > 0:
                st.warning(f"⚠️ Moderate Risk: Some toxic content detected ({toxic_count}/{len(messages)} messages).")
            else:
                st.success("✅ Safe Channel: No toxic content detected in these messages.")
                
            with st.expander("View Analyzed Messages"):
                for item in processed_msgs:
                    status = "🔴 Toxic" if item['is_toxic'] else "🟢 Safe"
                    img_status = "🖼️ (Image)" if item['has_image'] else ""
                    st.write(f"{status} {img_status}: {item['text'][:150]}...")
        else:
            st.warning("No messages fetched or failed to join the channel.")

with tab5:
    st.header("Image Analyzer")
    st.markdown("""
    **How Scoring Works:**
    This tool analyzes both text and images. 
    - **Text Analysis**: Uses a fine-tuned DistilBERT model.
    - **Image Analysis**: Uses a Vision Transformer (ViT) model fine-tuned on hateful memes.
    - **Final Score**: A post is considered **Toxic** if *either* the text or the image is classified as toxic/hateful. The overall "Toxicity Score" is the percentage of analyzed posts that are flagged as toxic.
    """)
    st.write("Upload an image to check for cyberbullying content (e.g., hate symbols, gore, or NSFW).")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        if st.button("Analyze Image"):
            with st.spinner('Analyzing...'):
                if not img_processor or not img_model:
                    st.error("Image model not found or not loaded correctly.")
                else:
                    try:
                        prediction, probabilities = predict_image(image)
                        
                        # Get label from config, fallback to manual mapping if just "0"/"1"
                        id2label = img_model.config.id2label
                        raw_label = id2label[prediction]
                        
                        # Force mapping for Hateful Memes dataset (0=Safe, 1=Toxic)
                        # The dataset usually labels 0 as "not_hateful" and 1 as "hateful"
                        if str(raw_label) == "1" or str(raw_label).lower() == "hateful":
                            display_label = "Toxic / Hateful Content"
                            is_toxic = True
                        else:
                            display_label = "Safe / Non-Toxic"
                            is_toxic = False
                        
                        confidence = probabilities.max().item()
                        
                        if is_toxic:
                             st.error(f"⚠️ Application Detected: **{display_label}**")
                        else:
                             st.success(f"✅ Application Detected: **{display_label}**")
                             
                        st.write(f"Confidence: {confidence*100:.2f}%")
                        
                        with st.expander("See predictions"):
                             # visualizes top N (min of 3 or num_classes)
                             num_classes = len(probabilities)
                             top_k = min(3, num_classes)
                             top_prob, top_indices = torch.topk(probabilities, top_k)
                             
                             for prob, idx in zip(top_prob, top_indices):
                                 lbl = id2label[idx.item()]
                                 # Map for display
                                 if str(lbl) == "1": lbl = "Toxic"
                                 if str(lbl) == "0": lbl = "Safe"
                                 st.write(f"{lbl}: {prob.item()*100:.2f}%")

                    except Exception as e:
                        st.error(f"Error analyzing image: {e}")

st.markdown("---")

with tab6:
    st.header("Audio Analyzer (Toxic Speech Detection)")
    st.write("Upload an audio file (mp3, wav, m4a) to detect toxic speech.")
    
    uploaded_audio = st.file_uploader("Choose an audio file...", type=["mp3", "wav", "m4a", "ogg"])
    
    if uploaded_audio is not None:
        st.audio(uploaded_audio, format='audio/mp3')
        
        if st.button("Analyze Audio"):
            if not audio_model:
                st.error("Audio model (Whisper) not loaded. Please install requirements.")
                st.warning("Make sure ffmpeg is installed on your system (`brew install ffmpeg` on Mac).")
            elif not tokenizer or not text_model:
                 st.error("Text model not loaded.")
            else:
                with st.spinner('Transcribing Audio (this may take a moment)...'):
                    try:
                        # Save to temp file because Whisper needs a path
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_audio.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_audio.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Transcribe
                        result = audio_model.transcribe(tmp_path)
                        transcribed_text = result["text"]
                        
                        # Cleanup temp file
                        os.unlink(tmp_path)
                        
                        st.success("Transcription Complete!")
                        st.text_area("Transcribed Text:", value=transcribed_text, height=100)
                        
                        # Analyze Text
                        if transcribed_text.strip():
                             pred, probs = predict_toxicity(clean_text(transcribed_text))
                             
                             confidence = probs[pred].item()
                             
                             if pred == 1:
                                 st.error(f"⚠️ Application Detected: **Toxic Speech** (Confidence: {confidence*100:.2f}%)")
                             else:
                                 st.success(f"✅ Application Detected: **Safe Speech** (Confidence: {confidence*100:.2f}%)")
                        else:
                             st.warning("Audio was silent or could not be transcribed.")
                             
                    except Exception as e:
                        st.error(f"Error processing audio: {e}")
                        if "ffmpeg" in str(e).lower():
                             st.error("Wait! It looks like **ffmpeg** is missing. You need to install it manually.")
                             st.code("brew install ffmpeg", language="bash")
