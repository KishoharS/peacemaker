from dotenv import load_dotenv
load_dotenv()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import importlib
import os
import random
import re
import sys
import tempfile
from io import BytesIO
import base64

import pandas as pd
import requests
import streamlit as st
import torch
import whisper
from PIL import Image
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    ViTForImageClassification,
    ViTImageProcessor,
)

from src.instagram_utils import get_loader, get_instagram_posts
from backend_api import analyze_image

@st.cache_resource
def load_session():
    return get_loader(
        username=os.getenv("IG_USERNAME"),
        password=os.getenv("IG_PASSWORD")
    )


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from src.instagram_utils import get_loader, get_instagram_posts
from src.preprocessing_text import clean_text
from scrapping import ChannelManager, TelegramClient, join_channel, scrape_messages


# Load all models
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "bert_model")
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        return None, None


@st.cache_resource
def load_image_model():
    # Try custom trained model first
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "vit_model")
    if not os.path.exists(model_path):
        return None, None

    try:
        processor = ViTImageProcessor.from_pretrained(model_path)
        model = ViTForImageClassification.from_pretrained(model_path)
        return processor, model
    except Exception as e:
        st.error(f"Error loading image model: {e}")
        return None, None


# this model causing error!
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
# Audio model load omitted or moved to where it is used to save memory if needed
audio_model = load_audio_model()


# helper function for prediction
def predict_toxicity(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
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

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Telegram", "Instagram", "Text Analyzer", "Image Analyzer", "Audio Analyzer"]
)

with tab1:
    st.header("Telegram Channel Analyzer")
    st.markdown("""
    **Note:** This feature analyzes recent messages of a Telegram channel using Telethon modules from `scrapping.py`.
    """)
    tg_input = st.text_input("Channel Username (e.g., news_channel)")
    message_depth = st.number_input(
        "Number of messages to analyze", min_value=1, max_value=100, value=30
    )

    if st.button("Analyze Telegram") and tg_input:
        API_ID = "36191698"
        API_HASH = "fe0b2dbe66ea20681ce48b3b1ba4d95b"

        # Async runner for telethon inside streamlit
        async def run_telethon_scrape(channel_link, depth):
            session_path = os.path.join(
                os.path.dirname(__file__), "..", "src", "session_name"
            )
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
                messages = loop.run_until_complete(
                    run_telethon_scrape(tg_input, message_depth)
                )
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

                is_toxic = t_pred == 1 or i_pred == 1
                if is_toxic:
                    toxic_count += 1

                processed_msgs.append(
                    {
                        "text": text_content,
                        "is_toxic": is_toxic,
                        "has_image": bool(image_bytes),
                    }
                )

            toxicity_score = (toxic_count / len(messages)) * 100
            st.metric("Toxicity Score", f"{toxicity_score:.1f}%")

            if toxicity_score > 50:
                st.error(
                    f"⚠️ High Risk: High indications of toxic behavior ({toxic_count}/{len(messages)} toxic messages)."
                )
            elif toxicity_score > 0:
                st.warning(
                    f"⚠️ Moderate Risk: Some toxic content detected ({toxic_count}/{len(messages)} messages)."
                )
            else:
                st.success(
                    "✅ Safe Channel: No toxic content detected in these messages."
                )

            with st.expander("View Analyzed Messages"):
                for item in processed_msgs:
                    status = "🔴 Toxic" if item["is_toxic"] else "🟢 Safe"
                    img_status = "🖼️ (Image)" if item["has_image"] else ""
                    st.write(f"{status} {img_status}: {item['text'][:150]}...")
        else:
            st.warning("No messages fetched or failed to join the channel.")


with tab2:
    st.header("Instagram Profile Analyzer")
    st.markdown("""
    **Note:** This feature analyzes the public posts and captions of a user 
    to determine if their content is toxic.
    """)

    username = st.text_input("Enter Instagram Username (e.g. ig_kishohar)")

    if st.button("Analyze Profile"):
        if not username:
            st.warning("Please enter a username.")
        elif not tokenizer or not text_model:
            st.error("Text model not loaded.")
        else:
            with st.spinner(f"Fetching posts for {username}..."):
                try:
                    loader = load_session()  # cached, no re-login
                    posts = get_instagram_posts(username, loader, max_posts=30)
                except Exception as e:
                    st.error(f"Session error: {e}")
                    posts = []

            if not posts:
                st.info("No posts found or profile is private.")
            else:
                st.write(f"Fetched {len(posts)} recent posts.")

                toxic_count = 0
                results = []

                for post in posts:
                    text_content = post.get("text", "")
                    image = post.get("image")  # already a PIL.Image or None

                    # text analysis
                    cleaned = clean_text(text_content)
                    pred, _ = predict_toxicity(cleaned)

                    # image analysis
                    img_pred = 0
                    if image is not None and img_processor and img_model:
                        try:
                            img_pred, _ = predict_image(image)
                        except Exception as e:
                            print(f"Image analysis failed for {post['shortcode']}: {e}")

                    is_toxic = int(pred == 1 or img_pred == 1)
                    toxic_count += is_toxic

                    results.append({
                        "shortcode": post["shortcode"],
                        "text": text_content,
                        "text_toxic": pred,
                        "image_toxic": img_pred,
                        "toxic": is_toxic
                    })

                # overall verdict
                toxicity_score = (toxic_count / len(results)) * 100
                st.metric("Toxicity Score", f"{toxicity_score:.1f}%")

                if toxicity_score > 50:
                    st.error(
                        f"⚠️ High Risk: {toxic_count}/{len(results)} posts flagged as toxic."
                    )
                elif toxicity_score > 0:
                    st.warning(
                        f"⚠️ Moderate Risk: {toxic_count}/{len(results)} posts flagged as toxic."
                    )
                else:
                    st.success("✅ Safe Profile: No toxic content detected in recent posts.")

                # expanded post view
                with st.expander("View Analyzed Posts"):
                    for r in results:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.text(r["text"][:150] + "..." if len(r["text"]) > 150 else r["text"])
                            st.caption(
                                f"https://www.instagram.com/p/{r['shortcode']}/"
                            )
                        with col2:
                            if r["toxic"]:
                                st.error("Toxic")
                            else:
                                st.success("Safe")
                        st.divider()

    st.markdown("---")


with tab3:
    st.header("Analyze Text/Comments")
    st.write(
        "Paste your comments below to see whether you're being cyberbullied or not!!."
    )

    text_input = st.text_area("Input Text", height=150)

    if st.button("Analyze Text"):
        if not tokenizer or not text_model:
            st.error(
                "Model artifacts not found. Please train the model and download artifacts to models/bert_model."
            )
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
                    st.write(f"Confidence: {probabilities[1] * 100:.2f}%")
                else:
                    st.success("✅ Application Detected: **Non-Cyberbullying**")
                    st.write(f"Confidence: {probabilities[0] * 100:.2f}%")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
    
    st.markdown("---")

with tab4:
    st.header("Image Analyzer")
    st.markdown("""
    **How Scoring Works:**
    This tool analyzes both text and images.
    - **Text Analysis**: Uses OpenAI's CLIP model to extract text.
    - **Image Analysis**: Uses a Vision Transformer (ViT) model fine-tuned on hateful memes.
    - **Final Score**: A post is considered **Toxic** if *either* the text or the image is classified as toxic/hateful. The overall "Toxicity Score" is the percentage of analyzed posts that are flagged as toxic.
    """)
    st.write("Upload an image to check for cyberbullying content (e.g., hate symbols, gore, or NSFW).")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                try:
                    # Convert image to base64 and send to backend
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                    response = requests.post(
                        "http://localhost:8001/api/analyze-image",
                        json={"image": img_base64}
                    )
                    result = response.json()

                    is_harmful = result.get("is_harmful", False)
                    confidence = result.get("confidence", 0)
                    threat_level = result.get("threat_level", "low")
                    analysis = result.get("analysis", "")

                    if is_harmful:
                        st.error(f"⚠️ Application Detected: **Toxic / Harmful Content**")
                    else:
                        st.success(f"✅ Application Detected: **Safe / Non-Toxic**")

                    st.write(f"Confidence: {confidence:.2f}%")
                    st.write(f"Threat Level: {threat_level.upper()}")
                    st.write(analysis)

                    with st.expander("See label scores"):
                        label_scores = result.get("label_scores", {})
                        for label, score in label_scores.items():
                            st.write(f"{label}: {score:.1f}%")

                except Exception as e:
                    st.error(f"Error analyzing image: {e}")

    st.markdown("---")

with tab5:
    st.header("Audio Analyzer (Toxic Speech Detection)")
    st.markdown("""
    **How Scoring Works:**
    This tool transcripts audio to text using OpenAI's Whisper.
    - **Audio Analysis**: Uses OpenAI's Whisper model to extract text.
    - **Image Analysis**: Uses DistilBERT model which is fine-tuned on hateful text.
    - **Final Score**: A audio is considered **Toxic** if the text is classified as toxic/hateful. The overall "Toxicity Score" is the percentage of analyzed audio that are flagged as toxic.
    """)

    uploaded_audio = st.file_uploader(
        "Choose an audio file...", type=["mp3", "wav", "m4a", "ogg"]
    )

    if uploaded_audio is not None:
        st.audio(uploaded_audio, format="audio/mp3")

        if st.button("Analyze Audio"):
            if not audio_model:
                st.error(
                    "Audio model (Whisper) not loaded. Please install requirements."
                )
                st.warning(
                    "Make sure ffmpeg is installed on your system (`brew install ffmpeg` on Mac)."
                )
            elif not tokenizer or not text_model:
                st.error("Text model not loaded.")
            else:
                with st.spinner("Transcribing Audio (this may take a moment)..."):
                    try:
                        # Save to temp file because Whisper needs a path
                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=f".{uploaded_audio.name.split('.')[-1]}",
                        ) as tmp_file:
                            tmp_file.write(uploaded_audio.getvalue())
                            tmp_path = tmp_file.name

                        # Transcribe
                        result = audio_model.transcribe(tmp_path)
                        transcribed_text = result["text"]

                        # Cleanup temp file
                        os.unlink(tmp_path)

                        st.success("Transcription Complete!")
                        st.text_area(
                            "Transcribed Text:", value=transcribed_text, height=100
                        )

                        # Analyze Text
                        if transcribed_text.strip():
                            pred, probs = predict_toxicity(clean_text(transcribed_text))

                            confidence = probs[pred].item()

                            if pred == 1:
                                st.error(
                                    f"⚠️ Application Detected: **Toxic Speech** (Confidence: {confidence * 100:.2f}%)"
                                )
                            else:
                                st.success(
                                    f"✅ Application Detected: **Safe Speech** (Confidence: {confidence * 100:.2f}%)"
                                )
                        else:
                            st.warning("Audio was silent or could not be transcribed.")

                    except Exception as e:
                        st.error(f"Error processing audio: {e}")
                        if "ffmpeg" in str(e).lower():
                            st.error(
                                "Wait! It looks like **ffmpeg** is missing. You need to install it manually."
                            )
                            st.code("brew install ffmpeg", language="bash")
