import re
import string
import torch
import pytesseract
from PIL import Image
from transformers import AutoTokenizer, AutoModel

# ─────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────

def clean_text(text):
    """
    Cleans text by removing special characters, punctuation, and converting to lowercase.
    """
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet text
    text = re.sub(r'\@\w+|\#', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra spaces
    text = " ".join(text.split())
    return text


# ─────────────────────────────────────────
# Image Processing with OCR + ViT + Fusion
# ─────────────────────────────────────────

def process_image_input(image_path, vit_model, distilbert_model, tokenizer, fusion_model):
    """
    Takes an image, extracts visual features via ViT,
    extracts any overlaid text via OCR → DistilBERT,
    then fuses both for final cyberbullying prediction.
    """
    # --- Visual branch ---
    image = Image.open(image_path).convert("RGB")
    visual_features = vit_model(image)  # your existing ViT forward pass

    # --- OCR branch ---
    raw_text = pytesseract.image_to_string(image).strip()
    extracted_text = clean_text(raw_text)  # clean before passing to DistilBERT

    if extracted_text:
        inputs = tokenizer(
            extracted_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            text_features = distilbert_model(**inputs).last_hidden_state[:, 0, :]
    else:
        # no text found in image — use zero vector matching DistilBERT hidden size
        text_features = torch.zeros(1, 768)

    # --- Fusion ---
    combined = torch.cat([visual_features, text_features], dim=-1)
    prediction = fusion_model(combined)

    return prediction


# ─────────────────────────────────────────
# Initialisation (call this once at startup)
# ─────────────────────────────────────────

def load_models():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    distilbert_model = AutoModel.from_pretrained("distilbert-base-uncased")
    distilbert_model.eval()
    return tokenizer, distilbert_model


# ─────────────────────────────────────────
# Example Usage
# ─────────────────────────────────────────

if __name__ == "__main__":
    tokenizer, distilbert_model = load_models()

    # vit_model and fusion_model are your existing trained models
    prediction = process_image_input(
        image_path="test_image.jpg",
        vit_model=vit_model,
        distilbert_model=distilbert_model,
        tokenizer=tokenizer,
        fusion_model=fusion_model
    )
    print("Prediction:", prediction)