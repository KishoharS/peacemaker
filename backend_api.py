from flask import Flask, request, jsonify
import torch
from transformers import (
    DistilBertForSequenceClassification, DistilBertTokenizer,
    ViTForImageClassification, ViTImageProcessor
)
import os
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64
import json

import re
import string
from transformers import AutoModel, AutoTokenizer

import open_clip

def clean_ocr_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join(text.split())
    return text

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type"]}})

# Model paths
BERT_MODEL_PATH = '/Users/kishohars/Projects/cyberbullying_detection_project/models/bert_model'
VIT_MODEL_PATH = '/Users/kishohars/Projects/cyberbullying_detection_project/models/vit_model'

# Load models on startup
print("Loading BERT model...")
bert_tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_PATH)
bert_model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)

print("Loading Vision Transformer model...")
try:
    vit_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_PATH)
    vit_model = ViTForImageClassification.from_pretrained(VIT_MODEL_PATH)
    vit_available = True
except:
    print("Vision Transformer model not found")
    vit_available = False

print("Loading CLIP model...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip_model.eval()

CLIP_LABELS = [
    "a safe normal image with no threats",
    "a weapon used to threaten someone",
    "violent threatening image with text overlay",
    "hate speech or harassment content",
    "nudity or sexual content",
]

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # Tokenize and predict
        inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = bert_model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        
        bullying_confidence = probabilities[0][1].item() * 100
        
        # Determine threat level
        if bullying_confidence > 80:
            threat_level = 'high'
        elif bullying_confidence > 50:
            threat_level = 'medium'
        else:
            threat_level = 'low'
        
        return jsonify({
            'is_cyberbullying': bool(predicted_class),
            'confidence': round(bullying_confidence, 2),
            'threat_level': threat_level,
            'analysis': f"Text {'contains' if predicted_class else 'does not contain'} cyberbullying patterns with {bullying_confidence:.1f}% confidence."
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    try:
        data = request.json
        image_data = data.get('image', '')

        if not image_data:
            return jsonify({'error': 'Image data is required'}), 400

        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')

        # --- CLIP classification ---
        clip_input = clip_preprocess(image).unsqueeze(0)
        text_tokens = clip_tokenizer(CLIP_LABELS)

        with torch.no_grad():
            image_features = clip_model.encode_image(clip_input)
            text_features = clip_model.encode_text(text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # First label is "safe", rest are harmful
        safe_score = probs[0][0].item() * 100
        harmful_score = max(probs[0][1:]).item() * 100
        is_harmful = harmful_score > 40

        # Find which harmful label triggered
        scores = {label: round(probs[0][i].item() * 100, 1) 
                  for i, label in enumerate(CLIP_LABELS)}

        if harmful_score > 80:
            threat_level = 'high'
        elif harmful_score > 50:
            threat_level = 'medium'
        else:
            threat_level = 'low'

        return jsonify({
            'is_harmful': is_harmful,
            'confidence': round(harmful_score, 2),
            'threat_level': threat_level,
            'label_scores': scores,
            'analysis': f"Image {'contains harmful content' if is_harmful else 'appears safe'} with {harmful_score:.1f}% confidence."
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/analyze-social', methods=['POST'])
def analyze_social():
    try:
        data = request.json
        platform = data.get('platform', '').lower()
        username = data.get('username', '').strip()
        post_count = data.get('count', 10)
        
        if not platform or not username:
            return jsonify({'error': 'Platform and username are required'}), 400
        
        posts = []
        
        if platform == 'instagram':
            try:
                import instaloader
                L = instaloader.Instaloader()
                profile = instaloader.Profile.from_username(L.context, username)
                
                post_idx = 0
                for post in profile.get_posts():
                    if post_idx >= post_count:
                        break
                    
                    caption = post.caption or ""
                    image_url = post.url if hasattr(post, 'url') else None
                    
                    posts.append({
                        'caption': caption,
                        'image': image_url,
                        'likes': post.likes,
                        'date': str(post.date)
                    })
                    post_idx += 1
            except Exception as e:
                return jsonify({'error': f'Instagram Error: {str(e)}. Account may be private or not found.'}), 400
        
        elif platform == 'telegram':
            # Telegram requires API credentials setup at my.telegram.org
            return jsonify({
                'error': 'Telegram requires API setup. Follow setup guide.',
                'setup_guide': {
                    'step1': 'Visit https://my.telegram.org/apps',
                    'step2': 'Login with your Telegram phone number',
                    'step3': 'Create a new application to get API_ID and API_HASH',
                    'step4': 'Update backend_api.py with your credentials',
                    'step5': 'Then test with public channels: @telegram, @durov, @daily, @cybersecurity'
                },
                'popular_channels': [
                    '@telegram (Official Telegram)',
                    '@durov (Pavel Durov)',
                    '@daily (Daily news)',
                    '@nasa_official (NASA)',
                    '@cybersecurity (Cybersecurity news)',
                    '@cnbc (CNBC)',
                    '@bbc_news (BBC News)',
                    '@blockchain (Blockchain news)'
                ],
                'note': 'For now, Instagram works without setup. Use Instagram accounts for testing.'
            }), 400
        else:
            return jsonify({'error': 'Platform not supported. Use: instagram, telegram'}), 400
        
        if not posts:
            return jsonify({'error': 'No posts found or account is private'}), 404
        
        # Analyze each post/caption
        bullying_count = 0
        total_analyzed = 0
        threat_details = []
        all_posts_details = []
        
        for post in posts:
            text = post.get('caption', '')
            
            if not text or len(text.strip()) == 0:
                continue
            
            # Analyze text
            inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = bert_model(**inputs)
            
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][1].item() * 100
            
            total_analyzed += 1
            
            post_detail = {
                'content': text[:150] + '...' if len(text) > 150 else text,
                'full_content': text,
                'confidence': round(confidence, 1),
                'severity': 'high' if confidence > 80 else 'medium' if confidence > 50 else 'low',
                'is_bullying': bool(predicted_class),
                'image': post.get('image'),
                'likes': post.get('likes', 0),
                'date': post.get('date', '')
            }
            
            # Add to all posts for display
            all_posts_details.append(post_detail)
            
            if predicted_class == 1:
                bullying_count += 1
                threat_details.append(post_detail)
        
        if total_analyzed == 0:
            return jsonify({'error': 'No valid captions found to analyze'}), 400
        
        bullying_percentage = (bullying_count / total_analyzed * 100) if total_analyzed > 0 else 0
        
        return jsonify({
            'username': username,
            'platform': platform,
            'posts_analyzed': total_analyzed,
            'bullying_detected': bullying_count,
            'percentage': round(bullying_percentage, 1),
            'threat_level': 'high' if bullying_percentage > 30 else 'medium' if bullying_percentage > 10 else 'low',
            'threats': threat_details[:10],  # Show flagged threats
            'all_posts': all_posts_details,  # Show all analyzed posts for visibility
            'recommendation': f"{'High risk account - immediate review recommended' if bullying_percentage > 30 else 'Moderate concern - continued monitoring suggested' if bullying_percentage > 10 else 'Account appears safe'}"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=8001)
