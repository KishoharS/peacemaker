from flask import Flask, request, jsonify
import asyncio
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
import tempfile
import open_clip
import instaloader
 
try:
    import whisper
except ImportError:
    whisper = None
 
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type"]}})
 
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_MODEL_PATH = os.getenv("PEACEMAKER_TEXT_MODEL_PATH", os.path.join(BASE_DIR, "models", "bert_model"))
VIT_MODEL_PATH  = os.getenv("PEACEMAKER_VIT_MODEL_PATH",  os.path.join(BASE_DIR, "models", "vit_model"))
 
_SKIP_MODEL_LOAD = os.getenv("PEACEMAKER_SKIP_MODEL_LOAD", "0").strip().lower() in {"1", "true", "yes", "y"}
 
bert_tokenizer = None
bert_model     = None
vit_processor  = None
vit_model      = None
clip_model     = None
clip_preprocess= None
clip_tokenizer = None
audio_model    = None
 
# ── Instaloader session (created once, reused) ──────────────────────────────
_insta_loader = None
 
def _get_insta_loader():
    """
    Returns an authenticated Instaloader instance.
 
    Priority order:
    1. Already initialised in memory — return immediately.
    2. Saved Instaloader session file on disk — load it.
    3. Import session cookies directly from Safari — save to disk for reuse.
 
    No password login is attempted. Instagram blocks programmatic logins
    for new accounts. Importing the Safari session is reliable and safe.
    """
    global _insta_loader
    if _insta_loader is not None:
        return _insta_loader
 
    ig_user = os.getenv("IG_USERNAME", "").strip()
    if not ig_user:
        raise RuntimeError("INSTAGRAM_USERNAME is missing in .env")
 
    L = instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        save_metadata=False,
        quiet=True,
    )
 
    session_dir  = os.path.join(BASE_DIR, "sessions")
    session_file = os.path.join(session_dir, ig_user)
    os.makedirs(session_dir, exist_ok=True)
 
    # Session is created once by running: python import_safari_session.py
    if os.path.exists(session_file):
        try:
            L.load_session_from_file(ig_user, session_file)
            _insta_loader = L
            print(f"Instagram: loaded saved session for @{ig_user}")
            return L
        except Exception as e:
            raise RuntimeError(
                f"Instagram session expired: {e}. Run: python import_safari_session.py"
            )
 
    raise RuntimeError(
        f"No Instagram session found. Run: python import_safari_session.py"
    )
 
 
def _load_models():
    global bert_tokenizer, bert_model
    global vit_processor, vit_model
    global clip_model, clip_preprocess, clip_tokenizer
    global audio_model
 
    if _SKIP_MODEL_LOAD:
        print("PEACEMAKER_SKIP_MODEL_LOAD is set — skipping model init.")
        return
 
    print("Loading BERT model...")
    try:
        bert_tokenizer = DistilBertTokenizer.from_pretrained(TEXT_MODEL_PATH)
        bert_model     = DistilBertForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
        bert_model.eval()
    except Exception as e:
        print(f"Failed to load BERT: {e}")
 
    print("Loading CLIP model...")
    try:
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        clip_model.eval()
    except Exception as e:
        print(f"Failed to load CLIP: {e}")
 
    print("Loading Whisper (Audio) model...")
    try:
        if whisper:
            audio_model = whisper.load_model("base")
            print("Whisper loaded successfully.")
        else:
            print("Whisper library not installed — audio tab will be unavailable.")
    except Exception as e:
        print(f"Failed to load Whisper: {e}")
 
_load_models()
 
 
CLIP_LABELS = [
    "a safe normal image with no threats",
    "a weapon used to threaten someone",
    "violent threatening image with text overlay",
    "hate speech or harassment content",
    "nudity or sexual content",
]
 
 
# ── Helper: run BERT on a piece of text ────────────────────────────────────
def _bert_predict(text):
    """Returns (is_cyberbullying: bool, confidence: float 0-100)"""
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    probs          = torch.softmax(outputs.logits, dim=-1)
    predicted_class= torch.argmax(probs, dim=-1).item()
    confidence     = probs[0][1].item() * 100
    return bool(predicted_class), round(confidence, 2)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TEXT TAB
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    try:
        if bert_tokenizer is None or bert_model is None:
            return jsonify({"error": "Text model not loaded"}), 503
 
        data = request.json
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'Text is required'}), 400
 
        is_cyberbullying, confidence = _bert_predict(text)
        threat_level = 'high' if confidence > 80 else 'medium' if confidence > 50 else 'low'
 
        return jsonify({
            'is_cyberbullying': is_cyberbullying,
            'confidence':       confidence,
            'threat_level':     threat_level,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
 
# ══════════════════════════════════════════════════════════════════════════════
# IMAGE TAB
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    try:
        if clip_model is None or clip_preprocess is None or clip_tokenizer is None:
            return jsonify({"error": "CLIP model not loaded"}), 503
 
        data       = request.json
        image_data = data.get('image', '')
        if not image_data:
            return jsonify({'error': 'Image data is required'}), 400
 
        image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        image       = Image.open(BytesIO(image_bytes)).convert('RGB')
 
        clip_input  = clip_preprocess(image).unsqueeze(0)
        text_tokens = clip_tokenizer(CLIP_LABELS)
 
        with torch.no_grad():
            image_features = clip_model.encode_image(clip_input)
            text_features  = clip_model.encode_text(text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features  /= text_features.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
 
        harmful_score = max(probs[0][1:]).item() * 100
        is_harmful    = harmful_score > 40
        scores        = {label: round(probs[0][i].item() * 100, 1) for i, label in enumerate(CLIP_LABELS)}
        threat_level  = 'high' if harmful_score > 80 else 'medium' if harmful_score > 50 else 'low'
 
        return jsonify({
            'is_harmful':    is_harmful,
            'confidence':    round(harmful_score, 2),
            'threat_level':  threat_level,
            'label_scores':  scores,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
 
# ══════════════════════════════════════════════════════════════════════════════
# AUDIO TAB
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/analyze-audio', methods=['POST'])
def analyze_audio():
    tmp_path = None
    try:
        if audio_model is None:
            return jsonify({"error": "Whisper model not loaded. Try restarting the server."}), 503
        if bert_tokenizer is None or bert_model is None:
            return jsonify({"error": "Text model not loaded"}), 503
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
 
        audio_file = request.files['audio']
        ext        = audio_file.filename.rsplit('.', 1)[-1].lower() if '.' in audio_file.filename else 'mp3'
 
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name
 
        result       = audio_model.transcribe(tmp_path)
        transcription= result["text"].strip()
 
        if not transcription:
            return jsonify({'error': 'Could not extract speech from audio. Make sure the file has clear speech.'}), 400
 
        is_cyberbullying, confidence = _bert_predict(transcription)
        threat_level = 'high' if confidence > 80 else 'medium' if confidence > 50 else 'low'
 
        return jsonify({
            'is_cyberbullying': is_cyberbullying,
            'confidence':       confidence,
            'threat_level':     threat_level,
            'transcription':    transcription,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# INSTAGRAM TAB  (Instaloader — no RapidAPI key needed)
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/analyze-instagram', methods=['POST'])
def analyze_instagram():
    try:
        if bert_tokenizer is None or bert_model is None:
            return jsonify({"error": "Text model not loaded"}), 503
 
        data       = request.json
        username   = data.get('username', '').strip().lstrip('@')
        post_count = int(data.get('count', 10))
 
        if not username:
            return jsonify({'error': 'Username is required'}), 400
 
        try:
            L = _get_insta_loader()
        except RuntimeError as e:
            return jsonify({'error': str(e)}), 500
 
        try:
            profile = instaloader.Profile.from_username(L.context, username)
        except instaloader.exceptions.ProfileNotExistsException:
            return jsonify({'error': f'Instagram profile @{username} does not exist.'}), 404
        except instaloader.exceptions.LoginRequiredException:
            return jsonify({'error': 'Login required to view this profile. Check your Instagram credentials in .env.'}), 403
        except instaloader.exceptions.ConnectionException as e:
            return jsonify({'error': f'Instagram connection error: {str(e)}'}), 502
 
        if profile.is_private:
            return jsonify({'error': f'@{username} is a private account. Only public accounts can be scanned.'}), 403
 
        posts = []
        for post in profile.get_posts():
            if len(posts) >= post_count:
                break
            caption = post.caption or ""
            if caption.strip():
                posts.append(caption)
 
        if not posts:
            return jsonify({'error': 'No captions found on this profile (posts may have no text).'}), 404
 
        bullying_count    = 0
        all_posts_details = []
 
        for caption in posts:
            is_toxic, confidence = _bert_predict(caption)
            if is_toxic:
                bullying_count += 1
            all_posts_details.append({
                'content':     caption,
                'is_bullying': is_toxic,
                'confidence':  confidence,
            })
 
        percentage = (bullying_count / len(posts)) * 100
 
        return jsonify({
            'posts_analyzed':   len(posts),
            'bullying_detected':bullying_count,
            'percentage':       round(percentage, 1),
            'all_posts':        all_posts_details,
            'recommendation':   "High risk detected." if percentage > 30 else "Account appears generally safe.",
        })
 
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
 
# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM TAB
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/analyze-social', methods=['POST'])
def analyze_social():
    """
    Kept the original route path (/api/analyze-social) so the frontend
    doesn't need any changes for the Telegram tab.
    """
    try:
        if bert_tokenizer is None or bert_model is None:
            return jsonify({"error": "Text model not loaded"}), 503
 
        data       = request.json
        platform   = data.get('platform', '').lower()
        username   = data.get('username', '').strip()
        post_count = int(data.get('count', 30))
 
        if not username:
            return jsonify({'error': 'Username is required'}), 400
 
        # ── Instagram ─────────────────────────────────────────────
        if platform == 'instagram':
            # Forward to the same logic as /api/analyze-instagram
            # (frontend calls /api/analyze-social with platform=instagram)
            try:
                L = _get_insta_loader()
            except RuntimeError as e:
                return jsonify({'error': str(e)}), 500
 
            clean_username = username.lstrip('@')
            try:
                profile = instaloader.Profile.from_username(L.context, clean_username)
            except instaloader.exceptions.ProfileNotExistsException:
                return jsonify({'error': f'Instagram profile @{clean_username} does not exist.'}), 404
            except instaloader.exceptions.LoginRequiredException:
                return jsonify({'error': 'Login required. Check INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD in .env.'}), 403
            except instaloader.exceptions.ConnectionException as e:
                return jsonify({'error': f'Instagram connection error: {str(e)}'}), 502
 
            if profile.is_private:
                return jsonify({'error': f'@{clean_username} is a private account. Only public accounts can be scanned.'}), 403
 
            ig_posts = []
            for post in profile.get_posts():
                if len(ig_posts) >= post_count:
                    break
                caption = post.caption or ""
                if caption.strip():
                    ig_posts.append(caption)
 
            if not ig_posts:
                return jsonify({'error': 'No captions found on this profile.'}), 404
 
            bullying_count    = 0
            all_posts_details = []
            for caption in ig_posts:
                is_toxic, confidence = _bert_predict(caption)
                if is_toxic:
                    bullying_count += 1
                all_posts_details.append({'content': caption, 'is_bullying': is_toxic, 'confidence': confidence})
 
            percentage = (bullying_count / len(ig_posts)) * 100
            return jsonify({
                'posts_analyzed':    len(ig_posts),
                'bullying_detected': bullying_count,
                'percentage':        round(percentage, 1),
                'all_posts':         all_posts_details,
                'recommendation':    "High risk detected." if percentage > 30 else "Account appears generally safe.",
            })
 
        # ── Telegram ─────────────────────────────────────────────
        if platform == 'telegram':
            api_id   = os.getenv("TELEGRAM_API_ID", "").strip()
            api_hash = os.getenv("TELEGRAM_API_HASH", "").strip()
            session  = os.getenv("TELEGRAM_SESSION", "peacemaker_session").strip()
 
            if not api_id or not api_hash:
                return jsonify({"error": "Telegram credentials missing. Add TELEGRAM_API_ID and TELEGRAM_API_HASH to your .env file."}), 400
 
            from telethon.sync import TelegramClient
 
            posts = []
            loop  = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
 
            client = TelegramClient(session, int(api_id), api_hash, loop=loop)
 
            try:
                client.connect()
 
                if not client.is_user_authorized():
                    phone = os.getenv("TELEGRAM_PHONE_NUMBER", "").strip()
                    if not phone:
                        client.disconnect()
                        loop.close()
                        return jsonify({
                            "error": "Telegram session not authorised. Add TELEGRAM_PHONE_NUMBER to your .env and run a one-time auth script."
                        }), 401
 
                    client.send_code_request(phone)
                    # First-run authorisation is a terminal flow — can't be done via HTTP.
                    client.disconnect()
                    loop.close()
                    return jsonify({
                        "error": "Telegram first-time auth required. Run: python auth_telegram.py from your project folder."
                    }), 401
 
                entity = client.get_entity(username)
                for message in client.iter_messages(entity, limit=post_count):
                    if message.text and message.text.strip():
                        posts.append(message.text.strip())
 
            except Exception as e:
                return jsonify({'error': f'Telegram error: {str(e)}'}), 400
            finally:
                try:
                    client.disconnect()
                except Exception:
                    pass
                try:
                    loop.close()
                except Exception:
                    pass
 
            if not posts:
                return jsonify({'error': 'No messages found in this channel.'}), 404
 
            bullying_count    = 0
            all_posts_details = []
 
            for text in posts:
                is_toxic, confidence = _bert_predict(text)
                if is_toxic:
                    bullying_count += 1
                all_posts_details.append({
                    'content':     text,
                    'is_bullying': is_toxic,
                    'confidence':  confidence,
                })
 
            percentage = (bullying_count / len(posts)) * 100
 
            return jsonify({
                'posts_analyzed':    len(posts),
                'bullying_detected': bullying_count,
                'percentage':        round(percentage, 1),
                'all_posts':         all_posts_details,
                'recommendation':    "High risk detected." if percentage > 30 else "Channel appears generally safe.",
            })
 
        return jsonify({'error': f'Platform "{platform}" is not supported.'}), 400
 
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
 
# ══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status':  'ok',
        'bert':    bert_model    is not None,
        'clip':    clip_model    is not None,
        'whisper': audio_model   is not None,
    })
 
 
if __name__ == '__main__':
    debug = os.getenv("FLASK_DEBUG", "0").strip().lower() in {"1", "true", "yes", "y"}
    port  = int(os.getenv("PORT", "8001"))
    app.run(debug=debug, port=port)