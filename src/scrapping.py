import asyncio
import re
import pandas as pd
import torch
from transformers import pipeline, ViTImageProcessor, ViTForImageClassification
import io
from PIL import Image
from telethon.sync import TelegramClient
from telethon.tl.functions.channels import JoinChannelRequest
from telethon.errors import FloodWaitError, ChannelPrivateError
from telethon.tl.types import Channel, User, Channel, Chat
import multiprocessing
from functools import partial
import argparse
import json
import random
import signal
import os
from datetime import datetime
from colorama import init, Fore, Back, Style

init(autoreset=True)

PURPLE_BLUE = '\033[38;2;100;100;255m'
LIGHT_PURPLE = '\033[38;2;200;180;255m'
BOLD_WHITE = '\033[1;37m'

def print_info(message):
    print(f"{PURPLE_BLUE}ℹ {BOLD_WHITE}{message}")

def print_success(message):
    print(f"{LIGHT_PURPLE}✔ {BOLD_WHITE}{message}")

def print_warning(message):
    print(f"{Fore.YELLOW}{Style.BRIGHT}⚠ {BOLD_WHITE}{message}")

def print_error(message):
    print(f"{Fore.RED}✘ {message}")

def print_header(message):
    print(f"\n{PURPLE_BLUE}{Style.BRIGHT}{message}")
    print(f"{PURPLE_BLUE}{'-' * len(message)}{Style.RESET_ALL}")

def print_subheader(message):
    print(f"\n{LIGHT_PURPLE}{Style.BRIGHT}{message}")
    print(f"{LIGHT_PURPLE}{'-' * len(message)}{Style.RESET_ALL}")

# Extract Telegram channel links from messages
def extract_channel_links(text):
    if not text or not isinstance(text, str):
        return []
    pattern = r't\.me/(?:joinchat/)?[a-zA-Z0-9_-]+'
    return re.findall(pattern, text)

# Clean and format channel links
def clean_link(link):
    if not link or not isinstance(link, str):
        return None
    
    link = link.split(')')[0].strip()
    
    if re.match(r'^[a-zA-Z0-9_]{5,}$', link):
        return link
    
    match = re.search(r't\.me/(?:joinchat/)?([a-zA-Z0-9_-]+)', link)
    if match:
        username_or_hash = match.group(1)
        
        if 'joinchat' in link:
            return f'https://t.me/joinchat/{username_or_hash}'
        
        return username_or_hash
    
    return None

# Manage discovered channels
class ChannelManager:
    def __init__(self):
        self.discovered_channels = set()
        self.joined_channels = set()
        self.processed_channels = set()
        self.channel_affiliations = {}
        self.initial_channels = set()

    def add_channel(self, link, source_channel=None):
        cleaned_link = clean_link(link)
        if cleaned_link and cleaned_link not in self.joined_channels and cleaned_link not in self.processed_channels:
            self.discovered_channels.add(cleaned_link)
            if source_channel:
                self.channel_affiliations[cleaned_link] = source_channel
            else:
                self.initial_channels.add(cleaned_link)  # Mark as initial channel if no source

    def mark_as_joined(self, link):
        cleaned_link = clean_link(link)
        if cleaned_link:
            self.joined_channels.add(cleaned_link)
            self.discovered_channels.discard(cleaned_link)

    def mark_as_processed(self, link):
        cleaned_link = clean_link(link)
        if cleaned_link:
            self.processed_channels.add(cleaned_link)
            self.discovered_channels.discard(cleaned_link)

    def has_unprocessed_channels(self):
        return len(self.discovered_channels) > 0

    def get_next_channel(self):
        if self.discovered_channels:
            return self.discovered_channels.pop()
        return None

    def get_affiliation(self, link):
        cleaned_link = clean_link(link)
        return self.channel_affiliations.get(cleaned_link, None)

    def display_status(self):
        print_subheader("Channel Status")
        print(f"  Channels waiting to be processed: {len(self.discovered_channels)}")
        print(f"  Channels joined: {len(self.joined_channels)}")
        print(f"  Channels processed: {len(self.processed_channels)}")

# Join channel by url
async def join_channel(client, channel_manager, link, max_retries=3):
    cleaned_link = clean_link(link)
    if not cleaned_link:
        print_warning(f"Invalid link format: {link}")
        return False

    retries = 0
    while retries < max_retries:
        try:
            entity = await client.get_entity(cleaned_link)
            entity_name = await get_entity_name(entity)
            
            if isinstance(entity, (Channel, Chat)):
                if entity.username:
                    await client(JoinChannelRequest(entity))
                else:
                    print_warning(f"Cannot join private channel {entity_name} without an invite link")
                    return False
            elif isinstance(entity, User):
                print_info(f"Entity {entity_name} is a user, no need to join")
            else:
                print_warning(f"Unknown entity type for {entity_name}")
                return False
            
            print_success(f"Successfully processed entity: {entity_name}")
            channel_manager.mark_as_joined(cleaned_link)
            return True

        except FloodWaitError as e:
            wait_time = min(e.seconds, 30)
            print_warning(f"FloodWaitError encountered. Waiting for {wait_time} seconds. (Attempt {retries + 1}/{max_retries})")
            await asyncio.sleep(wait_time)
        except Exception as e:
            print_error(f"Failed to process entity {cleaned_link}: {e}")
        
        retries += 1
        await asyncio.sleep(1)

    print_warning(f"Max retries exceeded. Failed to process entity: {cleaned_link}")
    return False

# Load configuration
def load_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

# Create a default config file, if no config present (providing one anyways for clarity sake)
def create_default_config(config_path):
    default_config = {
        "initial_channel_links": [],
        "message_keywords": [],
        "batch_size": 100
    }
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=4)
    print_success(f"Default config file created at {config_path}")
    print_info("Please edit this file with your channel links and keywords.")
    return default_config

# Using the fine-tuned DistilBERT and ViT models for cyberbullying detection
class CyberbullyingDetector:
    def __init__(self):
        # Determine device appropriately
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = "cuda" if torch.cuda.is_available() else device
        
        # Load BERT
        bert_path = os.path.join(os.path.dirname(__file__), "..", "models", "bert_model")
        if not os.path.exists(bert_path):
            print_warning(f"BERT model not found at {bert_path}. Fallback to neutral score.")
            self.text_pipeline = None
        else:
            print_info(f"Loading BERT sentiment model from {bert_path} on {self.device}...")
            self.text_pipeline = pipeline("text-classification", model=bert_path, tokenizer=bert_path, device=self.device)

        # Load ViT
        vit_path = os.path.join(os.path.dirname(__file__), "..", "models", "vit_model")
        if not os.path.exists(vit_path):
            print_warning(f"ViT model not found at {vit_path}. Image analysis disabled.")
            self.img_processor = None
            self.img_model = None
        else:
            print_info(f"Loading ViT image model from {vit_path} on {self.device}...")
            self.img_processor = ViTImageProcessor.from_pretrained(vit_path)
            self.img_model = ViTForImageClassification.from_pretrained(vit_path).to(self.device)

    def analyze(self, text, image_bytes):
        result = {'text_toxic': False, 'image_toxic': False, 'is_toxic': False, 'text_score': 0.0, 'image_score': 0.0}
        
        # Text Analysis
        if text and isinstance(text, str) and text.strip() != "":
            if self.text_pipeline is not None:
                try:
                    res = self.text_pipeline(text, truncation=True, max_length=128)[0]
                    result['text_score'] = res['score']
                    if res['label'] == 'LABEL_1':
                        result['text_toxic'] = True
                except Exception as e:
                    pass
                    
        # Image Analysis
        if image_bytes and self.img_processor and self.img_model:
            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                inputs = self.img_processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.img_model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=-1).item()
                result['image_score'] = torch.softmax(logits, dim=-1)[0][predicted_class].item()
                
                id2label = self.img_model.config.id2label
                raw_label = str(id2label[predicted_class])
                if raw_label == "1" or raw_label.lower() == "hateful":
                    result['image_toxic'] = True
            except Exception as e:
                pass

        result['is_toxic'] = result['text_toxic'] or result['image_toxic']
        return result

# Global variables
# No batch saving in the cyberbullying backend version

def generate_cyberbullying_report(df, target_channel):
    try:
        total_messages = len(df)
        if total_messages == 0:
            print_warning("No messages to analyze.")
            return

        toxic_count = df['Is_Toxic'].sum()
        toxicity_score = (toxic_count / total_messages) * 100

        report_lines = [
            f"Cyberbullying Analysis Report for: {target_channel}",
            f"{'-' * 50}",
            f"Total posts analyzed: {total_messages}",
            f"Total toxic posts detected: {toxic_count}",
            f"Overall Toxicity Score: {toxicity_score:.1f}%",
            "",
            "Interpretation:"
        ]

        if toxicity_score > 50:
            report_lines.append("CRITICAL: Severe amounts of cyberbullying/toxic behavior detected in this channel.")
            color = Fore.RED
        elif toxicity_score > 10:
            report_lines.append("CONCERNING: Moderate levels of toxic behavior identified.")
            color = Fore.YELLOW
        elif toxicity_score > 0:
            report_lines.append("WARNING: Some toxic content detected, proceed with caution.")
            color = Fore.LIGHTYELLOW_EX
        else:
            report_lines.append("SAFE: No cyberbullying content detected in the analyzed posts.")
            color = Fore.GREEN

        report_lines.append(f"\nBreakdown:")
        report_lines.append(f"Text-based toxic posts: {df['Text_Toxic'].sum()}")
        report_lines.append(f"Image-based toxic posts: {df['Image_Toxic'].sum()}")

        report_text = "\n".join(report_lines)
        with open('cyberbullying_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)

        print_header("FINAL ANALYSIS")
        print(f"{color}{report_text}{Style.RESET_ALL}")
        
    except Exception as e:
        print_error(f"Error generating report: {e}")

# Unused multiprocessing blocks removed

async def get_entity_name(entity):
    if isinstance(entity, User):
        return f"@{entity.username}" if entity.username else f"User({entity.id})"
    elif isinstance(entity, (Channel, Chat)):
        return entity.title or f"Channel({entity.id})"
    else:
        return f"Unknown({type(entity).__name__})"

async def scrape_messages(client, entity, message_limit, keywords, channel_manager, affiliated_channel=None):
    messages = []
    try:
        entity_name = await get_entity_name(entity)
        async for message in client.iter_messages(entity, limit=message_limit):
            if message.text or message.media:
                text_content = message.text if message.text else ""
                
                # Fetch image if present
                image_bytes = None
                if getattr(message, 'photo', None):
                    image_bytes = await client.download_media(message.photo, file=bytes)
                
                if affiliated_channel:
                    print_info(f"Message/Media from {Fore.CYAN}{Style.BRIGHT}{entity_name}{Style.RESET_ALL}.{Fore.YELLOW}{Style.BRIGHT} <-- {affiliated_channel}{Style.RESET_ALL}: {text_content[:50]}...")
                else:
                    print_info(f"Message/Media from {Fore.CYAN}{Style.BRIGHT}{entity_name}{Style.RESET_ALL}: {text_content[:50]}...")
                messages.append([message.sender_id, message.date, text_content, image_bytes])
                
                # Process t.me links in the message
                links = extract_channel_links(text_content)
                for link in links:
                    channel_manager.add_channel(link, source_channel=entity_name)
            
            await asyncio.sleep(0.1)
    except FloodWaitError as e:
        print_warning(f"FloodWaitError in scrape_messages: {e}")
        await asyncio.sleep(min(e.seconds, 30))
    except Exception as e:
        print_error(f"Error scraping entity {entity_name}: {e}")
    
    return messages, entity_name

async def process_channels(client, channel_manager, message_depth, keywords, batch_processor):
    while channel_manager.has_unprocessed_channels():
        link = channel_manager.get_next_channel()
        affiliated_channel = channel_manager.get_affiliation(link)
        try:
            join_success = await retry_with_backoff(join_channel(client, channel_manager, link))
            if join_success:
                entity = await client.get_entity(link)
                entity_messages, channel_name = await scrape_messages(client, entity, message_depth, keywords, channel_manager, affiliated_channel)
                
                # Add messages to batch processor with channel name and affiliation
                batch_processor.add_messages(entity_messages, channel_name, affiliated_channel)
            else:
                print_warning(f"Skipping entity {link} due to joining failure")
        except Exception as e:
            print_error(f"Failed to process entity {link}: {e}")
        finally:
            channel_manager.mark_as_processed(link)
        
        await asyncio.sleep(1)  # Small delay between processing channels

async def process_single_channel(client, channel_manager, link, message_depth, keywords):
    try:
        join_success = await retry_with_backoff(join_channel(client, channel_manager, link))
        if join_success:
            entity = await client.get_entity(link)
            entity_name = await get_entity_name(entity)
            print_info(f"Scraping messages from: {entity_name}")
            entity_messages = await scrape_messages(client, entity, message_depth, keywords, channel_manager)
            return entity_messages
        else:
            print_warning(f"Skipping entity {link} due to joining failure")
    except Exception as e:
        print_error(f"Failed to process entity {link}: {e}")
    return []

async def retry_with_backoff(coroutine, max_retries=5, base_delay=1, max_delay=60):
    retries = 0
    while True:
        try:
            return await coroutine
        except FloodWaitError as e:
            if retries >= max_retries:
                raise
            delay = min(base_delay * (2 ** retries) + random.uniform(0, 1), max_delay)
            print_warning(f"FloodWaitError encountered. Retrying in {delay:.2f} seconds. (Attempt {retries + 1}/{max_retries})")
            await asyncio.sleep(delay)
            retries += 1
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            raise



class CyberbullyingProcessor:
    def __init__(self, detector):
        self.detector = detector
        self.all_messages = []

    def add_messages(self, messages, channel_name, affiliated_channel):
        for msg in messages:
            # msg = [sender_id, date, text_content, image_bytes]
            text_content = msg[2]
            image_bytes = msg[3]
            
            analysis = self.detector.analyze(text_content, image_bytes)
            
            # Use 'True' or 'False' mapped directly from analysis details
            record = {
                'Sender ID': msg[0],
                'Date': msg[1],
                'Message': text_content,
                'Channel Name': channel_name,
                'Text_Toxic': analysis['text_toxic'],
                'Image_Toxic': analysis['image_toxic'],
                'Is_Toxic': analysis['is_toxic']
            }
            self.all_messages.append(record)
        print_success(f"Processed batch of {len(messages)} items for channel {channel_name}")

    def finalize(self, target_channel):
        df = pd.DataFrame(self.all_messages)
        generate_cyberbullying_report(df, target_channel)

# pretty much our main func at this point
# backend channel scraping runner
async def run_scraper(config, message_depth, channel_depth, specific_channel=None):
    await client.start()
    
    try:
        channel_manager = ChannelManager()
        detector = CyberbullyingDetector()
        processor = CyberbullyingProcessor(detector)
        
        if specific_channel:
            channel_manager.add_channel(specific_channel)
            target = specific_channel
        else:
            for link in config['initial_channel_links']:
                channel_manager.add_channel(link)
            target = "Configured Channels"
        
        start_time = datetime.now()
        print_header(f"Cyberbullying Background Scanner started at {start_time}")

        depth = 0
        while channel_manager.has_unprocessed_channels() and depth < channel_depth:
            print_subheader(f"Crawling at depth {depth + 1}/{channel_depth}")
            channel_manager.display_status()
            
            await process_channels(client, channel_manager, message_depth, config['message_keywords'], processor)
            
            depth += 1
            await asyncio.sleep(5)
        
        print_header(f"Scraping completed. Finalizing analysis...")
        processor.finalize(target)

    except Exception as e:
        print_error(f"An error occurred during scraping: {e}")
    finally:
        await client.disconnect()

async def process_all_channels(client, channel_manager, message_depth, keywords):
    all_messages = []
    channels_to_process = list(channel_manager.discovered_channels)
    
    for link in channels_to_process:
        try:
            join_success = await retry_with_backoff(join_channel(client, channel_manager, link))
            if join_success:
                entity = await client.get_entity(link)
                entity_name = await get_entity_name(entity)
                print_info(f"Scraping messages from: {entity_name}")
                entity_messages = await scrape_messages(client, entity, message_depth, keywords, channel_manager)
                all_messages.extend(entity_messages)
                
                # Process newly discovered channels
                new_channels = channel_manager.get_new_channels()
                for new_link in new_channels:
                    channel_manager.add_channel(new_link)
            else:
                print_warning(f"Skipping entity {link} due to joining failure")
        except Exception as e:
            print_error(f"Failed to process entity {link}: {e}")
        
        await asyncio.sleep(1)  # Small delay between processing channels
    
    return all_messages

async def process_discovered_channels(client, channel_manager, message_depth, keywords, max_channels_per_depth):
    channels_processed = 0
    while channel_manager.discovered_channels and channels_processed < max_channels_per_depth:
        link = channel_manager.get_next_channel()
        if await join_channel(client, channel_manager, link):
            try:
                channel = await client.get_entity(link)
                print_info(f"Scraping messages from newly discovered channel: {channel.title}")
                await scrape_messages(client, channel, message_depth, keywords, channel_manager)
                channels_processed += 1
            except Exception as e:
                print_error(f"Failed to scrape newly discovered channel {link}: {e}")
        
        await asyncio.sleep(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Telegram Content Crawler')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file')
    parser.add_argument('--message-depth', type=int, default=1000, help='Number of messages to crawl per channel')
    parser.add_argument('--channel-depth', type=int, default=2, help='Depth of channel crawling')
    parser.add_argument('--api-id', type=str, help='API ID for Telegram client')
    parser.add_argument('--api-hash', type=str, help='API hash for Telegram client')
    parser.add_argument('--phone-number', type=str, help='Phone number for Telegram client')
    parser.add_argument('--channel', type=str, default=None, help='Specific Telegram channel to scrape')
    args = parser.parse_args()

    config = load_config(args.config)
    if config is None and not args.channel:
        user_input = input(f"Config file '{args.config}' not found. Create a default config? (y/n): ")
        if user_input.lower() == 'y':
            config = create_default_config(args.config)
        else:
            print_error("Please provide a valid config file or specify --channel. Exiting.")
            exit(1)
    elif config is None:
        config = {"initial_channel_links": [], "message_keywords": [], "batch_size": 100}

    API_ID = "36191698"
    API_HASH = "fe0b2dbe66ea20681ce48b3b1ba4d95b"
    PHONE_NUMBER = "+917695884327" # Added +91 (India country code) or it will be marked invalid

    api_id = args.api_id or API_ID
    api_hash = args.api_hash or API_HASH
    phone_number = args.phone_number or PHONE_NUMBER

    if not api_id or not api_hash or not phone_number:
        print_error("API credentials are missing. Please provide them either as command-line arguments or in the script. (Line 664-666)")
        exit(1)

    client = TelegramClient('session_name', api_id, api_hash)

    with client:
        client.loop.run_until_complete(run_scraper(config, args.message_depth, args.channel_depth, args.channel))