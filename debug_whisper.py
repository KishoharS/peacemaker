import os
import sys

import whisper

print(f"Python version: {sys.version}")
print(f"Whisper file: {whisper.__file__}")
print(f"Whisper dir: {dir(whisper)}")

try:
    model = whisper.load_model("base")
    print("Successfully loaded model!")
except AttributeError as e:
    print(f"Caught expected error: {e}")
except Exception as e:
    print(f"Caught unexpected error: {type(e).__name__}: {e}")
