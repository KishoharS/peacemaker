import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import os

model_path = '/Users/kishohars/Projects/cyberbullying_detection_project/models/bert_model'

try:
    print("Loading model...")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    
    test_texts = ['fuck you', 'I hate you', 'You are great', 'Go kill yourself', 'hello how are you']
    
    for text in test_texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        
        print(f'\nText: "{text}"')
        print(f'Predicted class: {predicted_class}')
        print(f'  No bullying prob: {probabilities[0][0].item():.4f}')
        print(f'  Bullying prob: {probabilities[0][1].item():.4f}')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
