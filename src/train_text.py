# basic imports
print('Starting script...')
import pandas as pd
import joblib
import os

# for machine learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset

def compute_metrics(pred):
    labels = pred.label_ids
    pred = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, pred)
    return {'accuracy': acc}

def train_bert_model():
    print('loading dateset')
    ds = load_dataset("karthikarunr/Cyberbullying-Toxicity-Tweets")
    df = pd.DataFrame(ds['train'])

    # rename columns to standard 'text' and 'label'
    if 'Text' in df.columns and 'oh_label' in df.columns:
        df = df.rename(columns = {'Text': 'text', 'oh_label': 'label'})

    # split
    train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 42)

    # convert back to hugging face dataset
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    # 2.Tokenization
    print('Tokenizing data')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding = 'max_length', truncation = True, max_length = 128)

    tokenized_train = train_dataset.map(tokenize_function, batched = True)
    tokenized_test = test_dataset.map(tokenize_function, batched =True)

    # 3.Model setup
    print('Initializing model')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 2)

    # 4.Training arguments
    training_args = TrainingArguments(
        output_dir = './results',
        num_train_epochs = 3,
        per_device_train_batch_size = 32,
        per_device_eval_batch_size = 32,
        warmup_steps = 500,
        weight_decay = 0.01,
        logging_dir = './logs',
        logging_steps = 100,
        eval_strategy = 'epoch',
        fp16 = True,
        dataloader_num_workers = 0,
        report_to = 'none'
    )

    # 5.Trainer
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_train,
        eval_dataset = tokenized_test,
        compute_metrics = compute_metrics,
    )

    # 6. Train
    print('Starting training...')
    trainer.train()

    # 7. Save
    print('Saving model')
    model.save_pretrained('./models/bert_model')
    tokenizer.save_pretrained('./models/bert_model')
    print('Model saved to ./models/bert_model')

if __name__ == '__main__':
    train_bert_model()
