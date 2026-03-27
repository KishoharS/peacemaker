kimport torch
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from PIL import Image

def train_image_model():
    print("Initializing Image Training for 'xyyyy2025/hateful_memes_dataset'...")

    # 1. Load Dataset from Hugging Face
    dataset_name = "xyyyy2025/hateful_memes_dataset"
    try:
        # Load dataset
        # This will automatically download images if they are in Parquet/Arrow format
        ds = load_dataset(dataset_name)
        print(f"Dataset loaded: {ds}")
        print(f"Features: {ds['train'].features}")

        # Rename columns if needed
        # We need 'image' column for input and 'label' for target
        # Check available columns
        cols = ds['train'].column_names

        # Standardize 'image' column
        if 'img' in cols and 'image' not in cols:
            ds = ds.rename_column("img", "image")
        elif 'text' in cols and 'image' not in cols:
            # Maybe it's multi-modal text + image?
            # But ViT needs image.
            pass

        # Standardize 'label' column
        if 'label' not in cols:
             # Check for common alternatives
             if 'labels' in cols:
                 ds = ds.rename_column("labels", "label")
             elif 'class' in cols:
                 ds = ds.rename_column("class", "label")
             elif 'answer' in cols: # Added for xyyyy2025/hateful_memes_dataset
                 ds = ds.rename_column("answer", "label")

        # Check if we have image column processing
        if 'image' not in ds['train'].column_names:
            print(f"Error: Dataset {dataset_name} does not seem to have an 'image' or 'img' column.")
            print(f"Columns found: {ds['train'].column_names}")
            # Try to handle if image is a path string?
            # If it's a string path, we need to load it.
            # But `load_dataset` usually handles images as Image feature if configured correctly.
            print("Attempting to deduce if image column is named differently...")
            image_col = None
            for col in ds['train'].column_names:
                if 'img' in col or 'image' in col:
                    image_col = col
                    break
            if image_col:
                ds = ds.rename_column(image_col, "image")
                print(f"Renamed column '{image_col}' to 'image'.")
            else:
                 return # Cannot proceed without images

        # Handle string labels (e.g. "yes"/"no", "hateful"/"not_hateful")
        if isinstance(ds['train'].features['label'], torch.dtype) == False and \
           not hasattr(ds['train'].features['label'], 'names'): # Ensure it's not already ClassLabel

            # Get unique labels
            unique_labels = sorted(list(set(ds['train']['label'])))
            print(f"Found String Labels: {unique_labels}. Encoding to Integers...")

            label2id = {label: i for i, label in enumerate(unique_labels)}
            id2label = {i: label for i, label in enumerate(unique_labels)}

            def encode_labels(example):
                example['label'] = label2id[example['label']]
                return example

            ds = ds.map(encode_labels)
            # Update detection logic later to use this map

    except Exception as e:
        print(f"Critical Error loading dataset: {e}")
        return

    # 2. Config & Preprocessing
    model_name = "google/vit-base-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(model_name)

    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]
    normalize = Normalize(mean=image_mean, std=image_std)

    _train_transforms = Compose([
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ])

    _val_transforms = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ])

    def transform(examples, transforms):
        # Handle if image is path string vs PIL Image
        images = []
        for img in examples['image']:
            # Handle List of Images (taking first one)
            if isinstance(img, list):
                img = img[0]

            if isinstance(img, str):
                # It's a path, load it.
                # Assuming local path if string? Or URL?
                # HF datasets usually return PIL images directly for Image feature.
                try:
                    img = Image.open(img).convert("RGB")
                    images.append(img)
                except Exception as e:
                    # Fallback or error
                    print(f"Error loading image path {img}: {e}")
                    # Use skipped image or dummy? Better to crash early to debug.
                    return None
            else:
                # Assuming PIL Image
                images.append(img.convert("RGB"))

        examples['pixel_values'] = [transforms(img) for img in images]
        return examples

    def train_transforms(examples):
        return transform(examples, _train_transforms)

    def val_transforms(examples):
        return transform(examples, _val_transforms)

    # 3. Splits
    if 'validation' not in ds:
        # Create split
        ds = ds['train'].train_test_split(test_size=0.1)

    ds['train'].set_transform(train_transforms)
    # Handle test/validation naming
    val_key = 'validation' if 'validation' in ds else 'test'
    ds[val_key].set_transform(val_transforms)

    # 4. Model Setup
    # Auto-detect number of labels
    # Use id2label/label2id from encoding step if available
    if 'id2label' not in locals():
         # If ClassLabel feature
        if hasattr(ds['train'].features['label'], 'names'):
            labels = ds['train'].features['label'].names
            id2label = {str(i): label for i, label in enumerate(labels)}
            label2id = {label: str(i) for i, label in enumerate(labels)}
        else:
            # Fallback (should be covered by encoding step)
            unique_labels = sorted(list(set(ds['train']['label'])))
            labels = [str(l) for l in unique_labels]
            id2label = {str(i): label for i, label in enumerate(labels)}
            label2id = {label: str(i) for i, label in enumerate(labels)}
    else:
        # Use existing (convert keys to string for JSON compat if needed, but int keys fine for dict)
        # Verify alignment
        labels = list(label2id.keys())
        # Ensure keys are strings for config
        id2label = {str(k): v for k, v in id2label.items()}


    print(f"Detected Labels: {labels}")

    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # 5. Trainer
    training_args = TrainingArguments(
        output_dir="./results_image",
        per_device_train_batch_size=16,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        fp16=True,
        learning_rate=2e-5,
        save_total_limit=2,
        dataloader_num_workers=2, # Usually safe on Kaggle unless constrained resources
        report_to='none',
    )

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds[val_key],
        tokenizer=processor,
        data_collator=collate_fn,
    )

    # 6. Train
    print("Starting training...")
    trainer.train()

    # 7. Save
    print("Saving model...")
    model.save_pretrained("./models/vit_model")
    processor.save_pretrained("./models/vit_model")
    print("Model saved to ./models/vit_model")

if __name__ == "__main__":
    train_image_model()
