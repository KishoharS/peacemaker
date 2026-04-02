#!/usr/bin/env python3
"""
evaluate_multimodel.py
======================
Evaluates the multi-modal cyberbullying / hate-content detection system.

  Text model  : DistilBERT  →  ../models/bert_model
  Image model : ViT          →  ../models/vit_model
  Eval data   : xyyyy2025/hateful_memes_dataset  (has both image + text)
  Fusion      : Late fusion – weighted average of softmax probabilities

Usage
-----
  # Full evaluation
  python src/evaluate_multimodel.py

  # Quick smoke-test on first 200 samples
  python src/evaluate_multimodel.py --max-samples 200

  # Custom fusion weights
  python src/evaluate_multimodel.py --text-weight 0.3 --image-weight 0.7
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    ViTForImageClassification,
    ViTImageProcessor,
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
TEXT_MODEL_DIR = ROOT / "models" / "bert_model"
IMAGE_MODEL_DIR = ROOT / "models" / "vit_model"
RESULTS_DIR = ROOT / "eval_outputs"

# ── Reuse existing text cleaner ───────────────────────────────────────────────
sys.path.insert(0, str(SRC_DIR))
try:
    from preprocessing_text import clean_text
except ImportError:
    import re
    import string

    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\@\w+|\#", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        return " ".join(text.split())


# ── Device ────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Model loaders ─────────────────────────────────────────────────────────────
def load_text_model(device: torch.device):
    """Load DistilBERT from local checkpoint; fall back to base HF weights."""
    if TEXT_MODEL_DIR.exists():
        print(f"[Text model]  loading from: {TEXT_MODEL_DIR}")
        tokenizer = DistilBertTokenizer.from_pretrained(str(TEXT_MODEL_DIR))
        model = DistilBertForSequenceClassification.from_pretrained(str(TEXT_MODEL_DIR))
    else:
        print(
            "[Text model]  local checkpoint not found — loading base weights (untrained)."
        )
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
    return tokenizer, model.to(device).eval()


def load_image_model(device: torch.device):
    """Load ViT from local checkpoint; fall back to base HF weights."""
    if IMAGE_MODEL_DIR.exists():
        print(f"[Image model] loading from: {IMAGE_MODEL_DIR}")
        processor = ViTImageProcessor.from_pretrained(str(IMAGE_MODEL_DIR))
        model = ViTForImageClassification.from_pretrained(str(IMAGE_MODEL_DIR))
    else:
        print(
            "[Image model] local checkpoint not found — loading base weights (untrained)."
        )
        processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k", num_labels=2
        )
    return processor, model.to(device).eval()


# ── Dataset ───────────────────────────────────────────────────────────────────
def load_eval_data(max_samples: int = None):
    """
    Load xyyyy2025/hateful_memes_dataset.
    Returns (texts, images, labels_np_array, label2id_or_None).
    """
    print("Loading eval dataset: xyyyy2025/hateful_memes_dataset …")
    ds = load_dataset("xyyyy2025/hateful_memes_dataset")

    # Pick best available split
    split = next((s for s in ("test", "validation", "train") if s in ds), "train")
    data = ds[split]
    print(f"  Split: '{split}'  |  Rows: {len(data)}")

    cols = data.column_names

    # Normalise image column name
    if "img" in cols and "image" not in cols:
        data = data.rename_column("img", "image")
    elif "images" in cols and "image" not in cols:
        data = data.rename_column("images", "image")

    if "problem" in cols and "text" not in cols:
        data = data.rename_column("problem", "text")

    # Normalise label column name
    if "label" not in data.column_names:
        for alt in ("labels", "answer", "class"):
            if alt in cols:
                data = data.rename_column(alt, "label")
                break

    # Encode string labels → int  (sorted alphabetically e.g. no→0, yes→1)
    label2id = None
    if isinstance(data[0]["label"], str):
        unique = sorted(set(data["label"]))
        label2id = {lbl: i for i, lbl in enumerate(unique)}
        data = data.map(lambda ex: {"label": label2id[ex["label"]]})
        print(f"  Label encoding: {label2id}")

    if max_samples:
        data = data.select(range(min(max_samples, len(data))))

    # Find text column
    text_col = next(
        (c for c in ("text", "caption", "sentence") if c in data.column_names), None
    )
    texts = data[text_col] if text_col else [""] * len(data)
    images = data["image"]
    labels = np.array(data["label"])

    return texts, images, labels, label2id


# ── Batch prediction ──────────────────────────────────────────────────────────
@torch.no_grad()
def predict_text(texts, tokenizer, model, device, batch_size=32) -> np.ndarray:
    """Returns softmax probability array  (N, num_labels)."""
    all_probs = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = [clean_text(t) for t in texts[i : i + batch_size]]
        enc = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        print(f"  text  {min(i + batch_size, total)}/{total}", end="\r")
    print()
    return np.concatenate(all_probs, axis=0)


@torch.no_grad()
def predict_image(images, processor, model, device, batch_size=16) -> np.ndarray:
    """Returns softmax probability array  (N, num_labels)."""
    size = processor.size["height"]
    tfm = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )
    all_probs = []
    total = len(images)
    for i in range(0, total, batch_size):
        tensors = []
        for img in images[i : i + batch_size]:
            if isinstance(img, list):
                img = img[0]
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            else:
                img = img.convert("RGB")
            tensors.append(tfm(img))
        pixel_values = torch.stack(tensors).to(device)
        logits = model(pixel_values=pixel_values).logits
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        print(f"  image {min(i + batch_size, total)}/{total}", end="\r")
    print()
    return np.concatenate(all_probs, axis=0)


# ── Late fusion ───────────────────────────────────────────────────────────────
def late_fusion(
    text_probs: np.ndarray,
    image_probs: np.ndarray,
    text_w: float,
    image_w: float,
):
    """Weighted average of softmax probabilities → predicted class indices."""
    tw = text_w / (text_w + image_w)
    iw = image_w / (text_w + image_w)

    if text_probs.shape[1] != image_probs.shape[1]:
        # Both are binary so this shouldn't happen, but guard anyway
        print("  ⚠  Label-space mismatch — using image model output as fused result.")
        return image_probs.argmax(axis=1)

    fused = tw * text_probs + iw * image_probs
    return fused.argmax(axis=1)


# ── Metrics ───────────────────────────────────────────────────────────────────
def report_metrics(y_true, y_pred, title: str) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    rpt = classification_report(y_true, y_pred, zero_division=0)

    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (macro)")
    print(f"  Recall    : {rec:.4f}  (macro)")
    print(f"  F1 Score  : {f1:.4f}  (macro)")
    print(f"\n{rpt}")
    print(f"  Confusion Matrix:\n{cm}\n")

    return {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec), 4),
        "recall_macro": round(float(rec), 4),
        "f1_macro": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": rpt,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multi-modal cyberbullying evaluator")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap eval at N samples (handy for quick tests)",
    )
    parser.add_argument(
        "--text-weight",
        type=float,
        default=0.4,
        help="Fusion weight for the text model  (default: 0.4)",
    )
    parser.add_argument(
        "--image-weight",
        type=float,
        default=0.6,
        help="Fusion weight for the image model (default: 0.6)",
    )
    parser.add_argument("--batch-size-text", type=int, default=32)
    parser.add_argument("--batch-size-image", type=int, default=16)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}\n")

    # 1. Load models
    tokenizer, text_model = load_text_model(device)
    processor, image_model = load_image_model(device)

    # 2. Load evaluation data
    texts, images, labels, label2id = load_eval_data(max_samples=args.max_samples)
    print(f"\nSamples : {len(labels)}")
    print(f"Classes : {np.bincount(labels)}  (index = class id)\n")

    # 3. Run both models
    print("[1/3] Text model predictions …")
    text_probs = predict_text(
        texts, tokenizer, text_model, device, args.batch_size_text
    )

    print("[2/3] Image model predictions …")
    image_probs = predict_image(
        images, processor, image_model, device, args.batch_size_image
    )

    print(f"[3/3] Fusing  (text={args.text_weight}, image={args.image_weight}) …\n")
    fused_preds = late_fusion(
        text_probs, image_probs, args.text_weight, args.image_weight
    )

    # 4. Evaluate each model + fused
    results = {
        "text_model": report_metrics(
            labels, text_probs.argmax(axis=1), "Text Model  (DistilBERT)"
        ),
        "image_model": report_metrics(
            labels, image_probs.argmax(axis=1), "Image Model (ViT)"
        ),
        "fused": report_metrics(
            labels,
            fused_preds,
            f"Fused  (text_w={args.text_weight} / image_w={args.image_weight})",
        ),
        "meta": {
            "num_samples": int(len(labels)),
            "device": str(device),
            "text_weight": args.text_weight,
            "image_weight": args.image_weight,
            "label2id": label2id,
        },
    }

    # 5. Save JSON report
    out_path = RESULTS_DIR / "multimodel_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {out_path}")


if __name__ == "__main__":
    main()
