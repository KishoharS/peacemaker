# Cyberbullying Detection Project

## Overview
This project detects cyberbullying in text using Machine Learning. It includes a training pipeline and a web interface for real-time predictions.

## Structure
- `data/`: Datasets.
- `models/`: Saved trained models.
- `src/`: Source code for preprocessing and training.
- `app/`: Streamlit web application.
- `tests/`: Unit tests.

## Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Generate synthetic data (if no real data present):
   ```bash
   python src/data_generation.py
   ```
2. Train the model:
   ```bash
   python src/train.py
   ```
3. Run the web app:
   ```bash
   streamlit run app/app.py
   ```
