# Cyberbullying Detection Project

## Overview
This project detects cyberbullying in text and images using Machine Learning (DistilBERT and Vision Transformer). It includes a training pipeline and a web interface for real-time predictions.

## Structure
- `models/`: Saved trained models (Ignored in Git, generated via training scripts).
- `src/`: Source code for preprocessing and training pipelines.
- `app/`: Streamlit web application.
- `tests/`: Unit tests.

## Installation & Setup
To run this project on a new local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/KishoharS/cyberbullying_detection_system_using_machinelearning.git
   cd cyberbullying_detection_system_using_machinelearning
   ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For the audio analyzer, you may also need to install `ffmpeg` on your system (e.g., `brew install ffmpeg` on macOS, or via `apt` on Linux).*

## Generating the Models
**Important:** The trained model files are too large to host on GitHub directly. Before running the web application, you **must train the models** locally to generate the weights in the `models/` directory.

1. **Train the Text Model (DistilBERT):**
   ```bash
   python src/train_text.py
   ```
2. **Train the Image Model (ViT):**
   ```bash
   python src/train_image.py
   ```

## Usage
Once the models are generated and saved in the `models/` directory, you can launch the Streamlit frontend.

1. **Run the web app:**
   ```bash
   streamlit run app/app.py
   ```
2. Open your browser to the URL provided in the terminal (usually `http://localhost:8501`).
