#!/usr/bin/env bash
# exit on error
set -o errexit

echo "=== STARTING DEPLOYMENT BUILD SCRIPT ==="

# Upgrade pip
echo "1. Upgrading Pip..."
pip install --upgrade pip

# Install project requirements
echo "2. Installing requirements.txt..."
pip install -r requirements.txt

# Pre-download NLTK data to ensure zero runtime latency
echo "3. Pre-downloading NLTK datasets..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)"

# Pre-download DistilBERT model weights so the slug contains the pre-cached weights
echo "4. Pre-caching BERT transformer weights..."
python -c "from transformers import pipeline; pipeline('fill-mask', model='distilbert-base-uncased')"

echo "=== BUILD SCRIPT COMPLETED SUCCESSFULLY ==="
