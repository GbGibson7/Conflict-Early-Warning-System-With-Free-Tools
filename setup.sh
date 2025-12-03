#!/bin/bash
# setup.sh - For Streamlit Cloud deployment

echo "🚀 Setting up Conflict Early Warning System..."

# Install Python dependencies
pip install -r requirements-deploy.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Create necessary directories
mkdir -p data/raw data/processed models logs

echo "✅ Setup complete!"