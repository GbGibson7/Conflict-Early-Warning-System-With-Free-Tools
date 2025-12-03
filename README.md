# ⚠️   Conflict-Early-Warning-System-With-Free-Tools
An open-source conflict early warning system (CEWS) designed to monitor, forecast, and alert on potential conflict risks. This platform integrates socioeconomic, political, and event data using statistical and machine learning models to provide actionable insights for researchers, NGOs, and policymakers.

A 100% free, open-source early warning system for conflict detection and prediction.

## 🌟 Features
- **Real-time Data Collection**: NewsAPI, Reddit, GDELT
- **AI-Powered Analysis**: NLP sentiment analysis + ML prediction
- **Interactive Dashboard**: Streamlit web interface
- **Geographic Visualization**: Heatmaps and location tracking
- **Early Warning Alerts**: Risk level classification

## 🚀 Quick Start

### 1. Local Development
```bash
# Clone repository
git clone https://github.com/yourusername/conflict-early-warning-free.git
cd conflict-early-warning-free

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Get API keys (free)
# 1. NewsAPI: https://newsapi.org/register
# 2. Reddit: https://www.reddit.com/prefs/apps

# Run dashboard
streamlit run dashboard/app.py