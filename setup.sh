#!/bin/bash
# VoiceTranslate AI — Setup Script
set -e

echo "========================================="
echo "  VoiceTranslate AI — Setup"
echo "========================================="

# Check Python
python3 --version || { echo "Python 3 is required"; exit 1; }

# Check ffmpeg
if ! command -v ffmpeg &>/dev/null; then
  echo "Installing ffmpeg..."
  if [[ "$OSTYPE" == "darwin"* ]]; then
    brew install ffmpeg
  else
    sudo apt-get install -y ffmpeg
  fi
fi

# Create virtual environment
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU — change to cu118 for CUDA GPU)
echo "Installing PyTorch..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install main requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Install gTTS as Tamil TTS fallback
pip install gtts

# Copy .env if not exists
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo ""
  echo "⚠️  Created .env — please fill in your API keys:"
  echo "   ANTHROPIC_API_KEY  — from console.anthropic.com"
  echo "   HF_TOKEN           — from huggingface.co/settings/tokens"
fi

echo ""
echo "========================================="
echo "  Setup complete!"
echo "  Run:  source venv/bin/activate && python app.py"
echo "  Open: http://localhost:5000"
echo "========================================="
