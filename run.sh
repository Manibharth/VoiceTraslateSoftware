#!/bin/bash
# VoiceTranslate AI — always uses the venv Python directly
DIR="$(cd "$(dirname "$0")" && pwd)"
"$DIR/venv/bin/python" "$DIR/app.py"
