#!/bin/bash
# Start KK Clips on this Mac (keep this window open while using it).
cd "$HOME/kk-clips"
for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do [ -x "$b" ] && eval "$($b shellenv)"; done
[ -f .env ] && source .env
export PATH="$PWD/.venv/bin:$PATH" PORT=8000 DATA_DIR="$HOME/kk-clips/data"
IP=$(ipconfig getifaddr en0 || ipconfig getifaddr en1)
echo
echo "=============================================="
echo " KK Clips is running."
echo " On this Mac:   http://localhost:8000"
echo " On your iPhone (same Wi-Fi):  http://$IP:8000"
echo " Keep this window open. Close it to stop."
echo "=============================================="
.venv/bin/python app.py
