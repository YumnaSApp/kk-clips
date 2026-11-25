#!/bin/bash
# KK Clips — one-time Mac setup. Run in Terminal:
#   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/YumnaSApp/kk-clips/main/mac/setup.sh)"
set -e
echo "== KK Clips Mac setup =="

# 1. Homebrew
if ! command -v brew >/dev/null 2>&1; then
  for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do [ -x "$b" ] && eval "$($b shellenv)"; done
fi
if ! command -v brew >/dev/null 2>&1; then
  echo "Installing Homebrew (it may ask for your Mac password)..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do [ -x "$b" ] && eval "$($b shellenv)"; done
fi

# 2. Tools
brew install python@3.12 ffmpeg deno git

# 3. App code
APP="$HOME/kk-clips"
if [ -d "$APP/.git" ]; then git -C "$APP" pull -q; else git clone -q https://github.com/YumnaSApp/kk-clips "$APP"; fi
cd "$APP"
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
.venv/bin/pip install -q -U "yt-dlp[default]"

# 4. Groq key (saved only on this Mac)
if [ ! -f .env ]; then
  echo
  echo "Paste your GROQ API key (same one as on Render) and press Enter:"
  read -r -s GROQ < /dev/tty
  printf 'export GROQ_API_KEY=%q\n' "$GROQ" > .env
  chmod 600 .env
fi

# 5. Desktop launcher
cat > "$HOME/Desktop/KK Clips.command" <<'L'
#!/bin/bash
cd "$HOME/kk-clips" && git pull -q 2>/dev/null; bash mac/start.sh
L
chmod +x "$HOME/Desktop/KK Clips.command"

echo
echo "Setup done! Starting KK Clips..."
bash mac/start.sh
