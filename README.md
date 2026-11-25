# KK Clips — AI viral clip maker

Upload a long video or paste a link. The AI then:
1. transcribes it (Whisper),
2. picks the most viral moments and scores each one out of 100,
3. renders vertical 9:16 clips (1080×1920) with word-by-word captions and a multi-colour hook title,
4. writes a title, description and hashtags for each clip.

## What you need
- A free **Groq API key** (console.groq.com → API Keys). The same key handles transcription and the AI.
- A server with ffmpeg. The included Dockerfile installs everything.

## Run on Render
1. Put this folder in a GitHub repo.
2. In Render, create a **New → Blueprint** and choose the repo (it reads `render.yaml`).
3. Set `GROQ_API_KEY` and `APP_PASSWORD` (your login password for the site).
4. Open the URL on your phone and add it to your Home Screen.

The Starter plan ($7/mo) is recommended. The free plan is very slow at video rendering and sleeps when idle.

## Run on your Mac
```
brew install ffmpeg yt-dlp
GROQ_API_KEY=your_key python3 app.py
```
Then open http://localhost:8000

## Settings (environment variables)
| Name | Default | What it does |
|---|---|---|
| GROQ_API_KEY | — | AI key (required) |
| APP_PASSWORD | — | Password for the site |
| LLM_MODEL | llama-3.3-70b-versatile | AI that picks the moments |
| STT_MODEL | whisper-large-v3-turbo | Transcription model |
| LLM_URL / LLM_API_KEY | Groq | Any OpenAI-compatible AI service |
| YTDLP_COOKIES | — | Path to cookies.txt if YouTube blocks link downloads |
| YTDLP_COOKIES_TXT | — | Paste the full cookies.txt text here (easier than a file on Render) |
| KEEP_JOBS | 15 | How many past videos are kept |
| FFMPEG_PRESET | veryfast | Render speed/quality setting |

Note: YouTube often blocks downloads from cloud servers. If a link fails, download the video and upload it instead.
Only clip videos you have the rights to post.
