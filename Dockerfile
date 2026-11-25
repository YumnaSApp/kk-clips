FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -U "yt-dlp[default]" \
    && (pip install --no-cache-dir deno || true)
WORKDIR /app
COPY . .
ENV PORT=8000 DATA_DIR=/app/data PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "app.py"]
