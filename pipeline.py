"""KK Clips pipeline: long video -> AI-picked viral moments -> 9:16 clips with captions + hook.

Steps: get video -> extract audio -> transcribe (Whisper via Groq) -> AI picks moments
-> render vertical clips (blur-fill or center-crop, word-highlight captions, hook title).
Uses only the Python standard library + ffmpeg (+ yt-dlp for links).
"""
import json, math, os, re, shutil, subprocess, time, uuid, urllib.request, urllib.error

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
STT_URL = os.environ.get("STT_URL", "https://api.groq.com/openai/v1/audio/transcriptions")
STT_KEY = os.environ.get("STT_API_KEY", GROQ_KEY)
STT_MODEL = os.environ.get("STT_MODEL", "whisper-large-v3-turbo")
LLM_URL = os.environ.get("LLM_URL", "https://api.groq.com/openai/v1/chat/completions")
LLM_KEY = os.environ.get("LLM_API_KEY", GROQ_KEY)
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
FONTS_DIR = os.environ.get("FONTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts"))
CAPTION_FONT = os.environ.get("CAPTION_FONT", "Anton")
MOCK = os.environ.get("KK_MOCK") == "1"  # offline test mode: fake transcript + picks
YTDLP_COOKIES = os.environ.get("YTDLP_COOKIES", "")  # optional path to cookies.txt
PRESET = os.environ.get("FFMPEG_PRESET", "veryfast")
THREADS = os.environ.get("FFMPEG_THREADS", "2")


class PipelineError(Exception):
    pass


def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise PipelineError(f"{cmd[0]} failed: {p.stderr[-800:]}")
    return p.stdout


def probe_duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path])
    return float(json.loads(out)["format"]["duration"])


def probe_has_audio(path):
    out = run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "json", path])
    return bool(json.loads(out).get("streams"))


# ---------------------------------------------------------------- input

def _cookies_file(job_dir):
    """Cookies from a file path (YTDLP_COOKIES) or pasted text (YTDLP_COOKIES_TXT)."""
    if YTDLP_COOKIES and os.path.exists(YTDLP_COOKIES):
        return YTDLP_COOKIES
    txt = os.environ.get("YTDLP_COOKIES_TXT", "").replace("\\t", "\t").replace("\\n", "\n").strip()
    if not txt:
        return ""
    lines = []
    for line in txt.splitlines():
        line = line.strip()
        if not line or (line.startswith("#") and not line.startswith("#HttpOnly_")):
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) >= 7:  # domain flag path secure expiry name value
            lines.append("\t".join(parts[:6] + [" ".join(parts[6:])]))
    if not lines:
        raise PipelineError("The YTDLP_COOKIES_TXT value on Render is not a cookies file. "
                            "Open cookies.txt, copy ALL the text inside it, and paste that as the value (not the file name).")
    if not any("youtube.com" in l.split("\t")[0] for l in lines):
        raise PipelineError("The cookies on Render are not YouTube cookies. Open youtube.com in Chrome first, "
                            "then click the extension and Copy, and paste that on Render.")
    path = os.path.join(job_dir, "cookies.txt")
    with open(path, "w") as f:
        f.write("# Netscape HTTP Cookie File\n" + "\n".join(lines) + "\n")
    return path


# Different YouTube "clients" — if one is blocked from the server, the next often works.
YT_CLIENTS = [None, "tv_simply", "mweb", "web_safari", "android_vr", "ios"]


def download_link(url, job_dir):
    """Download a YouTube/other link with yt-dlp (max 1080p), trying several clients."""
    if not shutil.which("yt-dlp"):
        raise PipelineError("yt-dlp is not installed on the server.")
    out_tmpl = os.path.join(job_dir, "source.%(ext)s")
    base = ["yt-dlp", "--no-playlist", "-f",
            "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b",
            "--merge-output-format", "mp4", "-o", out_tmpl, "--print", "title", "--no-simulate",
            "--retries", "3", "--remote-components", "ejs:github"]
    deno = shutil.which("deno")
    if deno:  # JS runtime YouTube now requires to unlock video formats
        base += ["--js-runtimes", "deno:" + deno]
    cookies = _cookies_file(job_dir)
    if cookies:
        base += ["--cookies", cookies]
    is_yt = "youtu" in url
    err = ""
    clients = YT_CLIENTS if is_yt else [None]
    if is_yt and cookies:  # these clients support logged-in cookies
        clients = [None, "web", "mweb", "tv", "web_safari"]
    first_err = ""
    for client in clients:
        cmd = list(base)
        if client:
            cmd += ["--extractor-args", f"youtube:player_client={client}"]
        p = subprocess.run(cmd + [url], capture_output=True, text=True)
        if p.returncode == 0:
            title = (p.stdout.strip().splitlines() or ["Video"])[0]
            for f in os.listdir(job_dir):
                if f.startswith("source.") and not f.endswith(".part"):
                    return os.path.join(job_dir, f), title
        err = "\n".join(l for l in p.stderr.splitlines() if "WARNING" in l or "ERROR" in l)[-700:] or p.stderr[-500:]
        print(f"[yt-dlp client={client}] rc={p.returncode}\n{p.stderr[-1500:]}", flush=True)
        first_err = first_err or err
        for f in os.listdir(job_dir):  # clean partial files before next try
            if f.startswith("source."):
                os.remove(os.path.join(job_dir, f))
    if is_yt and any(k in err for k in ("403", "Sign in", "bot", "player response")):
        raise PipelineError("YouTube blocked the server from downloading this video. "
                            "Quick fix: download the video on your phone and use Upload. "
                            + ("Your cookies were used but YouTube still refused (" + err.strip().splitlines()[-1][:200] + ")."
                               if cookies else "Permanent fix: add your YouTube cookies (YTDLP_COOKIES_TXT) on Render."))
    raise PipelineError("Link download failed: " + (first_err or err))


# ---------------------------------------------------------------- HTTP helpers

def _post(url, key, body, headers, timeout=600):
    for attempt in range(6):
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Authorization": f"Bearer {key}", "User-Agent": "Mozilla/5.0 (KKClips/1.0)",
                                              "Accept": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="ignore")[:500]
            if e.code in (429, 500, 502, 503) and attempt < 5:
                wait = float(e.headers.get("retry-after") or 0) or 10 * (attempt + 1)
                time.sleep(min(wait, 90))
                continue
            if e.code == 401:
                raise PipelineError("AI key is wrong or missing (check GROQ_API_KEY).")
            raise PipelineError(f"AI service error {e.code}: {msg}")
        except urllib.error.URLError as e:
            if attempt < 5:
                time.sleep(5)
                continue
            raise PipelineError(f"Could not reach AI service: {e}")


def _multipart(fields, file_field, file_path):
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    with open(file_path, "rb") as f:
        data = f.read()
    parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
                  f'filename="{os.path.basename(file_path)}"\r\nContent-Type: audio/mpeg\r\n\r\n').encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


# ---------------------------------------------------------------- transcription

def transcribe(video, job_dir, language=None, progress=lambda m: None):
    """Returns {"segments":[{start,end,text}], "words":[{w,start,end}]}"""
    if MOCK:
        return _mock_transcript(probe_duration(video))
    if not STT_KEY:
        raise PipelineError("No AI key set. Add GROQ_API_KEY on the server (free at console.groq.com).")
    if not probe_has_audio(video):
        return {"segments": [], "words": []}  # no sound: clips get picked without speech
    chunk_len = 1200  # 20 min per chunk (~5 MB at 32 kbps) stays under the 25 MB upload limit
    adir = os.path.join(job_dir, "audio")
    os.makedirs(adir, exist_ok=True)
    run(["ffmpeg", "-y", "-i", video, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k",
         "-f", "segment", "-segment_time", str(chunk_len), "-reset_timestamps", "1",
         os.path.join(adir, "part%03d.mp3")])
    parts = sorted(f for f in os.listdir(adir) if f.endswith(".mp3"))
    segments, words = [], []
    offset = 0.0
    for i, part in enumerate(parts):
        progress(f"Transcribing audio {i + 1}/{len(parts)}")
        path = os.path.join(adir, part)
        fields = [("model", STT_MODEL), ("response_format", "verbose_json"),
                  ("timestamp_granularities[]", "word"), ("timestamp_granularities[]", "segment")]
        if language:
            fields.append(("language", language))
        body, ctype = _multipart(fields, "file", path)
        res = _post(STT_URL, STT_KEY, body, {"Content-Type": ctype})
        for s in res.get("segments") or []:
            segments.append({"start": s["start"] + offset, "end": s["end"] + offset, "text": s["text"].strip()})
        for w in res.get("words") or []:
            words.append({"w": w["word"].strip(), "start": w["start"] + offset, "end": w["end"] + offset})
        offset += probe_duration(path)
    if not words and segments:  # fallback: spread words evenly inside segments
        for s in segments:
            toks = s["text"].split()
            step = (s["end"] - s["start"]) / max(len(toks), 1)
            for k, t in enumerate(toks):
                words.append({"w": t, "start": s["start"] + k * step, "end": s["start"] + (k + 1) * step})
    print(f"[kk] transcript: {len(segments)} segments, {len(words)} words", flush=True)
    return {"segments": segments, "words": words}


def _mock_transcript(duration):
    vocab = ("this is the moment everyone was waiting for and honestly nobody expected "
             "what happened next you have to see this crazy part right here").split()
    words, t, i = [], 0.5, 0
    while t < duration - 0.5:
        words.append({"w": vocab[i % len(vocab)], "start": t, "end": t + 0.32})
        t += 0.4
        i += 1
    segs = []
    for k in range(0, len(words), 12):
        chunk = words[k:k + 12]
        segs.append({"start": chunk[0]["start"], "end": chunk[-1]["end"], "text": " ".join(w["w"] for w in chunk)})
    return {"segments": segs, "words": words}


# ---------------------------------------------------------------- AI moment picking

PICK_PROMPT = """You are a top YouTube Shorts / Instagram Reels editor. From the timestamped transcript of a long video, pick the {n} best moments to post as standalone viral short clips.

A great clip:
- starts with a strong hook in the first 2 seconds (a bold claim, question, conflict, surprise or funny line) — never starts mid-sentence
- tells one complete idea or story and ends on a payoff, punchline or strong line
- makes sense with zero context from the rest of the video
- is between {min_len} and {max_len} seconds long
Clips must not overlap. Use the exact timestamps from the transcript lines.

Reply with JSON only:
{{"clips":[{{"start":<seconds>,"end":<seconds>,"score":<1-100 viral potential>,
"hook":"<2-6 word on-screen hook in English, ALL CAPS, no emojis>",
"title":"<YouTube Shorts title in English, under 90 characters, with 1-2 emojis and #shorts>",
"description":"<2-3 sentence keyword-rich English description>",
"hashtags":["<5-8 lowercase hashtags without #>"],
"why":"<one short sentence on why it will go viral>"}}]}}

Video title: {video_title}
Transcript (each line is [start-end] text):
{transcript}"""


def _lines(segments, t0, t1):
    out = []
    for s in segments:
        if s["end"] < t0 or s["start"] > t1:
            continue
        out.append(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}")
    return "\n".join(out)


_MODEL_CACHE = {}
PREFERRED = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b",
             "meta-llama/llama-4-maverick-17b-128e-instruct", "qwen/qwen3-32b", "llama-3.1-8b-instant"]


def _pick_model():
    """Use LLM_MODEL if this key can use it, otherwise the best model the key has access to."""
    if "m" in _MODEL_CACHE:
        return _MODEL_CACHE["m"]
    model = LLM_MODEL
    try:
        url = LLM_URL.rsplit("/chat/completions", 1)[0] + "/models"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {LLM_KEY}",
                                                   "User-Agent": "Mozilla/5.0 (KKClips/1.0)"})
        with urllib.request.urlopen(req, timeout=30) as r:
            ids = [m["id"] for m in json.loads(r.read().decode()).get("data", []) if m.get("active", True)]
        chat = [i for i in ids if not any(x in i for x in ("whisper", "guard", "tts", "orpheus", "distil"))]
        print(f"[kk] models available: {chat}", flush=True)
        _MODEL_CACHE["all"] = chat
        if model not in chat:
            model = next((m for m in PREFERRED if m in chat), chat[0] if chat else model)
    except Exception as e:
        print(f"[kk] model list failed: {e}", flush=True)
    _MODEL_CACHE["m"] = model
    print(f"[kk] using model {model}", flush=True)
    return model


def _vision_model():
    """A model on this key that can look at images (for titles/descriptions of what's in the clip)."""
    if "v" in _MODEL_CACHE:
        return _MODEL_CACHE["v"]
    _pick_model()
    chat = _MODEL_CACHE.get("all", [])
    v = next((m for m in chat if "llama-4-scout" in m), None) or \
        next((m for m in chat if "llama-4-maverick" in m or "vision" in m or "-vl" in m), None)
    _MODEL_CACHE["v"] = v
    print(f"[kk] vision model: {v}", flush=True)
    return v


def _chat_json(prompt, images=None, model=None):
    model = model or _pick_model()
    content = prompt
    if images:
        content = [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b}} for b in images]
    msg = [{"role": "user", "content": content}]
    body = {"model": model, "temperature": 0.4, "response_format": {"type": "json_object"}, "messages": msg}
    try:
        res = _post(LLM_URL, LLM_KEY, json.dumps(body).encode(), {"Content-Type": "application/json"}, timeout=180)
    except PipelineError as e:
        if "404" in str(e) or "model" in str(e).lower():
            _MODEL_CACHE.pop("m", None)
        if "400" not in str(e):
            raise
        body.pop("response_format")  # some models reject JSON mode
        res = _post(LLM_URL, LLM_KEY, json.dumps(body).encode(), {"Content-Type": "application/json"}, timeout=180)
    content = res["choices"][0]["message"]["content"] or ""
    m = re.search(r"\{.*\}", content, re.S)
    return m.group(0) if m else content


def _parse_t(v):
    """Accepts 12.5, "12.5", "1:23", "01:02:03", "[12.5-20]"."""
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().strip("[]s ")
    s = s.split("-")[0].strip() if re.match(r"^[\d.:]+\s*-\s*[\d.:]+$", s) else s
    if ":" in s:
        t = 0.0
        for part in s.split(":"):
            t = t * 60 + float(part)
        return t
    return float(s)


def _energy_picks(video, duration, n, min_len, max_len, taken):
    """Fallback for videos with little/no speech (music, raves, sports): pick the loudest moments."""
    L = max(min_len, min(max_len, 30, duration / max(1, n)))
    step = 0.5
    try:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", "4000",
                              "-f", "s16le", "-"], capture_output=True).stdout
        import array
        a = array.array("h", raw)
        per = int(4000 * step)
        rms = [math.sqrt(sum(x * x for x in a[i:i + per][::4]) / max(1, len(a[i:i + per][::4])))
               for i in range(0, len(a), per)]
    except Exception:
        rms = []
    if not rms:
        rms = [1.0] * int(duration / step)
    win = max(1, int(L / step))
    scores = []
    run_sum = sum(rms[:win])
    for i in range(0, max(1, len(rms) - win)):
        scores.append((run_sum, i * step))
        if i + win < len(rms):
            run_sum += rms[i + win] - rms[i]
    scores.sort(reverse=True)
    top = scores[0][0] or 1
    out = []
    hooks = ["WAIT FOR IT", "THIS MOMENT WAS INSANE", "NOBODY EXPECTED THIS", "WATCH TILL THE END", "THE BEST PART"]
    for sc, s in scores:
        e = min(duration, s + L)
        if all(e <= f["start"] or s >= f["end"] for f in taken + out):
            k = len(out)
            out.append({"start": round(s, 2), "end": round(e, 2), "score": max(40, int(90 * sc / top) - k * 3),
                        "hook": hooks[k % len(hooks)], "title": hooks[k % len(hooks)].title() + " 🔥 #shorts",
                        "description": "The most intense moment from this video. Watch till the end and follow for more!",
                        "hashtags": ["shorts", "viral", "trending", "reels", "fyp"],
                        "why": "Picked as one of the loudest, most energetic moments.", "auto": True})
        if len(out) >= n:
            break
    return out


def _ask_picks(text, n, min_len, max_len, video_title, depth=0):
    """Ask the AI for moments; if the request is too big for the free tier, split it in half."""
    prompt = PICK_PROMPT.format(n=n, min_len=min_len, max_len=max_len,
                                video_title=video_title or "unknown", transcript=text)
    try:
        content = _chat_json(prompt)
        print(f"[kk] AI reply: {content[:1500]}", flush=True)
        data = json.loads(content)
        if isinstance(data, dict):
            data = data.get("clips") or next((v for v in data.values() if isinstance(v, list)), [])
        return [c for c in data if isinstance(c, dict)]
    except PipelineError as e:
        if ("413" in str(e) or "too large" in str(e)) and depth < 3:
            lines = text.splitlines()
            half = len(lines) // 2
            if half:
                print("[kk] AI request too large, splitting", flush=True)
                time.sleep(20)  # let the per-minute token limit reset
                a = _ask_picks("\n".join(lines[:half]), max(1, n // 2 + 1), min_len, max_len, video_title, depth + 1)
                time.sleep(20)
                return a + _ask_picks("\n".join(lines[half:]), max(1, n // 2 + 1), min_len, max_len, video_title, depth + 1)
        print(f"[kk] AI error: {e}", flush=True)
    except (KeyError, ValueError, IndexError, TypeError) as e:
        print(f"[kk] AI reply unreadable: {e}", flush=True)
    return []


def pick_moments(tr, duration, n, min_len, max_len, video_title, video=None, progress=lambda m: None):
    if MOCK:
        return _mock_picks(duration, n, min_len, max_len)
    if not LLM_KEY:
        raise PipelineError("No AI key set. Add GROQ_API_KEY on the server.")
    # short videos: shrink clip lengths so clips still fit
    max_len = min(max_len, max(10, int(duration)))
    min_len = min(min_len, max(5, int(duration * 0.4)))
    segs = tr["segments"]
    found = []
    if segs:
        window = 420  # 7-min windows keep each AI request small for free-tier limits
        n_win = max(1, math.ceil(duration / window))
        per_win = max(2, math.ceil(n / n_win) + 1)
        for k in range(n_win):
            t0, t1 = k * window, min(duration, (k + 1) * window + max_len)
            text = _lines(segs, t0, t1)
            if len(text) < 20:
                continue
            progress(f"AI is finding viral moments {k + 1}/{n_win}")
            found += _ask_picks(text, per_win if n_win > 1 else n, min_len, max_len, video_title)
    picks = _clean_picks(found, tr["words"], duration, n, min_len, max_len)
    print(f"[kk] duration={duration:.0f}s AI gave {len(found)}, kept {len(picks)}", flush=True)
    if not picks and video:  # little or no speech: use the most energetic moments
        progress("Finding the most exciting moments")
        picks = _energy_picks(video, duration, n, min_len, max_len, [])
    if not picks:
        raise PipelineError("Could not make clips from this video. Try a different video.")
    return picks


def _clean_picks(clips, words, duration, n, min_len, max_len):
    good = []
    for c in clips:
        try:
            s, e = _parse_t(c.get("start", c.get("start_time"))), _parse_t(c.get("end", c.get("end_time")))
        except (TypeError, ValueError, AttributeError):
            continue
        if e <= s:
            continue
        s, e = max(0.0, s), min(duration, e)
        # snap to word boundaries so clips never cut a word in half
        ws = [w for w in words if w["start"] >= s - 0.4 and w["end"] <= e + 0.6]
        if ws:
            s, e = ws[0]["start"], ws[-1]["end"]
        s = max(0.0, s - 0.15)
        e = min(duration, e + 0.35)
        if e - s > max_len + 5:
            e = s + max_len
        if e - s < min(5, min_len):
            continue
        c.update(start=round(s, 2), end=round(e, 2))
        try:
            c["score"] = int(float(c.get("score") or 50))
        except (TypeError, ValueError):
            c["score"] = 50
        for k in ("hook", "title", "description", "why"):
            c[k] = str(c.get(k) or "")
        if not isinstance(c.get("hashtags"), list):
            c["hashtags"] = []
        good.append(c)
    good.sort(key=lambda c: -c["score"])
    final = []
    for c in good:  # drop overlaps, keep higher score
        if all(c["end"] <= f["start"] or c["start"] >= f["end"] for f in final):
            final.append(c)
        if len(final) >= n:
            break
    return final


def _mock_picks(duration, n, min_len, max_len):
    L = min(max_len, max(min_len, 15))
    out = []
    for i in range(n):
        s = i * (L + 2)
        if s + L > duration:
            break
        out.append({"start": s, "end": s + L, "score": 95 - i * 7, "hook": "NOBODY EXPECTED THIS",
                    "title": "Nobody Expected This Moment 😱 #shorts", "description": "Test clip.",
                    "hashtags": ["viral", "shorts", "trending"], "why": "Mock clip for testing."})
    return out


DESCRIBE_PROMPT = """You are a viral YouTube Shorts / Instagram Reels growth expert.
Write the post text for ONE short vertical clip ({length:.0f} seconds) cut from a longer video.
{what}
Original video name: {video_title}
Speech in this clip: {speech}

Reply with JSON only:
{{"hook":"<2-5 word on-screen hook in English, ALL CAPS, no emojis, curiosity-driven>",
"thumbnail_text":"<2-4 word thumbnail text in English, ALL CAPS, no emojis>",
"title":"<viral YouTube Shorts title in English, 60-95 characters INCLUDING 2-3 lowercase hashtags like #shorts, 1-2 emojis>",
"description":"<long keyword-rich English YouTube description, 120-200 words: a hook first line, what happens in the clip, why it is worth watching, searchable keywords people type, a call to like, subscribe and comment>",
"hashtags":["<12-15 relevant lowercase hashtags without #>"]}}
Describe only what is really visible/said. Never invent names of people or places you cannot see or read."""


def _frames_b64(video, start, end, job_dir, k=3):
    import base64
    out = []
    for i in range(k):
        t = start + (end - start) * (i + 1) / (k + 1)
        fp = os.path.join(job_dir, f"f{i}.jpg")
        try:
            run(["ffmpeg", "-y", "-threads", THREADS, "-ss", f"{t:.2f}", "-i", os.path.abspath(video),
                 "-frames:v", "1", "-vf", "scale='if(gt(iw,ih),640,-2)':'if(gt(iw,ih),-2,640)'", "-q:v", "5", fp])
            out.append(base64.b64encode(open(fp, "rb").read()).decode())
            os.remove(fp)
        except PipelineError:
            pass
    return out


def describe_clip(video, clip, words, job_dir, video_title):
    """AI looks at frames + speech of the clip and writes hook, title, long description, hashtags."""
    if MOCK:
        clip.setdefault("thumbnail_text", clip.get("hook", ""))
        return clip
    speech = " ".join(w["w"] for w in words if clip["start"] <= w["start"] <= clip["end"]).strip() or "(no talking, only music/sound)"
    imgs, model = [], None
    vm = _vision_model() if LLM_KEY else None
    if vm:
        imgs = _frames_b64(video, clip["start"], clip["end"], job_dir)
        model = vm if imgs else None
    what = ("The attached images are 3 frames from the clip — describe what you see in them."
            if model else "Base it on the speech and the video name.")
    prompt = DESCRIBE_PROMPT.format(length=clip["end"] - clip["start"], what=what,
                                    video_title=video_title or "unknown", speech=speech[:3000])
    try:
        data = json.loads(_chat_json(prompt, imgs if model else None, model=model))
    except Exception as e:
        print(f"[kk] describe failed ({e}), retrying text-only", flush=True)
        try:
            data = json.loads(_chat_json(DESCRIBE_PROMPT.format(length=clip["end"] - clip["start"],
                              what="Base it on the speech and the video name.", video_title=video_title or "unknown",
                              speech=speech[:3000])))
        except Exception as e2:
            print(f"[kk] describe failed: {e2}", flush=True)
            data = {}
    print(f"[kk] describe: {json.dumps(data)[:600]}", flush=True)
    if not isinstance(data, dict):
        data = {}
    for k in ("hook", "title", "description", "thumbnail_text"):
        v = str(data.get(k) or "").strip()
        if not v:
            continue
        if k == "hook" and clip.get("hook") and not clip.get("auto"):
            continue  # keep the hook the AI picked together with the moment
        clip[k] = v
    if isinstance(data.get("hashtags"), list) and data["hashtags"]:
        clip["hashtags"] = [str(h).lstrip("#").lower().replace(" ", "") for h in data["hashtags"]][:15]
    t = clip.get("title", "")
    if len(t) > 100:
        clip["title"] = t[:100].rsplit(" ", 1)[0]
    clip.setdefault("thumbnail_text", clip.get("hook", ""))
    return clip


def make_cover(video, clip, job_dir, idx, size=(1080, 1920)):
    """Full-size 9:16 thumbnail: best-looking frame, boosted colours, big text."""
    W, H = size
    text = (clip.get("thumbnail_text") or clip.get("hook") or "").upper()
    ass_name = f"cover{idx}.ass"
    fs = int(H * 0.085)
    words, lines, cur = _esc(text).split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 11 and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    colors = ["&H00FFFFFF&", "&H0000E5FF&", "&H0066FF33&"]
    body = "\\N".join("{\\c" + colors[i % 3] + "}" + l for i, l in enumerate(lines))
    ass = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: C,{CAPTION_FONT},{fs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,{fs // 7},4,2,40,40,{int(H * 0.18)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:10.00,C,,0,0,0,,{body}
"""
    with open(os.path.join(job_dir, ass_name), "w", encoding="utf-8") as f:
        f.write(ass)
    subs = f"ass={ass_name}:fontsdir={FONTS_DIR}" if os.path.isdir(FONTS_DIR) else f"ass={ass_name}"
    pre = f"scale='if(lt(iw,ih),{W},-2)':'if(lt(iw,ih),-2,{W})'"
    vf = (f"fps=2,{pre},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},thumbnail=n=20,"
          f"eq=brightness=0.05:contrast=1.15:saturation=1.25,unsharp=5:5:0.8,setpts=0,{subs}")
    out = f"clip{idx}_cover.jpg"
    run(["ffmpeg", "-y", "-threads", THREADS, "-ss", f"{clip['start']:.2f}", "-i", os.path.abspath(video),
         "-t", f"{clip['end'] - clip['start']:.2f}", "-vf", vf, "-frames:v", "1", "-q:v", "2", out], cwd=job_dir)
    run(["ffmpeg", "-y", "-i", out, "-vf", "scale=360:-2", "-q:v", "4", f"clip{idx}.jpg"], cwd=job_dir)
    os.remove(os.path.join(job_dir, ass_name))
    return out


# ---------------------------------------------------------------- rendering

def _ass_time(t):
    t = max(0.0, t)
    h, m = int(t // 3600), int(t % 3600 // 60)
    s = t - h * 3600 - m * 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _esc(t):
    return t.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def build_ass(words, clip_start, clip_end, hook, W, H, captions=True, hook_on=True):
    fs = int(H * 0.056)
    hook_fs = int(H * 0.066)
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{CAPTION_FONT},{fs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{max(3, fs // 9)},2,2,60,60,{int(H * 0.27)},1
Style: Hook,{CAPTION_FONT},{hook_fs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,{max(4, hook_fs // 7)},3,8,50,50,{int(H * 0.09)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    dur = clip_end - clip_start
    if hook_on and hook:
        # multi-colour hook: alternate white / yellow / green per word, wrapped to 2 lines
        colors = ["&H00FFFFFF&", "&H0000E5FF&", "&H0066FF33&"]
        hw = _esc(hook.upper()).split()
        lines, cur = [], ""
        for w in hw:
            if len(cur) + len(w) + 1 > 14 and cur:
                lines.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        lines.append(cur)
        k, parts = 0, []
        for li, line in enumerate(lines):
            seg = []
            for w in line.split():
                seg.append("{\\c" + colors[k % 3] + "}" + w)
                k += 1
            parts.append(" ".join(seg))
        text = "{\\fad(150,0)}" + "\\N".join(parts)
        ev.append(f"Dialogue: 1,{_ass_time(0)},{_ass_time(dur)},Hook,,0,0,0,,{text}")
    if captions:
        cw = [w for w in words if w["end"] > clip_start and w["start"] < clip_end and w["w"]]
        groups, g = [], []
        for w in cw:
            g.append(w)
            if len(g) == 3 or re.search(r"[.!?,]$", w["w"]):
                groups.append(g)
                g = []
        if g:
            groups.append(g)
        for gi, grp in enumerate(groups):
            g_end = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else grp[-1]["end"] + 0.3
            g_end = min(g_end, grp[-1]["end"] + 0.6)
            for wi, w in enumerate(grp):
                st = w["start"]
                en = grp[wi + 1]["start"] if wi + 1 < len(grp) else g_end
                toks = []
                for wj, x in enumerate(grp):
                    t = _esc(x["w"]).upper()
                    if wj == wi:
                        toks.append("{\\c&H0000E5FF&\\fscx112\\fscy112}" + t + "{\\c&H00FFFFFF&\\fscx100\\fscy100}")
                    else:
                        toks.append(t)
                ev.append(f"Dialogue: 0,{_ass_time(st - clip_start)},{_ass_time(en - clip_start)},Cap,,0,0,0,,{' '.join(toks)}")
    return head + "\n".join(ev) + "\n"


END_LEN = 2.5  # seconds of end card


def _endcard_events(kind, handle, dur, W, H):
    """LIKE & SUBSCRIBE (YouTube) or LIKE & FOLLOW (Instagram) end card + channel handle."""
    word = "FOLLOW" if kind == "ig" else "SUBSCRIBE"
    a, b = _ass_time(dur), _ass_time(dur + END_LEN + 1)
    fs = int(H * 0.075)
    ev = [f"Dialogue: 3,{a},{b},End,,0,0,0,,{{\\fad(200,0)\\pos({W // 2},{int(H * 0.40)})\\fs{fs}}}LIKE &",
          f"Dialogue: 3,{a},{b},EndBox,,0,0,0,,{{\\fad(200,0)\\pos({W // 2},{int(H * 0.50)})\\fs{fs}}} {word} ",
          f"Dialogue: 3,{a},{b},End,,0,0,0,,{{\\fad(300,0)\\pos({W // 2},{int(H * 0.60)})\\fs{int(fs * 0.62)}\\c&H0000E5FF&}}FOR MORE"]
    if handle:
        ev.append(f"Dialogue: 3,{a},{b},End,,0,0,0,,{{\\fad(400,0)\\pos({W // 2},{int(H * 0.68)})\\fs{int(fs * 0.55)}}}{_esc(handle)}")
    return ev


def render_clip(video, clip, words, job_dir, idx, layout="fit", captions=True, hook_on=True, size=(1080, 1920),
                endcard="yt", handle=""):
    W, H = size
    s, e = clip["start"], clip["end"]
    dur = e - s
    ass_name = f"clip{idx}.ass"
    ass = build_ass(words, s, e, clip.get("hook", ""), W, H, captions, hook_on)
    if endcard in ("yt", "ig"):
        fs = int(H * 0.075)
        ass = ass.replace("\n[Events]", f"\nStyle: End,{CAPTION_FONT},{fs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,{fs // 8},3,5,40,40,0,1"
                          f"\nStyle: EndBox,{CAPTION_FONT},{fs},&H00FFFFFF,&H00FFFFFF,&H000000E6,&H000000E6,-1,0,0,0,100,100,0,0,3,{fs // 5},0,5,40,40,0,1"
                          "\n\n[Events]", 1)
        ass = ass.rstrip("\n") + "\n" + "\n".join(_endcard_events(endcard, handle, dur, W, H)) + "\n"
    with open(os.path.join(job_dir, ass_name), "w", encoding="utf-8") as f:
        f.write(ass)
    subs = f"ass={ass_name}:fontsdir={FONTS_DIR}" if os.path.isdir(FONTS_DIR) else f"ass={ass_name}"
    if endcard in ("yt", "ig"):  # freeze last frame, darken it, show end card on top
        subs = (f"tpad=stop_mode=clone:stop_duration={END_LEN},"
                f"drawbox=x=0:y=0:w=iw:h=ih:color=black@0.6:t=fill:enable='gte(t,{dur:.2f})',{subs}")
    pre = f"scale='if(lt(iw,ih),{W},-2)':'if(lt(iw,ih),-2,{W})',"  # shrink 4K early: short side = {W}
    if layout == "crop":
        vf = f"[0:v]{pre}scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,{subs}[v]"
    else:  # fit: full frame in the middle over a blurred zoomed copy
        vf = (f"[0:v]{pre}split=2[a][b];"
              f"[a]scale={W // 4}:{H // 4}:force_original_aspect_ratio=increase,crop={W // 4}:{H // 4},"
              f"boxblur=10:2,scale={W}:{H},eq=brightness=-0.08[bg];"
              f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
              f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,{subs}[v]")
    out = f"clip{idx}.mp4"
    audio_f, amap = "", []
    if probe_has_audio(video):
        if endcard in ("yt", "ig"):
            audio_f = f";[0:a]afade=t=out:st={max(0, dur - 0.3):.2f}:d={END_LEN:.1f},apad=pad_dur={END_LEN}[a]"
            amap = ["-map", "[a]"]
        else:
            amap = ["-map", "0:a"]
    # low-memory settings so 4K phone videos fit on a small (512 MB) server
    cmd = ["ffmpeg", "-y", "-threads", THREADS, "-ss", f"{s:.2f}", "-t", f"{e - s:.2f}", "-i", os.path.abspath(video),
           "-filter_complex_threads", "1", "-filter_complex", vf + audio_f, "-map", "[v]", *amap,
           "-c:v", "libx264", "-preset", PRESET, "-crf", "21", "-pix_fmt", "yuv420p", "-r", "30",
           "-threads", THREADS, "-x264-params", "rc-lookahead=8:sync-lookahead=0",
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out]
    run(cmd, cwd=job_dir)
    thumb = f"clip{idx}.jpg"
    run(["ffmpeg", "-y", "-ss", f"{min(1.5, (e - s) / 2):.2f}", "-i", out, "-frames:v", "1",
         "-vf", "scale=360:-2", "-q:v", "4", thumb], cwd=job_dir)
    os.remove(os.path.join(job_dir, ass_name))
    return out, thumb


# ---------------------------------------------------------------- full job

def process(job, job_dir, update):
    """job: dict with source (path or url), options. update(**fields) saves progress."""
    o = job["options"]
    title = job.get("name") or "Video"
    if job.get("url"):
        update(stage="Downloading video from link", pct=3)
        video, title = download_link(job["url"], job_dir)
        update(name=title)
    else:
        video = job["source"]
    duration = probe_duration(video)
    update(stage="Preparing audio", pct=8, duration=round(duration))
    tr = transcribe(video, job_dir, o.get("language") or None, progress=lambda m: update(stage=m, pct=15))
    with open(os.path.join(job_dir, "transcript.json"), "w") as f:
        json.dump(tr, f)
    update(stage="AI is finding viral moments", pct=35)
    min_len, max_len = int(o.get("min_len", 20)), int(o.get("max_len", 60))
    picks = pick_moments(tr, duration, int(o.get("count", 5)), min_len, max_len, o.get("topic") or title, video=video,
                         progress=lambda m: update(stage=m, pct=40))
    clips = []
    for i, c in enumerate(picks, 1):
        base = 45 + int(50 * (i - 1) / len(picks))
        update(stage=f"AI is writing title & description {i} of {len(picks)}", pct=base)
        describe_clip(video, c, tr["words"], job_dir, o.get("topic") or title)
        update(stage=f"Editing clip {i} of {len(picks)}", pct=base + 3)
        mp4, jpg = render_clip(video, c, tr["words"], job_dir, i, o.get("layout", "fit"),
                               o.get("captions", True), o.get("hook", True),
                               endcard=o.get("endcard", "yt"), handle=o.get("handle", ""))
        update(stage=f"Making thumbnail {i} of {len(picks)}", pct=base + 8)
        try:
            cover = make_cover(video, c, job_dir, i)
        except PipelineError as e:
            print(f"[kk] cover failed: {e}", flush=True)
            cover = None
        c.update(file=mp4, thumb=jpg, cover=cover, length=round(c["end"] - c["start"], 1))
        clips.append(c)
        update(clips=clips)
    shutil.rmtree(os.path.join(job_dir, "audio"), ignore_errors=True)
    update(stage="Done", pct=100, status="done", clips=clips)
