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

def download_link(url, job_dir):
    """Download a YouTube/other link with yt-dlp (max 1080p)."""
    if not shutil.which("yt-dlp"):
        raise PipelineError("yt-dlp is not installed on the server.")
    out_tmpl = os.path.join(job_dir, "source.%(ext)s")
    cmd = ["yt-dlp", "--no-playlist", "-f",
           "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/bv*[height<=1080]+ba/b",
           "--merge-output-format", "mp4", "-o", out_tmpl, "--print", "title", "--no-simulate"]
    if YTDLP_COOKIES and os.path.exists(YTDLP_COOKIES):
        cmd += ["--cookies", YTDLP_COOKIES]
    p = subprocess.run(cmd + [url], capture_output=True, text=True)
    if p.returncode != 0:
        err = p.stderr[-500:]
        if "Sign in to confirm" in err or "bot" in err:
            raise PipelineError("YouTube blocked the server download. Download the video on your phone and use Upload instead.")
        raise PipelineError("Link download failed: " + err)
    title = (p.stdout.strip().splitlines() or ["Video"])[0]
    for f in os.listdir(job_dir):
        if f.startswith("source."):
            return os.path.join(job_dir, f), title
    raise PipelineError("Download finished but no file was found.")


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
        raise PipelineError("This video has no sound, so there is nothing to transcribe.")
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
    if not words:
        raise PipelineError("No speech was found in this video.")
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


def pick_moments(tr, duration, n, min_len, max_len, video_title, progress=lambda m: None):
    if MOCK:
        return _mock_picks(duration, n, min_len, max_len)
    if not LLM_KEY:
        raise PipelineError("No AI key set. Add GROQ_API_KEY on the server.")
    segs = tr["segments"]
    window = 900  # 15-min windows keep each AI request small for free-tier limits
    n_win = max(1, math.ceil(duration / window))
    per_win = max(2, math.ceil(n / n_win) + 1)
    found = []
    for k in range(n_win):
        t0, t1 = k * window, min(duration, (k + 1) * window + max_len)
        text = _lines(segs, t0, t1)
        if len(text) < 80:
            continue
        progress(f"AI is finding viral moments {k + 1}/{n_win}")
        prompt = PICK_PROMPT.format(n=per_win if n_win > 1 else n, min_len=min_len, max_len=max_len,
                                    video_title=video_title or "unknown", transcript=text)
        body = json.dumps({"model": LLM_MODEL, "temperature": 0.4,
                           "response_format": {"type": "json_object"},
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        res = _post(LLM_URL, LLM_KEY, body, {"Content-Type": "application/json"}, timeout=180)
        try:
            data = json.loads(res["choices"][0]["message"]["content"])
            found += data.get("clips", [])
        except (KeyError, ValueError, IndexError):
            continue
    return _clean_picks(found, tr["words"], duration, n, min_len, max_len)


def _clean_picks(clips, words, duration, n, min_len, max_len):
    good = []
    for c in clips:
        try:
            s, e = float(c["start"]), float(c["end"])
        except (KeyError, TypeError, ValueError):
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
        if e - s < max(5, min_len * 0.6):
            continue
        c.update(start=round(s, 2), end=round(e, 2))
        c["score"] = int(c.get("score") or 50)
        good.append(c)
    good.sort(key=lambda c: -c["score"])
    final = []
    for c in good:  # drop overlaps, keep higher score
        if all(c["end"] <= f["start"] or c["start"] >= f["end"] for f in final):
            final.append(c)
        if len(final) >= n:
            break
    if not final:
        raise PipelineError("The AI could not find good moments. Try a longer video or shorter clip length.")
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


def render_clip(video, clip, words, job_dir, idx, layout="fit", captions=True, hook_on=True, size=(1080, 1920)):
    W, H = size
    s, e = clip["start"], clip["end"]
    ass_name = f"clip{idx}.ass"
    with open(os.path.join(job_dir, ass_name), "w", encoding="utf-8") as f:
        f.write(build_ass(words, s, e, clip.get("hook", ""), W, H, captions, hook_on))
    subs = f"ass={ass_name}:fontsdir={FONTS_DIR}" if os.path.isdir(FONTS_DIR) else f"ass={ass_name}"
    if layout == "crop":
        vf = f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,{subs}[v]"
    else:  # fit: full frame in the middle over a blurred zoomed copy
        vf = (f"[0:v]split=2[a][b];"
              f"[a]scale={W // 4}:{H // 4}:force_original_aspect_ratio=increase,crop={W // 4}:{H // 4},"
              f"boxblur=10:2,scale={W}:{H},eq=brightness=-0.08[bg];"
              f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
              f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,{subs}[v]")
    out = f"clip{idx}.mp4"
    cmd = ["ffmpeg", "-y", "-ss", f"{s:.2f}", "-i", os.path.abspath(video), "-t", f"{e - s:.2f}",
           "-filter_complex", vf, "-map", "[v]", "-map", "0:a?",
           "-c:v", "libx264", "-preset", PRESET, "-crf", "21", "-pix_fmt", "yuv420p", "-r", "30",
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
    picks = pick_moments(tr, duration, int(o.get("count", 5)), min_len, max_len, title,
                         progress=lambda m: update(stage=m, pct=40))
    clips = []
    for i, c in enumerate(picks, 1):
        update(stage=f"Rendering clip {i} of {len(picks)}", pct=45 + int(50 * (i - 1) / len(picks)))
        mp4, jpg = render_clip(video, c, tr["words"], job_dir, i, o.get("layout", "fit"),
                               o.get("captions", True), o.get("hook", True))
        c.update(file=mp4, thumb=jpg, length=round(c["end"] - c["start"], 1))
        clips.append(c)
        update(clips=clips)
    shutil.rmtree(os.path.join(job_dir, "audio"), ignore_errors=True)
    update(stage="Done", pct=100, status="done", clips=clips)
