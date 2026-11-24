"""KK Clips web server (standard library only)."""
import json, os, queue, re, shutil, threading, time, traceback, uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

import pipeline

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))
PASSWORD = os.environ.get("APP_PASSWORD", "")
MAX_UPLOAD = int(os.environ.get("MAX_UPLOAD_MB", "2048")) * 1024 * 1024
KEEP_JOBS = int(os.environ.get("KEEP_JOBS", "15"))
os.makedirs(DATA, exist_ok=True)

jobs = {}
lock = threading.Lock()
work = queue.Queue()
ID_RE = re.compile(r"^[a-f0-9]{12}$")


def save(job):
    with open(os.path.join(DATA, job["id"], "job.json"), "w") as f:
        json.dump(job, f)


def load_jobs():
    for d in os.listdir(DATA):
        p = os.path.join(DATA, d, "job.json")
        if os.path.exists(p):
            try:
                j = json.load(open(p))
                if j.get("status") in ("queued", "working"):
                    j.update(status="error", error="Server restarted while working. Please try again.")
                jobs[j["id"]] = j
            except Exception:
                pass


def cleanup():
    done = sorted(jobs.values(), key=lambda j: j["created"], reverse=True)
    for j in done[KEEP_JOBS:]:
        if j["status"] in ("done", "error"):
            shutil.rmtree(os.path.join(DATA, j["id"]), ignore_errors=True)
            jobs.pop(j["id"], None)


def worker():
    while True:
        jid = work.get()
        job = jobs[jid]
        jdir = os.path.join(DATA, jid)

        def update(**kw):
            with lock:
                job.update(kw)
                save(job)

        update(status="working", stage="Starting", pct=1)
        try:
            pipeline.process(job, jdir, update)
            if job.get("source") and os.path.exists(job["source"]):
                os.remove(job["source"])  # free disk; clips are kept
        except pipeline.PipelineError as e:
            update(status="error", error=str(e))
        except Exception as e:
            traceback.print_exc()
            update(status="error", error=f"Something went wrong: {e}")
        with lock:
            cleanup()


def new_job(name, options, url=None):
    jid = uuid.uuid4().hex[:12]
    os.makedirs(os.path.join(DATA, jid))
    job = {"id": jid, "name": name, "url": url, "source": None, "options": options, "status": "queued",
           "stage": "Waiting in line", "pct": 0, "clips": [], "created": time.time()}
    return job


def parse_options(q):
    g = lambda k, d: (q.get(k) or [d])[0]
    clamp = lambda v, lo, hi: max(lo, min(hi, int(v)))
    min_len = clamp(g("min_len", 20), 5, 170)
    return {"count": clamp(g("count", 5), 1, 15), "min_len": min_len,
            "max_len": clamp(g("max_len", 60), min_len + 5, 180),
            "layout": "crop" if g("layout", "fit") == "crop" else "fit",
            "captions": g("captions", "1") == "1", "hook": g("hook", "1") == "1",
            "language": re.sub(r"[^a-z]", "", g("language", ""))[:3]}


class H(BaseHTTPRequestHandler):
    server_version = "KKClips"

    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def authed(self, q):
        if not PASSWORD:
            return True
        return self.headers.get("X-Pass") == PASSWORD or (q.get("p") or [""])[0] == PASSWORD

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self.send_file(os.path.join(ROOT, "static", "index.html"), "text/html; charset=utf-8")
        if u.path == "/api/health":
            return self.send_json({"ok": True, "ai_key": bool(pipeline.LLM_KEY), "password": bool(PASSWORD)})
        if not self.authed(q):
            return self.send_json({"error": "Wrong password"}, 401)
        if u.path == "/api/jobs":
            with lock:
                lst = sorted(jobs.values(), key=lambda j: j["created"], reverse=True)
                return self.send_json([{k: j.get(k) for k in ("id", "name", "status", "stage", "pct", "created")}
                                       | {"count": len(j.get("clips") or [])} for j in lst])
        m = re.match(r"^/api/jobs/([a-f0-9]{12})$", u.path)
        if m:
            j = jobs.get(m.group(1))
            return self.send_json(j) if j else self.send_json({"error": "Not found"}, 404)
        m = re.match(r"^/files/([a-f0-9]{12})/(clip\d+(?:_cover)?\.(mp4|jpg))$", u.path)
        if m:
            ctype = "video/mp4" if m.group(3) == "mp4" else "image/jpeg"
            dl = (q.get("dl") or [""])[0]
            return self.send_file(os.path.join(DATA, m.group(1), m.group(2)), ctype, download=dl)
        self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if not self.authed(q):
            return self.send_json({"error": "Wrong password"}, 401)
        opts = parse_options(q)
        if u.path == "/api/link":
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            link = (body.get("url") or "").strip()
            if not re.match(r"^https?://", link):
                return self.send_json({"error": "Paste a full video link (https://...)"}, 400)
            job = new_job(link, opts, url=link)
        elif u.path == "/api/upload":
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return self.send_json({"error": "Empty file"}, 400)
            if n > MAX_UPLOAD:
                return self.send_json({"error": f"File too big (max {MAX_UPLOAD // 1048576} MB)"}, 413)
            name = unquote((q.get("name") or ["video.mp4"])[0])[:120]
            ext = os.path.splitext(name)[1].lower()
            ext = ext if ext in (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi") else ".mp4"
            job = new_job(os.path.splitext(name)[0], opts)
            path = os.path.join(DATA, job["id"], "source" + ext)
            left = n
            with open(path, "wb") as f:
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            if left > 0:
                shutil.rmtree(os.path.join(DATA, job["id"]), ignore_errors=True)
                return self.send_json({"error": "Upload was cut off. Try again."}, 400)
            job["source"] = path
        else:
            return self.send_json({"error": "Not found"}, 404)
        with lock:
            jobs[job["id"]] = job
            save(job)
        work.put(job["id"])
        self.send_json({"id": job["id"]})

    def do_DELETE(self):
        u = urlparse(self.path)
        if not self.authed(parse_qs(u.query)):
            return self.send_json({"error": "Wrong password"}, 401)
        m = re.match(r"^/api/jobs/([a-f0-9]{12})$", u.path)
        j = jobs.get(m.group(1)) if m else None
        if not j or j["status"] in ("queued", "working"):
            return self.send_json({"error": "Can't delete this one now"}, 400)
        with lock:
            jobs.pop(j["id"], None)
        shutil.rmtree(os.path.join(DATA, j["id"]), ignore_errors=True)
        self.send_json({"ok": True})

    def send_file(self, path, ctype, download=""):
        if not os.path.exists(path):
            return self.send_json({"error": "Not found"}, 404)
        size = os.path.getsize(path)
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if download:
            safe = re.sub(r"[^A-Za-z0-9 _.-]", "", download)[:80] or "clip.mp4"
            self.send_header("Content-Disposition", f'attachment; filename="{safe}"')
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = f.read(min(1 << 18, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass


if __name__ == "__main__":
    load_jobs()
    threading.Thread(target=worker, daemon=True).start()
    port = int(os.environ.get("PORT", "8000"))
    print(f"KK Clips running on http://0.0.0.0:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
