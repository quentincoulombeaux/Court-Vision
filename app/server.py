"""Interface web CourtVision (serveur local).

    .venv\\Scripts\\python.exe app\\server.py            puis ouvrir http://127.0.0.1:8000

Bibliotheque : dossier videos\\ (glisser une video dans l'interface l'y copie).
Les analyses tournent une par une en arriere-plan (scripts\\pipeline.py) ; l'interface suit leur
progression. Resultats : runs\\analysis\\<nom>\\analysis.json, dessines en direct par le navigateur.
"""
import importlib
import json
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import pipeline  # noqa: E402
import render_overlay  # noqa: E402

VIDEOS = ROOT / "videos"
VIDEOS.mkdir(exist_ok=True)
STATIC = Path(__file__).resolve().parent / "static"
EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

app = FastAPI(title="CourtVision")


@app.middleware("http")
async def no_stale_cache(request, call_next):
    """Le navigateur doit toujours revalider l'interface et les analyses : sinon, apres une mise a jour,
    il garde l'ancien app.js / style.css et les nouveautes n'apparaissent pas."""
    response = await call_next(request)
    if not request.url.path.endswith("/file"):  # les videos, elles, peuvent rester en cache
        response.headers["Cache-Control"] = "no-cache"
    return response
jobs = queue.Queue()
state = {"current": None, "queued": [], "last": None}   # suivi des taches, lu par l'interface
lock = threading.Lock()


def video_path(name):
    p = VIDEOS / name
    if p.parent != VIDEOS or not p.exists():
        raise HTTPException(404, "video introuvable")
    return p


def worker():
    """Execute les taches (analyse, export) une par une : le CPU ne supporte pas mieux."""
    while True:
        job = jobs.get()
        with lock:
            state["queued"] = [j for j in state["queued"] if j is not job]
            state["current"] = job
        job.update(status="running", started=time.time())
        try:
            # toujours la derniere version du code d'analyse, meme si le serveur tourne depuis longtemps
            # (sinon une mise a jour des scripts n'est prise en compte qu'apres un redemarrage)
            for mod in ("predict_court", "predict_video_yolo", "detect_bounces", "detect_players", "pipeline", "render_overlay"):
                importlib.reload(sys.modules[mod])
            if job["type"] == "analyze":
                def progress(step, frac, msg):
                    done = sum(w for k, _, w in pipeline.STEPS[:[s[0] for s in pipeline.STEPS].index(step)])
                    weight = dict((k, w) for k, _, w in pipeline.STEPS)[step]
                    job.update(step=step, step_progress=frac, message=msg, progress=done + weight * frac)
                pipeline.analyze(video_path(job["video"]), force=job.get("force", False), progress=progress)
            else:
                render_overlay.render(video_path(job["video"]), doubles=job["doubles"], close_m=job["close_m"],
                                      lang=job.get("lang", "en"),
                                      progress=lambda f, msg: job.update(progress=f, message=msg))
            job.update(status="done", progress=1.0, message="termine")
        except Exception as e:  # l'erreur est affichee dans l'interface
            traceback.print_exc()
            job.update(status="error", message=str(e))
        job["ended"] = time.time()
        with lock:
            state["current"], state["last"] = None, job


threading.Thread(target=worker, daemon=True).start()


def submit(job):
    job.update(status="queued", progress=0.0, message="en attente", step=None, created=time.time())
    with lock:
        busy = [j for j in state["queued"] + [state["current"]] if j and j["video"] == job["video"] and j["type"] == job["type"]]
        if busy:
            return busy[0]
        state["queued"].append(job)
    jobs.put(job)
    return job


@app.get("/api/videos")
def list_videos():
    out = []
    for p in sorted(VIDEOS.iterdir()):
        if p.suffix.lower() not in EXTS:
            continue
        a = sys.modules["pipeline"].ANALYSIS / p.stem / "analysis.json"
        ex = sorted((pipeline.ANALYSIS / p.stem).glob("export_*.mp4")) if a.exists() else []
        outdated = False
        if a.exists():
            try:
                with open(a, encoding="utf-8") as fh:
                    head = fh.read(4000)  # la version est au debut du fichier
                ver = json.loads(head.split(',"created"')[0] + "}").get("version", 0) if '"version"' in head else 0
                outdated = ver < sys.modules["pipeline"].PIPELINE_VERSION
            except (ValueError, OSError):
                outdated = True
        info = pipeline.video_info(p)
        out.append(dict(name=p.name, stem=p.stem, size_mb=round(p.stat().st_size / 1e6, 1), **info,
                        analyzed=a.exists(), outdated=outdated, exports=[e.name for e in ex],
                        analyzed_at=time.strftime("%d/%m %H:%M", time.localtime(a.stat().st_mtime)) if a.exists() else None))
    return out


@app.get("/api/videos/{name}/file")
def get_video(name: str):
    return FileResponse(video_path(name))


@app.get("/api/videos/{name}/analysis")
def get_analysis(name: str):
    a = pipeline.ANALYSIS / video_path(name).stem / "analysis.json"
    if not a.exists():
        raise HTTPException(404, "pas encore analysee")
    return FileResponse(a, media_type="application/json")


@app.post("/api/videos/{name}/analyze")
def analyze(name: str, force: bool = False):
    video_path(name)
    return public(submit(dict(type="analyze", video=name, force=force)))


@app.post("/api/videos/{name}/export")
def export(name: str, doubles: bool = False, close_cm: float = 5, lang: str = "en"):
    if not (pipeline.ANALYSIS / video_path(name).stem / "analysis.json").exists():
        raise HTTPException(400, "analyse la video d'abord")
    return public(submit(dict(type="export", video=name, doubles=doubles, close_m=close_cm / 100,
                              lang="fr" if lang == "fr" else "en")))


@app.get("/api/videos/{name}/export/{file}")
def get_export(name: str, file: str):
    p = pipeline.ANALYSIS / video_path(name).stem / file
    if not re.fullmatch(r"export_\w+\.mp4", file) or not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="video/mp4", filename=f"{Path(name).stem}_{file}")


@app.post("/api/upload")
async def upload(file: UploadFile):
    name = re.sub(r"[^\w.\-]+", "_", Path(file.filename).name)
    if Path(name).suffix.lower() not in EXTS:
        raise HTTPException(400, "format video non reconnu")
    dest = VIDEOS / name
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # le navigateur ne lit que le H.264 : on convertit si besoin (ffmpeg)
    codec = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                            "-of", "csv=p=0", str(dest)], capture_output=True, text=True).stdout.strip()
    if codec != "h264" or dest.suffix.lower() != ".mp4":
        h264 = dest.with_suffix(".h264.mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(dest), "-c:v", "libx264", "-preset", "fast",
                        "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", str(h264)], check=True)
        dest.unlink()
        dest = h264.rename(dest.with_suffix(".mp4"))
    return {"name": dest.name}


def public(job):
    return {k: v for k, v in job.items() if k in ("type", "video", "status", "progress", "step", "step_progress",
                                                  "message", "created", "started", "ended", "doubles")}


@app.get("/api/jobs")
def get_jobs():
    with lock:
        return dict(current=public(state["current"]) if state["current"] else None,
                    queued=[public(j) for j in state["queued"]],
                    last=public(state["last"]) if state["last"] else None,
                    steps=[dict(key=k, label=l, weight=w) for k, l, w in pipeline.STEPS])


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


if __name__ == "__main__":
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 pour y acceder depuis un autre appareil du reseau")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{a.port}")).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
