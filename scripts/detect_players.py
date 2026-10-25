"""Detection et suivi des deux joueurs (YOLO11n pre-entraine, classe "person") a partir de l'homographie du court.

    .venv\\Scripts\\python.exe scripts\\detect_players.py videos\\video_test.mp4   (apres pipeline : court.json)

Toutes les personnes detectees sont projetees au sol par leurs pieds (bas de la boite) : on ne garde que
celles qui se tiennent sur le court ou juste autour, puis UN joueur de chaque cote du filet. Le public,
l'arbitre de chaise, les ramasseurs pres des poteaux et les juges de ligne sont ecartes par leur position
et par la continuite du suivi (un joueur ne se teleporte pas).
Sortie (cache du pipeline) : runs\\analysis\\<nom>\\players.json
  {"every": 2, "frames": {frame: {"far": [x1, y1, x2, y2, X, Y], "near": [...]}}}  (image px, court m)
"""
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_court import CX, DW, L  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PERSON_WEIGHTS = ROOT / "runs" / "models" / "yolo11n.pt"
NET_Y = L / 2


def person_model():
    """YOLO11n COCO (telecharge une fois par Ultralytics, range dans runs/models)."""
    from ultralytics import YOLO
    if not PERSON_WEIGHTS.exists():
        PERSON_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        YOLO("yolo11n.pt")  # telechargement dans le dossier courant
        shutil.move("yolo11n.pt", PERSON_WEIGHTS)
    return YOLO(str(PERSON_WEIGHTS))


def to_court(H, x, y):
    p = cv2.perspectiveTransform(np.float32([[[x, y]]]), np.linalg.inv(np.array(H))).reshape(2)
    return float(p[0]), float(p[1])


def plausible(X, Y):
    """Zone ou peut se tenir un joueur : le court + 4 m sur les cotes, 9 m derriere les lignes de fond,
    sauf les abords des poteaux du filet (chaise d'arbitre, ramasseurs)."""
    if not (-4 < X < DW + 4 and -9 < Y < L + 9):
        return False
    return not (abs(Y - NET_Y) < 1.5 and (X < 0 or X > DW))


def track_players(video, model, homs, every=2, progress=None, imgsz=960):
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out, last = {}, {"far": None, "near": None}
    for f in range(n):
        ok = cap.grab()
        if not ok:
            break
        if f % every or f not in homs:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        if progress and f % 20 == 0:
            progress(f, n)
        res = model.predict(frame, imgsz=imgsz, conf=0.3, classes=[0], device="cpu", verbose=False)[0]
        cands = {"far": [], "near": []}
        for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(float, b)
            X, Y = to_court(homs[f], (x1 + x2) / 2, y2)  # les pieds
            if plausible(X, Y):
                cands["far" if Y < NET_Y else "near"].append((x1, y1, x2, y2, X, Y, float(c)))
        entry = {}
        for side, cs in cands.items():
            prev = last[side]
            dt = (f - prev[0]) / 30 if prev is not None else None
            if prev is not None and dt > 1.0:
                prev = None  # perdu depuis plus d'1 s : on le recherche comme au debut
            if prev is None:
                # (re)trouver un joueur : seulement sur le court ou juste a cote (pas le personnel au bord)
                cs = [p for p in cs if -1 < p[4] < DW + 1]
            else:
                # vitesse humaine : ~9 m/s au maximum (+ 1 m de marge de mesure)
                cs = [p for p in cs if math.dist(p[4:6], prev[1]) <= 1.0 + 9 * dt]
            if not cs:
                continue

            def score(p):
                s = p[6] + 0.002 * (p[3] - p[1])                       # confiance + taille
                s -= 0.5 * max(0.0, -p[4], p[4] - DW)                   # hors des lignes de cote
                s -= 0.05 * abs(p[4] - CX)
                if prev is not None:                                   # continuite
                    s -= 0.3 * math.dist(p[4:6], prev[1])
                return s
            best = max(cs, key=score)
            last[side] = (f, best[4:6])
            entry[side] = [round(v, 1) for v in best[:4]] + [round(best[4], 2), round(best[5], 2)]
        if entry:
            out[f] = entry
    cap.release()
    if progress:
        progress(n, n)
    return {"every": every, "frames": out}


def player_near(players, side, f, max_dt=6):
    """Position (boite + court) du joueur 'side' la plus proche dans le temps de la frame f."""
    fr = players.get("frames", {})
    for d in range(0, max_dt + 1):
        for g in (f - d, f + d):
            e = fr.get(g) or fr.get(str(g))
            if e and side in e:
                return e[side]
    return None


if __name__ == "__main__":
    video = Path(sys.argv[1])
    homs = {int(k): v for k, v in json.loads((ROOT / "runs" / "analysis" / video.stem / "court.json").read_text()).items()}
    res = track_players(video, person_model(), homs, progress=lambda i, n: i % 200 == 0 and print(f"  {i}/{n}", flush=True))
    fr = res["frames"]
    print(f"{len(fr)} frames ; loin : {sum('far' in e for e in fr.values())}, proche : {sum('near' in e for e in fr.values())}")
    (ROOT / "runs" / "analysis" / video.stem / "players.json").write_text(json.dumps(res, separators=(",", ":")))
