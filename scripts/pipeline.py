"""Analyse complete d'une video : terrain -> balle -> rebonds -> jugement in / out.

    .venv\\Scripts\\python.exe scripts\\pipeline.py videos\\video_test.mp4
    .venv\\Scripts\\python.exe scripts\\pipeline.py videos\\video_test.mp4 --force     # tout recalculer

Chaque etape est mise en cache dans runs\\analysis\\<nom>\\ (court.json, ball.csv) : relancer ne refait
que ce qui manque. Resultat final : runs\\analysis\\<nom>\\analysis.json, lu par l'interface web (app\\).

Le jugement in / out n'est PAS fait ici une fois pour toutes : on stocke la position de chaque rebond
en metres sur le court (via l'homographie de sa frame), et l'interface decide selon le mode simple /
double et la marge "limite" choisis. judge() ci-dessous fait la meme chose cote Python (export video).
"""
import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_bounces  # noqa: E402
import predict_court  # noqa: E402
import predict_video_yolo  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "runs" / "analysis"
BALL_WEIGHTS = ROOT / "runs" / "ball_rv" / "weights" / "best.pt"
COURT_WEIGHTS = ROOT / "runs" / "court_pose" / "weights" / "best.pt"
STEPS = [("court", "Terrain", 0.10), ("ball", "Balle", 0.85), ("bounces", "Rebonds", 0.03), ("judge", "Jugement", 0.02)]

_models = {}


def model(path):
    """Les modeles sont charges une seule fois (l'interface enchaine plusieurs analyses)."""
    from ultralytics import YOLO
    if str(path) not in _models:
        _models[str(path)] = YOLO(str(path))
    return _models[str(path)]


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    info = dict(fps=cap.get(cv2.CAP_PROP_FPS) or 30, width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def court_coords(H, x, y):
    """Point image -> position en metres sur le court (inverse de l'homographie terrain -> image)."""
    p = cv2.perspectiveTransform(np.float32([[[x, y]]]), np.linalg.inv(np.array(H))).reshape(2)
    return [round(float(p[0]), 3), round(float(p[1]), 3)]


BALL_R = 0.033   # rayon d'une balle de tennis (m)
MAX_OUT = 3.0    # au-dela (m hors des lignes), ce n'est pas un rebond : balle en l'air ou erreur de suivi


def vertical_scale(H, c):
    """Pixels par metre de HAUTEUR au point c du court. Approximation locale de la camera TV : le rapport
    des echelles profondeur / largeur donne l'angle e sous lequel la camera voit le sol (sin e), et un
    objet vertical y apparait avec l'echelle de largeur x cos e. Ne demande que l'homographie."""
    H = np.array(H)
    p = cv2.perspectiveTransform(np.float32([[[c[0] - 0.5, c[1]]], [[c[0] + 0.5, c[1]]],
                                             [[c[0], c[1] - 0.5]], [[c[0], c[1] + 0.5]]]), H).reshape(4, 2)
    s_lat, s_dep = float(np.linalg.norm(p[1] - p[0])), float(np.linalg.norm(p[3] - p[2]))
    return s_lat * math.cos(math.asin(min(0.99, s_dep / max(1e-6, s_lat))))


def ground_below(H, x, y, height):
    """Point du sol a la verticale d'une balle vue en (x, y) et situee a 'height' metres de haut."""
    c = court_coords(H, x, y)
    for _ in range(4):  # l'echelle depend du point : quelques iterations suffisent
        c = court_coords(H, x, y + height * vertical_scale(H, c))
    return c


def contact_coords(H, x, y):
    """Position au sol du point de contact : le detecteur donne le centre de la balle, un rayon au-dessus du sol."""
    return ground_below(H, x, y, BALL_R)


K_DRAG = 0.019            # 1/m : en vol, v(s) = v0 . exp(-K_DRAG . s) (frottement de l'air d'une balle de tennis)
HIT_HEIGHT = 1.0          # hauteur de frappe typique d'un coup de fond de court (m)
SERVE_INSIDE = 0.4        # au service, la balle est frappee ~40 cm a l'interieur de la ligne de fond (m)
NET_Y = predict_court.L / 2


def serve_height(H, x, y, bounce_y):
    """Hauteur de frappe d'un service : celle qui place le point de frappe juste a l'interieur de la ligne
    de fond du serveur (cote oppose au rebond). Recherche par dichotomie entre 1,8 et 3,6 m."""
    far = bounce_y > NET_Y                       # rebond cote proche -> serveur au fond eloigne
    target = SERVE_INSIDE if far else predict_court.L - SERVE_INSIDE
    lo, hi = 1.8, 3.6
    for _ in range(20):  # plus la balle est haute, plus son pied se rapproche de la camera (y augmente)
        mid = (lo + hi) / 2
        if ground_below(H, x, y, mid)[1] < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def contact_time(track, s):
    """Instant du contact raquette, en frames (fractionnaire). La frappe reperee (sommet de la courbe dans
    l'image) precede souvent le contact de quelques images ; or 3 images de trop sur 20 de vol = 15 % de
    vitesse en moins. Le contact est la ou la balle accelere brusquement : on cherche la derniere position
    j avant le depart rapide (rapport vitesse apres / avant maximal). Si la balle est perdue juste apres
    (service tres rapide, flou), la vitesse apres se mesure jusqu'a la premiere position retrouvee.
    Renvoie (frame j, instant du contact) ou None."""
    # positions distinctes seulement : une video a images dupliquees (ex. 15 img/s affichees en 30) montre
    # chaque position 2 fois, le vrai pas de temps est l'ecart entre positions distinctes
    frames = []
    for f in range(s - 8, s + 12):
        if f in track and (not frames or math.dist(track[f], track[frames[-1]]) > 1.0):
            frames.append(f)
    best = None
    for i, j in enumerate(frames):
        if not s - 2 <= j <= s + 6:
            continue
        prev, nxt = frames[max(0, i - 3):i], frames[i + 1:i + 4]
        if len(prev) < 2 or not nxt:
            continue
        v_b = math.dist(track[prev[0]], track[j]) / (j - prev[0])
        v_a = math.dist(track[j], track[nxt[-1]]) / (nxt[-1] - j)
        ratio = v_a / max(1.0, v_b)
        if best is None or ratio > best[0]:
            best = (ratio, j, nxt[0])
    if best is None or best[0] < 1.5:
        return None
    _, j, j_next = best
    return j, (j + j_next) / 2  # contact entre la derniere position lente et la premiere rapide


def shot_speeds(bounces, shots, track, homs, info):
    """Vitesse de chaque frappe suivie d'un rebond, a partir des dimensions officielles du court.
    Point de frappe : la balle a l'image de la frappe, placee a 1 m de haut et ramenee au sol (service :
    on cherche la hauteur qui place la frappe juste a l'interieur de la ligne de fond, ou se tient le
    serveur) ; point d'arrivee : le rebond. Distance 3D / duree du vol = vitesse moyenne, puis
    correction du freinage de l'air pour estimer la vitesse au depart de la raquette (celle des radars).
    La frappe doit etre de l'autre cote du filet que le rebond, sinon on essaie la frappe precedente."""
    fps, H_img = info["fps"], info["height"]
    prev_b = -10**9
    for b in bounces:
        cands = [s for s in shots if prev_b < s < b["frame"] and 0.25 * fps <= b["frame"] - s <= 3 * fps]
        first_of_point = b["frame"] - prev_b > 2.5 * fps
        prev_b = b["frame"]
        if b.get("court") is None:
            continue
        for s in reversed(cands):
            ct = contact_time(track, s)
            if ct is not None:
                fb, t_hit = ct
            else:
                fb = min((f for f in range(s - 2, s + 3) if f in track), key=lambda f: abs(f - s), default=None)
                t_hit = fb
            Hm = homs.get(fb) if fb is not None else None
            if Hm is None:
                continue
            # service : premiere frappe du point, balle quasi immobile juste avant (le lancer)
            before = [track[f] for f in range(s - 8, s) if f in track]
            slow = len(before) >= 3 and np.mean([math.dist(p, q) for p, q in zip(before, before[1:])]) / H_img < 0.006
            serve = first_of_point and slow
            h = serve_height(Hm, *track[fb], b["court"][1]) if serve else HIT_HEIGHT
            g = ground_below(Hm, *track[fb], h)
            if (g[1] - NET_Y) * (b["court"][1] - NET_Y) >= 0 or not (-12 < g[1] < predict_court.L + 12):
                continue  # pas de l'autre cote du filet : ce n'etait pas la frappe de ce rebond
            dist = math.dist((g[0], g[1], h), (b["court"][0], b["court"][1], 0.0))
            t = (b.get("t", b["frame"]) - t_hit) / fps   # instants precis (entre deux images)
            v0 = (math.exp(K_DRAG * dist) - 1) / (K_DRAG * t)
            if not 30 <= v0 * 3.6 <= 260:
                continue
            b["speed"] = dict(kmh=round(v0 * 3.6), avg_kmh=round(dist / t * 3.6), hit_frame=fb,
                              hit_x=round(track[fb][0], 1), hit_y=round(track[fb][1], 1), serve=bool(serve),
                              hit_height=round(h, 2),
                              hit_court=[round(g[0], 2), round(g[1], 2)])
            break


def signed_distance(c, doubles):
    """Distance (m) du rebond a la limite du terrain : < 0 dedans, > 0 dehors."""
    L, DW, AL = predict_court.L, predict_court.DW, predict_court.AL
    x0, x1 = (0.0, DW) if doubles else (AL, DW - AL)
    dx = max(x0 - c[0], c[0] - x1)
    dy = max(-c[1], c[1] - L)
    if dx > 0 and dy > 0:
        return float(np.hypot(dx, dy))
    return float(max(dx, dy))


def judge(bounce, doubles=False, close_m=0.05):
    """'in', 'out', 'close' (trop pres de la ligne pour trancher) ou 'unknown' (terrain non visible)."""
    if bounce.get("court") is None:
        return "unknown"
    s = signed_distance(bounce["court"], doubles)
    if abs(s) <= close_m:
        return "close"
    return "in" if s < 0 else "out"


def analyze(video, force=False, progress=None):
    """progress(step_key, fraction_de_l_etape, message). Renvoie le chemin de analysis.json."""
    video = Path(video)
    out = ANALYSIS / video.stem
    if force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    info = video_info(video)
    report = progress or (lambda *a: None)
    t0 = time.time()

    # 1. Terrain : homographie terrain -> image, frame par frame (le modele tourne toutes les 15 frames)
    court_path = out / "court.json"
    if not court_path.exists():
        a = SimpleNamespace(imgsz=960, conf=0.25, kp_conf=0.5, every=15, smooth=0.5, device="cpu", max_frames=0)
        homs, _ = predict_court.court_track(video, model(COURT_WEIGHTS), a,
                                            progress=lambda i, n: report("court", i / max(1, n), f"frame {i}/{n}"))
        court_path.write_text(json.dumps(homs))
    homs = {int(k): v for k, v in json.loads(court_path.read_text()).items()}
    report("court", 1.0, f"terrain visible sur {len(homs)}/{info['frames']} frames")

    # 2. Balle (l'etape longue : ~1 s par frame sur CPU)
    # candidates.json : toutes les detections YOLO (lent) ; ball.csv : la balle choisie parmi elles (rapide)
    ball_path, cand_path = out / "ball.csv", out / "candidates.json"
    if not cand_path.exists() and not ball_path.exists():
        cands = predict_video_yolo.detect_candidates(video, model(BALL_WEIGHTS),
                                                     progress=lambda i, n: report("ball", i / max(1, n), f"frame {i}/{n}"))
        cand_path.write_text(json.dumps(cands, separators=(",", ":")))
    if not ball_path.exists():  # (une ancienne analyse peut n'avoir que ball.csv : on le garde)
        rows = predict_video_yolo.link_track(json.loads(cand_path.read_text()), info["width"])
        predict_video_yolo.write_track_csv(rows, ball_path)
    track, _ = detect_bounces.load_track(ball_path)
    report("ball", 1.0, f"balle vue sur {len(track)} frames")

    # 3. Rebonds et frappes (reglages par defaut de detect_bounces.py)
    report("bounces", 0.0, "analyse de la trajectoire")
    args = detect_bounces.make_parser().parse_args([])
    samples = detect_bounces.to_samples(track, args.dedup, args.outlier, info["height"])
    bounces, shots = detect_bounces.find_bounces(samples, args, info["height"], info["fps"])
    report("bounces", 1.0, f"{len(bounces)} rebonds, {len(shots)} frappes")

    # 4. Position de chaque rebond sur le court (en metres) pour le jugement in / out
    report("judge", 0.0, "projection sur le court")
    out_bounces, dropped = [], 0
    for b in bounces:
        f = b["frame"]
        near = [g for g in range(f - 15, f + 16) if g in homs]  # terrain visible a ce moment-la ?
        H = homs[min(near, key=lambda g: abs(g - f))] if near else None
        c = contact_coords(H, b["x"], b["y"]) if H else None
        if c is not None and signed_distance(c, True) > MAX_OUT:
            dropped += 1  # ex. lancer de balle au service : projetee au sol, la balle en l'air tombe tres loin
            continue
        out_bounces.append(dict(frame=f, t=round(b.get("t", f), 2), x=round(b["x"], 1), y=round(b["y"], 1), court=c,
                                kick=round(b["kick"], 2), angle=round(b["angle"], 1)))
    shot_speeds(out_bounces, shots, track, homs, info)
    n_judged = sum(b["court"] is not None for b in out_bounces)
    report("judge", 1.0, f"{n_judged}/{len(out_bounces)} rebonds jugeables ({dropped} ecartes : trop loin du terrain)")

    result = dict(
        video=video.name, **info,
        created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 1),
        ball=[[f, round(x, 1), round(y, 1)] for f, (x, y) in sorted(track.items())],
        shots=shots, bounces=out_bounces,
        homographies={str(k): v for k, v in homs.items()},
    )
    path = out / "analysis.json"
    path.write_text(json.dumps(result, separators=(",", ":")))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--force", action="store_true", help="ignore le cache et recalcule tout")
    a = ap.parse_args()
    last = {}

    def show(step, frac, msg):
        if last.get(step) != round(frac, 1) or frac >= 1:
            last[step] = round(frac, 1)
            print(f"[{step:7s}] {frac:4.0%}  {msg}", flush=True)

    path = analyze(a.video, a.force, show)
    res = json.loads(path.read_text())
    for b in res["bounces"]:
        sp = b.get("speed")
        speed = f"  frappe {sp['hit_frame']:4d} -> {sp['kmh']:3d} km/h{' (service)' if sp['serve'] else ''}" if sp else ""
        print(f"  rebond frame {b['frame']:4d}  court {b['court']}  -> {judge(b)}{speed}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
