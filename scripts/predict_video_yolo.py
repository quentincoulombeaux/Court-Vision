"""Suit la balle de tennis sur une video avec YOLO (une detection par frame + filtre de suivi) et dessine sa trainee.

    python scripts\\predict_video_yolo.py video_test.mp4
    python scripts\\predict_video_yolo.py video_test.mp4 --weights runs\\ball_rv\\weights\\best.pt
    python scripts\\predict_video_yolo.py video_test.mp4 --max-frames 200 --conf 0.10

YOLO ne voit qu'une image a la fois : il peut sortir plusieurs candidats (balle, tete de joueur,
logo...). link_track() choisit a chaque frame le plus plausible, en predisant la position de la balle
avec sa vitesse (voir sa description).

Sortie : runs\\yolo_video\\<nom>_ball.mp4 (cercle ROUGE + trainee) et <nom>_ball.csv
(Frame, X, Y, Visibility, Confidence -- meme format que predict_video_tracknet.py).
"""
import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def detect_candidates(video, model, conf=0.10, imgsz=1280, device="cpu", max_frames=0, progress=None, top=8):
    """Etape lente (YOLO sur chaque frame) : toutes les detections plausibles, [[conf, x, y], ...] par frame.
    Les garder permet de refaire le suivi (link_track) en quelques secondes, sans relancer YOLO."""
    cap = cv2.VideoCapture(str(video))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_total, max_frames) if max_frames else n_total
    cands = []
    for fid in range(n):
        if progress and fid % 10 == 0:
            progress(fid, n)
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        cs = sorted(([round(float(b.conf), 4), round(float(b.xywh[0][0]), 1), round(float(b.xywh[0][1]), 1)]
                     for b in res.boxes), reverse=True)
        cands.append(cs[:top])
    cap.release()
    if progress:
        progress(n, n)
    return cands


def link_track(cands, width, lost=5, gate=0.03, min_conf=0.15, acquire_conf=0.3):
    """Choisit la balle parmi les candidats de chaque frame.
    On predit sa position avec sa vitesse (2 dernieres positions) et on n'accepte qu'un candidat dans une
    fenetre autour de la prediction : gate x largeur + 2 x vitesse (marge pour un rebond ou une frappe qui
    change brusquement la trajectoire), elargie a chaque frame perdue. Parmi les candidats de la fenetre,
    le meilleur compromis confiance / proximite. Apres --lost frames sans balle, on repart du candidat le
    plus confiant (au moins acquire_conf), ou qu'il soit."""
    rows, hist = [], []  # hist : positions acceptees (frame, x, y)
    for f, cs in enumerate(cands):
        cs = [c for c in cs if c[0] >= min_conf]
        pick = None
        if hist and f - hist[-1][0] <= lost:
            fl, xl, yl = hist[-1]
            dt = f - fl
            vx = vy = 0.0
            if len(hist) >= 2 and fl - hist[-2][0] <= 3:
                k = fl - hist[-2][0]
                vx, vy = (xl - hist[-2][1]) / k, (yl - hist[-2][2]) / k
            px, py = xl + vx * dt, yl + vy * dt
            radius = (gate * width + 2 * math.hypot(vx, vy)) * dt
            scored = [(c - 0.5 * math.dist((x, y), (px, py)) / radius, c, x, y)
                      for c, x, y in cs if math.dist((x, y), (px, py)) <= radius]
            if scored:
                _, c, x, y = max(scored)
                pick = (c, x, y)
        else:
            strong = [c for c in cs if c[0] >= acquire_conf]
            if strong:
                pick = tuple(strong[0])
        if pick is None:
            rows.append(dict(Frame=f, X=0, Y=0, Visibility=0, Confidence=0.0))
        else:
            c, x, y = pick
            hist.append((f, x, y))
            rows.append(dict(Frame=f, X=int(round(x)), Y=int(round(y)), Visibility=1, Confidence=round(c, 4)))
    return rows


def _poly(ts, pts, deg):
    """Ajuste x(t), y(t) par un polynome de degre deg ; renvoie (fonction, ecart max en px)."""
    ts, pts = np.asarray(ts, float), np.asarray(pts, float)
    deg = min(deg, len(ts) - 1)
    cx, cy = np.polyfit(ts, pts[:, 0], deg), np.polyfit(ts, pts[:, 1], deg)
    f = lambda t: np.c_[np.polyval(cx, t), np.polyval(cy, t)]  # noqa: E731
    return f, float(np.abs(f(ts) - pts).max())


def fill_gaps(rows, cands=None, height=720, max_gap=8, side=4):
    """Comble les trous du suivi (balle perdue quelques images : flou, rebond, filet, joueur devant).
      1. Recuperation : dans le trou, un candidat YOLO ecarte (confiance plus faible) situe la ou la balle
         devrait etre (interpolation entre les deux bords du trou) est repris.
      2. Interpolation physique : une seule parabole (la gravite) si les deux morceaux de trajectoire la
         suivent ; sinon, c'est qu'un REBOND (ou une frappe) a eu lieu dans le trou : on prolonge chaque
         morceau par sa propre parabole jusqu'a leur point de rencontre, qui reconstitue l'evenement manque.
    Les positions ajoutees ont Visibility = 2 (1 = vue par YOLO)."""
    k = height / 720
    obs = {r["Frame"]: (r["X"], r["Y"]) for r in rows if r["Visibility"]}
    added = {}

    def gaps():
        fs = sorted(set(obs) | set(added))
        return [(a, b) for a, b in zip(fs, fs[1:]) if 1 < b - a <= max_gap + 1]

    def pos(f):
        return added.get(f, obs.get(f))

    # 1. recuperation de detections ecartees
    if cands is not None:
        for a, b in gaps():
            pa, pb = np.array(pos(a), float), np.array(pos(b), float)
            for f in range(a + 1, b):
                pred = pa + (pb - pa) * (f - a) / (b - a)
                near = [(math.dist((x, y), pred), x, y) for c, x, y in (cands[f] if f < len(cands) else [])]
                near = [n for n in near if n[0] <= max(12 * k, 0.25 * math.dist(pa, pb))]
                if near:
                    _, x, y = min(near)
                    added[f] = (x, y)
    # 2. interpolation (une parabole) ou reconstitution du rebond (deux paraboles qui se rejoignent)
    for a, b in gaps():
        before = [f for f in range(a - 2 * side, a + 1) if pos(f) is not None][-side:]
        after = [f for f in range(b, b + 2 * side) if pos(f) is not None][:side]
        if len(before) < 2 or len(after) < 2 or before[-1] - before[0] > 2 * side or after[-1] - after[0] > 2 * side:
            continue
        both = before + after
        fit, err = _poly(both, [pos(f) for f in both], 2)
        missing = np.arange(a + 1, b)
        if err <= 3 * k:
            for f, p in zip(missing, fit(missing)):
                added[int(f)] = (float(p[0]), float(p[1]))
            continue
        # deux morceaux : prolonges par des droites (une parabole ajustee sur 4 points bruites s'emballe
        # des qu'on la prolonge) ; ils doivent se rejoindre a une distance raisonnable vu la taille du trou
        fb, eb = _poly(before[-3:], [pos(f) for f in before[-3:]], 1)
        fa, ea = _poly(after[:3], [pos(f) for f in after[:3]], 1)
        if max(eb, ea) > 5 * k:
            continue
        tt = np.linspace(a, b, 8 * (b - a) + 1)
        d = np.linalg.norm(fb(tt) - fa(tt), axis=1)
        i = int(np.argmin(d))
        if d[i] > max(6 * k, 0.2 * math.dist(pos(a), pos(b))):
            continue  # les deux morceaux ne se rejoignent pas : on ne devine rien
        t_meet = tt[i]
        for f in missing:
            p = (fb if f <= t_meet else fa)(np.array([f]))[0]
            added[int(f)] = (float(p[0]), float(p[1]))
    out = []
    for r in rows:
        f = r["Frame"]
        if not r["Visibility"] and f in added:
            x, y = added[f]
            r = dict(Frame=f, X=int(round(x)), Y=int(round(y)), Visibility=2, Confidence=0.0)
        out.append(r)
    return out


def track_ball(video, model, conf=0.10, imgsz=1280, device="cpu", max_frames=0, progress=None):
    """Detection + suivi. Renvoie (lignes du CSV, candidats bruts)."""
    cands = detect_candidates(video, model, conf, imgsz, device, max_frames, progress)
    width = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_WIDTH))
    return link_track(cands, width), cands


def write_track_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Frame", "X", "Y", "Visibility", "Confidence"])
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--weights", default=str(ROOT / "runs" / "ball_v1" / "weights" / "best.pt"))
    ap.add_argument("--conf", type=float, default=0.10, help="seuil bas : le suivi trie ensuite")
    ap.add_argument("--imgsz", type=int, default=1280, help="doit rester 1280, comme a l'entrainement")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = toute la video")
    ap.add_argument("--trail", type=int, default=10, help="longueur de la trainee affichee")
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"{vw}x{vh} @ {fps:.1f} fps -- {Path(args.weights).name} sur {args.device}")

    out_dir = ROOT / "runs" / "yolo_video"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem
    rows, _ = track_ball(args.video, model, args.conf, args.imgsz, args.device, args.max_frames,
                         progress=lambda i, n: i % 50 == 0 and print(f"  frame {i}/{n}", flush=True))
    write_track_csv(rows, out_dir / f"{stem}_ball.csv")

    cap = cv2.VideoCapture(args.video)
    writer = cv2.VideoWriter(str(out_dir / f"{stem}_ball.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))
    trail = []
    for row in rows:
        ok, frame = cap.read()
        if not ok:
            break
        trail.append((row["X"], row["Y"]) if row["Visibility"] else None)
        del trail[:-args.trail]
        pts = [t for t in trail if t is not None]
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, a, b, (0, 200, 255), 2)
        if trail[-1] is not None:
            cv2.circle(frame, trail[-1], 12, (0, 0, 255), 2)
        writer.write(frame)
    cap.release()
    writer.release()
    print(f"Balle vue sur {sum(r['Visibility'] for r in rows)}/{len(rows)} frames -> {out_dir}")


if __name__ == "__main__":
    main()
