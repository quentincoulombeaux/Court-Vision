"""Exporte la video annotee (memes calques que l'interface web) a partir de runs\\analysis\\<nom>\\analysis.json.

    .venv\\Scripts\\python.exe scripts\\render_overlay.py videos\\video_test.mp4
    .venv\\Scripts\\python.exe scripts\\render_overlay.py videos\\video_test.mp4 --doubles --close-cm 8

Calques : lignes du court (recalees), balle + trainee, rebonds. A chaque rebond : une onde qui
s'etale a plat sur le sol (dessinee dans le plan du court grace a l'homographie), un flash et une
etiquette IN / OUT / LIMITE ; puis une marque reste au sol, verte (bonne), rouge (faute) ou orange
(trop pres de la ligne pour trancher).
Sortie : runs\\analysis\\<nom>\\export_simple.mp4 (ou export_double.mp4), en H.264 si ffmpeg est la.
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline  # noqa: E402
from predict_court import COURT_LINES, NET  # noqa: E402

# couleurs BGR (memes valeurs que app/static/style.css)
BALL = (75, 242, 217)          # jaune balle  #D9F24B
COLORS = {"in": (153, 211, 52), "out": (94, 63, 244), "close": (11, 158, 245), "unknown": (184, 163, 148)}
LABELS = {"en": {"in": "IN", "out": "OUT", "close": "CLOSE", "unknown": "?", "serve": "SERVE"},
          "fr": {"in": "IN", "out": "OUT", "close": "LIMITE", "unknown": "?", "serve": "SERVICE"}}
TRAIL, ANIM = 12, 1.4          # frames de trainee ; duree de l'animation d'impact (s)


def project(H, pts):
    return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), H).reshape(-1, 2)


def ground_circle(H, c, r, n=48):
    """Cercle de rayon r (m) pose sur le court, vu en perspective."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return project(H, np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)]).astype(np.int32)


def blend(frame, layer, mask_alpha):
    """Superpose layer sur frame avec une opacite globale."""
    cv2.addWeighted(layer, mask_alpha, frame, 1 - mask_alpha, 0, dst=frame)


def ease_out(t):
    return 1 - (1 - t) ** 3


def draw_label(frame, text, center, color, alpha):
    font, scale, th = cv2.FONT_HERSHEY_DUPLEX, 0.7, 1
    (tw, tht), _ = cv2.getTextSize(text, font, scale, th)
    x, y = int(center[0] - tw / 2), int(center[1])
    layer = frame.copy()
    cv2.rectangle(layer, (x - 10, y - tht - 8), (x + tw + 10, y + 8), (22, 17, 13), -1, cv2.LINE_AA)
    cv2.rectangle(layer, (x - 10, y - tht - 8), (x + tw + 10, y + 8), color, 1, cv2.LINE_AA)
    cv2.putText(layer, text, (x, y), font, scale, color, th, cv2.LINE_AA)
    blend(frame, layer, alpha)


def draw_speed(frame, b, fid, fps, labels):
    """Vitesse de la frappe, affichee au depart de la trajectoire jusqu'au rebond."""
    sp = b.get("speed")
    if not sp or fid < sp["hit_frame"] or fid > b["frame"] + 0.4 * fps:
        return
    t_in, t_out = (fid - sp["hit_frame"]) / fps, (fid - b["frame"]) / fps
    alpha = min(1.0, t_in / 0.15) * (max(0.0, 1 - t_out / 0.4) if t_out > 0 else 1.0)
    k = frame.shape[0] / 720
    text = (labels["serve"] + "  " if sp["serve"] else "") + f"{sp['kmh']} km/h"
    font, scale, th = cv2.FONT_HERSHEY_DUPLEX, 0.62 * k, max(1, int(round(k)))
    (tw, tht), _ = cv2.getTextSize(text, font, scale, th)
    x = int(sp["hit_x"] - tw / 2)
    y = int(sp["hit_y"] - (34 + 8 * ease_out(min(1.0, t_in / 0.4))) * k)
    pad = int(10 * k)
    y = max(y, tht + pad)                                   # reste dans l'image (service : balle tout en haut)
    x = min(max(x, pad), frame.shape[1] - tw - pad)
    layer = frame.copy()
    cv2.rectangle(layer, (x - pad, y - tht - pad // 2 - 2), (x + tw + pad, y + pad // 2 + 2), (22, 17, 13), -1, cv2.LINE_AA)
    cv2.rectangle(layer, (x - pad, y - tht - pad // 2 - 2), (x + tw + pad, y + pad // 2 + 2), BALL, 1, cv2.LINE_AA)
    cv2.putText(layer, text, (x, y), font, scale, BALL, th, cv2.LINE_AA)
    blend(frame, layer, alpha)


def draw_bounce(frame, b, verdict, t, H, labels):
    """t : secondes depuis le rebond (>= 0)."""
    color = COLORS[verdict]
    c = b.get("court")
    if c is not None and H is not None:
        center = project(H, [c])[0]
        poly = lambda r: ground_circle(H, c, r)  # noqa: E731
    else:  # terrain inconnu : on reste dans le plan de l'image
        center = np.float32([b["x"], b["y"]])
        k = frame.shape[0] / 720
        poly = lambda r: cv2.ellipse2Poly(tuple(map(int, center)), (int(r * 40 * k), int(r * 16 * k)), 0, 0, 360, 8)  # noqa: E731

    if t < ANIM:  # animation d'impact
        layer = frame.copy()
        for delay, rmax in ((0.0, 1.1), (0.15, 0.7)):  # deux ondes qui s'etalent sur le sol
            u = (t - delay) / 0.9
            if 0 <= u <= 1:
                cv2.polylines(layer, [poly(0.12 + rmax * ease_out(u))], True, color, 2, cv2.LINE_AA)
                blend(frame, layer, 1 - u)
                layer = frame.copy()
        if t < 0.25:  # flash
            cv2.fillPoly(layer, [poly(0.12 + 0.35 * t / 0.25)], (255, 255, 255), cv2.LINE_AA)
            blend(frame, layer, 0.6 * (1 - t / 0.25))
        a = min(1.0, t / 0.15) * (1 - max(0.0, (t - 1.0) / (ANIM - 1.0)))
        draw_label(frame, labels[verdict], (center[0], center[1] - 28 - 14 * ease_out(min(1, t / 0.6))), color, a)

    # marque au sol, persistante
    layer = frame.copy()
    p = poly(0.11)
    cv2.fillPoly(layer, [p], color, cv2.LINE_AA)
    cv2.polylines(layer, [p], True, (255, 255, 255), 1, cv2.LINE_AA)
    blend(frame, layer, 0.9)


def render(video, doubles=False, close_m=0.05, progress=None, lang="en"):
    video = Path(video)
    out_dir = pipeline.ANALYSIS / video.stem
    res = json.loads((out_dir / "analysis.json").read_text())
    homs = {int(k): np.array(v, np.float32) for k, v in res["homographies"].items()}
    ball = {f: (int(x), int(y)) for f, x, y in res["ball"]}
    bounces = [(b, pipeline.judge(b, doubles, close_m)) for b in res["bounces"]]
    labels = LABELS.get(lang, LABELS["en"])
    fps, n = res["fps"], res["frames"]
    mode = "double" if doubles else "simple"
    tmp = out_dir / f"_export_{mode}.mp4"
    final = out_dir / f"export_{mode}.mp4"

    cap = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (res["width"], res["height"]))
    for fid in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        H = homs.get(fid)
        if H is not None:  # lignes du court
            layer = frame.copy()
            for a, b in COURT_LINES:
                p, q = project(H, [a, b]).astype(int)
                cv2.line(layer, tuple(p), tuple(q), (255, 255, 255), 2, cv2.LINE_AA)
            p, q = project(H, NET).astype(int)
            cv2.line(layer, tuple(p), tuple(q), (255, 255, 255), 1, cv2.LINE_AA)
            blend(frame, layer, 0.45)
        # trainee (s'estompe) puis balle
        pts = [ball.get(f) for f in range(fid - TRAIL, fid + 1)]
        for i in range(1, len(pts)):
            if pts[i - 1] and pts[i]:
                layer = frame.copy()
                cv2.line(layer, pts[i - 1], pts[i], BALL, 1 + i // 4, cv2.LINE_AA)
                blend(frame, layer, i / len(pts))
        if ball.get(fid):
            cv2.circle(frame, ball[fid], 10, BALL, 2, cv2.LINE_AA)
        # rebonds deja passes (le plus recent dessine en dernier, au-dessus)
        for b, verdict in bounces:
            if b["frame"] <= fid:
                draw_bounce(frame, b, verdict, (fid - b["frame"]) / fps, H if H is not None else homs.get(b["frame"]), labels)
        for b, _ in bounces:
            draw_speed(frame, b, fid, fps, labels)
        writer.write(frame)
        if progress and fid % 20 == 0:
            progress(fid / n, f"frame {fid}/{n}")
    cap.release()
    writer.release()

    if shutil.which("ffmpeg"):  # H.264 lisible partout, + le son de la video d'origine
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-i", str(video), "-map", "0:v:0", "-map", "1:a:0?",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "160k", "-shortest", str(final)], check=True)
        tmp.unlink()
    else:
        tmp.replace(final)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--doubles", action="store_true", help="couloirs bons (double)")
    ap.add_argument("--close-cm", type=float, default=5, help="marge 'limite' autour des lignes (cm)")
    ap.add_argument("--lang", default="en", choices=["en", "fr"], help="langue des etiquettes")
    a = ap.parse_args()
    path = render(a.video, a.doubles, a.close_cm / 100, lang=a.lang,
                  progress=lambda f, m: print(f"  {f:4.0%} {m}", flush=True) if int(f * 100) % 20 == 0 else None)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
