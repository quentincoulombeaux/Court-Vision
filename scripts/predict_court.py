"""Detecte le terrain (14 points, YOLO11-pose) et dessine toutes ses lignes, recalees par homographie.

    .venv\\Scripts\\python.exe scripts\\predict_court.py video_test.mp4
    .venv\\Scripts\\python.exe scripts\\predict_court.py video_test.mp4 --every 30 --max-frames 300
    .venv\\Scripts\\python.exe scripts\\predict_court.py une_image.jpg

Homographie : on connait les dimensions reglementaires du court (COURT_POINTS, en metres). A partir
des points detectes (au moins 4 surs), on calcule la transformation terrain -> image (RANSAC, qui
ecarte les points aberrants), puis on projette TOUTES les lignes, meme celles cachees par un joueur.
La camera TV bouge peu : on ne relance le modele que toutes les --every frames (et on lisse).

Sortie : runs\\court\\<nom>_court.mp4 (ou .jpg) et <nom>_court.json (homographie terrain -> image par
frame, pour convertir plus tard un rebond en position reelle sur le court : in / out, mini-carte).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Court reglementaire, en metres. x : de gauche a droite (0 = couloir de double gauche),
# y : du fond eloigne (0) au fond proche (23.77). Meme ordre que les 14 points du dataset.
L, DW, SW = 23.77, 10.97, 8.23            # longueur, largeur double, largeur simple
AL = (DW - SW) / 2                        # couloir : 1.37 m
SV = L / 2 - 6.40                         # ligne de service : 6.40 m du filet
CX = DW / 2
COURT_POINTS = np.float32([
    [0, 0], [DW, 0], [0, L], [DW, L],                     # 0-3  coins (double)
    [AL, 0], [AL, L], [DW - AL, 0], [DW - AL, L],         # 4-7  bouts des lignes de simple
    [AL, SV], [DW - AL, SV], [AL, L - SV], [DW - AL, L - SV],  # 8-11 lignes de service x simple
    [CX, SV], [CX, L - SV],                               # 12-13 les "T"
])
COURT_LINES = [
    ((0, 0), (DW, 0)), ((0, L), (DW, L)),                 # lignes de fond
    ((0, 0), (0, L)), ((DW, 0), (DW, L)),                 # lignes de cote (double)
    ((AL, 0), (AL, L)), ((DW - AL, 0), (DW - AL, L)),     # lignes de cote (simple)
    ((AL, SV), (DW - AL, SV)), ((AL, L - SV), (DW - AL, L - SV)),  # lignes de service
    ((CX, SV), (CX, L - SV)),                             # ligne mediane de service
]
NET = ((-0.914, L / 2), (DW + 0.914, L / 2))              # filet (poteaux a 0.914 m)


def fit_homography(kps, conf, min_conf, w, h):
    """Homographie terrain (m) -> image a partir des points surs. None si pas assez de points."""
    ok = (conf >= min_conf) & (kps[:, 0] >= 0) & (kps[:, 0] < w) & (kps[:, 1] >= 0) & (kps[:, 1] < h)
    if ok.sum() < 4:
        return None, ok
    Hm, inl = cv2.findHomography(COURT_POINTS[ok], kps[ok].astype(np.float32), cv2.RANSAC, 0.01 * w)
    return Hm, ok


def project(Hm, pts):
    return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), Hm).reshape(-1, 2)


def draw_court(img, Hm, kps=None, ok=None):
    for a, b in COURT_LINES:
        p, q = project(Hm, [a, b])
        cv2.line(img, tuple(map(int, p)), tuple(map(int, q)), (255, 0, 255), 2, cv2.LINE_AA)
    p, q = project(Hm, NET)
    cv2.line(img, tuple(map(int, p)), tuple(map(int, q)), (255, 200, 0), 1, cv2.LINE_AA)
    for i, (x, y) in enumerate(project(Hm, COURT_POINTS)):
        cv2.circle(img, (int(x), int(y)), 4, (255, 0, 255), -1)
    if kps is not None:  # points bruts du modele (vert = utilise, rouge = ecarte)
        for (x, y), good in zip(kps, ok):
            cv2.circle(img, (int(x), int(y)), 6, (0, 255, 0) if good else (0, 0, 255), 1)
    return img


MIN_MEDIAN_CONF = 0.85   # confiance mediane des 14 points en dessous de laquelle la detection est refusee
MIN_SURE_POINTS = 10     # nombre minimal de points surs (un joueur peut en cacher quelques-uns)


def detect(model, img, a):
    res = model.predict(img, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
    if res.keypoints is None or len(res.boxes) == 0:
        return None, None, None
    i = int(res.boxes.conf.argmax())
    data = res.keypoints.data[i].cpu().numpy()  # (14, 3) : x, y, confiance
    h, w = img.shape[:2]
    # Vraisemblance : sur un gros plan (jambes, raquette, public), le modele peut "inventer" un court ;
    # il est alors nettement moins sur de ses points. Sur un vrai court, la confiance mediane est ~1.
    if np.median(data[:, 2]) < MIN_MEDIAN_CONF or (data[:, 2] >= a.kp_conf).sum() < MIN_SURE_POINTS:
        return None, data[:, :2], data[:, 2] >= a.kp_conf
    Hm, ok = fit_homography(data[:, :2], data[:, 2], a.kp_conf, w, h)
    return Hm, data[:, :2], ok


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="video ou image")
    ap.add_argument("--weights", default=str(ROOT / "runs" / "court_pose" / "weights" / "best.pt"))
    ap.add_argument("--imgsz", type=int, default=960, help="comme a l'entrainement")
    ap.add_argument("--conf", type=float, default=0.25, help="confiance min du terrain")
    ap.add_argument("--kp-conf", type=float, default=0.5, help="confiance min d'un point pour l'homographie")
    ap.add_argument("--every", type=int, default=15, help="relance le modele toutes les N frames")
    ap.add_argument("--smooth", type=float, default=0.5, help="lissage de l'homographie (0 = aucun)")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = toute la video")
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    out_dir = ROOT / "runs" / "court"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(a.source)

    if src.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        img = cv2.imread(str(src))
        Hm, kps, ok = detect(model, img, a)
        if Hm is None:
            print("Terrain non trouve (moins de 4 points surs).")
        else:
            draw_court(img, Hm, kps, ok)
            print(f"{int(ok.sum())}/14 points utilises")
        cv2.imwrite(str(out_dir / f"{src.stem}_court.jpg"), img)
        print(f"-> {out_dir / (src.stem + '_court.jpg')}")
        return

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    writer = cv2.VideoWriter(str(out_dir / f"{src.stem}_court.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))

    def draw(frame, Hm):
        if Hm is not None:
            draw_court(frame, np.array(Hm))
        writer.write(frame)

    homs, n = court_track(src, model, a, progress=lambda i, n: i % 150 == 0 and print(f"  frame {i}/{n}", flush=True),
                          on_frame=draw)
    writer.release()
    (out_dir / f"{src.stem}_court.json").write_text(json.dumps(homs))
    print(f"Terrain trouve sur {len(homs)}/{n} frames -> {out_dir / (src.stem + '_court.mp4')}")


def court_track(video, model, a, progress=None, on_frame=None):
    """Homographie terrain (m) -> image pour chaque frame ou le terrain est visible : {frame: 3x3 en liste}.
    Le modele n'est relance que toutes les a.every frames ; entre deux, on garde la derniere (lissee).
    a : imgsz, conf, kp_conf, every, smooth, device, max_frames."""
    cap = cv2.VideoCapture(str(video))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_total, a.max_frames) if a.max_frames else n_total
    Hcur, homs = None, {}
    for fid in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        if fid % a.every == 0:
            if progress:
                progress(fid, n)
            Hnew, _, _ = detect(model, frame, a)
            if Hnew is None:
                Hcur = None  # plus de terrain (gros plan, ralenti...) : on n'affiche rien
            elif Hcur is None:
                Hcur = Hnew / Hnew[2, 2]
            else:  # lissage : evite que les lignes tremblent d'une detection a l'autre
                Hcur = a.smooth * Hcur + (1 - a.smooth) * (Hnew / Hnew[2, 2])
        Hl = (Hcur / Hcur[2, 2]).round(6).tolist() if Hcur is not None else None
        if Hl is not None:
            homs[fid] = Hl
        if on_frame:
            on_frame(frame, Hl)
    # Changement de plan entre deux detections (gros plan, ralenti) : on retrouve la frame exacte de la
    # coupure par dichotomie, sinon le terrain d'avant resterait affiche jusqu'a 'every' frames de trop.
    checks = list(range(0, n, a.every))
    seen = {f: (f in homs) for f in checks}
    for f0, f1 in zip(checks, checks[1:]):
        if seen[f0] == seen[f1]:
            continue
        lo, hi = f0, f1  # etat de f0 sur [f0, lo], etat de f1 sur [hi, f1]
        while hi - lo > 1:
            mid = (lo + hi) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ok, frame = cap.read()
            Hm = detect(model, frame, a)[0] if ok else None
            if (Hm is not None) == seen[f0]:
                lo = mid
            else:
                hi = mid
        for g in range(hi, f1):
            if seen[f0]:
                homs.pop(g, None)          # le terrain a disparu des la frame hi
            elif f1 in homs:
                homs[g] = homs[f1]         # le terrain est revenu des la frame hi
    cap.release()
    if progress:
        progress(n, n)
    return homs, n


if __name__ == "__main__":
    main()
