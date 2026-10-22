"""Evalue detect_bounces.py sur les 95 clips annotes du dataset tennis TrackNet (Huang et al., 2019).

    python scripts\\eval_bounces.py                    # trajectoires annotees (detecteur parfait)
    python scripts\\eval_bounces.py --noise            # + bruit et trous, comme un vrai detecteur
    python scripts\\eval_bounces.py --amin 20 --n 4    # memes reglages que detect_bounces.py

Annotations : bounce_eval\\tracknet_labels\\*.csv (1280x720, 30 fps), colonne status : 1 = frappe, 2 = rebond.
Un rebond detecte est juste s'il tombe a +/- --tol frames d'un rebond annote.

Les rebonds annotes sont classes par type de trajectoire avant le rebond :
  - "axial"   : la balle arrive dans l'axe de la camera (deplacement surtout vertical dans l'image) ;
  - "filet"   : rebond dans la bande centrale de l'image (zone du filet) ;
  - "autre".
"""
import csv
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_bounces import find_bounces, make_parser, to_samples  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "bounce_eval" / "tracknet_labels"
FRAME_H = 720


def load_clip(path):
    rows = list(csv.DictReader(open(path, newline="")))
    gt = []
    for i, r in enumerate(rows):
        v = int(r["visibility"])
        x = float(r["x-coordinate"] or 0)
        y = float(r["y-coordinate"] or 0)
        st = int(float(r["status"] or 0))
        gt.append((i, v, x, y, st))
    return gt


def category(gt, f):
    """Type de trajectoire a l'arrivee du rebond f."""
    pts = [(x, y) for i, v, x, y, _ in gt if f - 6 <= i <= f and v in (1, 2)]
    if len(pts) >= 3:
        dx, dy = abs(pts[-1][0] - pts[0][0]), abs(pts[-1][1] - pts[0][1])
        if dx < 0.35 * dy:
            return "axial"
    y = gt[f][3]
    if 0.38 * FRAME_H <= y <= 0.58 * FRAME_H:
        return "filet"
    return "autre"


def match(pred, truth, tol):
    """Appariement glouton un-pour-un a +/- tol frames."""
    pairs, used = [], set()
    for p in sorted(pred):
        best = min((t for t in truth if t not in used and abs(t - p) <= tol), key=lambda t: abs(t - p), default=None)
        if best is not None:
            used.add(best)
            pairs.append((p, best))
    return pairs


def main():
    ap = make_parser()
    ap.add_argument("--tol", type=int, default=2, help="tolerance en frames")
    ap.add_argument("--noise", action="store_true", help="bruit 1.5 px + 8 %% de detections manquees")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show-misses", action="store_true")
    a = ap.parse_args()
    rnd = random.Random(a.seed)

    tp = fp = fn = 0
    shot_tp = shot_fp = shot_fn = 0
    dist = []
    cat_total, cat_found = Counter(), Counter()
    misses = []
    for path in sorted(LABELS.glob("*.csv")):
        gt = load_clip(path)
        track = {}
        for i, v, x, y, _ in gt:
            if v in (1, 2):  # 3 = balle cachee : un detecteur ne la voit pas
                if a.noise:
                    if rnd.random() < 0.08:
                        continue
                    x, y = x + rnd.gauss(0, 1.5), y + rnd.gauss(0, 1.5)
                track[i] = (x, y)
        samples = to_samples(track, a.dedup, a.outlier)
        bounces, shots = find_bounces(samples, a, FRAME_H)

        truth = [i for i, *_, st in gt if st == 2]
        pairs = match([b["frame"] for b in bounces], truth, a.tol)
        found = {t for _, t in pairs}
        tp += len(pairs)
        fp += len(bounces) - len(pairs)
        fn += len(truth) - len(pairs)
        bmap = {b["frame"]: b for b in bounces}
        for p, t in pairs:
            b = bmap[p]
            dist.append(((b["x"] - gt[t][2]) ** 2 + (b["y"] - gt[t][3]) ** 2) ** 0.5)
        for t in truth:
            c = category(gt, t)
            cat_total[c] += 1
            if t in found:
                cat_found[c] += 1
            else:
                misses.append((path.stem, t, c))

        strue = [i for i, *_, st in gt if st == 1]
        sp = match(shots, strue, 3)
        shot_tp += len(sp)
        shot_fp += len(shots) - len(sp)
        shot_fn += len(strue) - len(sp)

    def prf(t, f_p, f_n):
        p = t / max(1, t + f_p)
        r = t / max(1, t + f_n)
        return p, r, 2 * p * r / max(1e-9, p + r)

    p, r, f1 = prf(tp, fp, fn)
    print(f"REBONDS  precision {p:.3f}  rappel {r:.3f}  F1 {f1:.3f}   (TP {tp}, FP {fp}, FN {fn})")
    if dist:
        dist.sort()
        print(f"         erreur de position : mediane {dist[len(dist) // 2]:.1f} px, 90 % < {dist[int(len(dist) * 0.9)]:.1f} px")
    print("         rappel par type :", ", ".join(
        f"{c} {cat_found[c]}/{cat_total[c]} ({cat_found[c] / max(1, cat_total[c]):.0%})" for c in ("axial", "filet", "autre")))
    sp, sr, sf = prf(shot_tp, shot_fp, shot_fn)
    print(f"FRAPPES  precision {sp:.3f}  rappel {sr:.3f}  F1 {sf:.3f}")
    if a.show_misses:
        for m in misses:
            print("  manque", *m)


if __name__ == "__main__":
    main()
