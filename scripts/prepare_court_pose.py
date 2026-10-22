"""Convertit le dataset de points du terrain (data\\) en dataset YOLO-pose, puis le zippe pour Colab.

Lancer depuis la racine du projet :  python scripts\\prepare_court_pose.py
Sortie : court_pose\\ (images JPG + labels) et court_pose.zip (a deposer dans Mon Drive/CourtVision_court/).

- 1 objet "court" par image, 14 points (ordre du dataset, voir COURT_POINTS dans predict_court.py).
- Separation PAR VIDEO (80 / 10 / 10) : les splits d'origine melangeaient les memes videos en train et val.
- Point hors de l'image -> visibilite 0 (le modele n'est pas penalise dessus).
- Boite = enveloppe des points visibles + marge.
"""
import json
import random
import shutil
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data"
OUT = ROOT / "court_pose"
W, H = 1280, 720
MARGIN = 0.03  # marge de la boite, en fraction de l'image
# miroir gauche/droite des 14 points (pour l'augmentation fliplr d'Ultralytics)
FLIP_IDX = [1, 0, 3, 2, 6, 7, 4, 5, 9, 8, 11, 10, 12, 13]


def convert(job):
    item, split = job
    img = cv2.imread(str(SRC / "images" / f"{item['id']}.png"))
    h, w = img.shape[:2]
    cv2.imwrite(str(OUT / "images" / split / f"{item['id']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    kps, vis = [], []
    for x, y in item["kps"]:
        if 0 <= x < w and 0 <= y < h:
            kps.append(f"{x / w:.6f} {y / h:.6f} 2")
            vis.append((x / w, y / h))
        else:
            kps.append("0 0 0")
    xs, ys = [p[0] for p in vis], [p[1] for p in vis]
    x0, x1 = max(0, min(xs) - MARGIN), min(1, max(xs) + MARGIN)
    y0, y1 = max(0, min(ys) - MARGIN), min(1, max(ys) + MARGIN)
    line = f"0 {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f} " + " ".join(kps)
    (OUT / "labels" / split / f"{item['id']}.txt").write_text(line + "\n")
    return split, len(vis)


def main():
    items = json.load(open(SRC / "data_train.json")) + json.load(open(SRC / "data_val.json"))
    videos = sorted({it["id"].rsplit("_", 1)[0] for it in items})
    random.Random(0).shuffle(videos)
    n = len(videos)
    split_of = {v: ("train" if i < 0.8 * n else "val" if i < 0.9 * n else "test") for i, v in enumerate(videos)}

    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ("train", "val", "test"):
        (OUT / "images" / s).mkdir(parents=True)
        (OUT / "labels" / s).mkdir(parents=True)

    jobs = [(it, split_of[it["id"].rsplit("_", 1)[0]]) for it in items]
    counts, n_vis = Counter(), Counter()
    with Pool(6) as pool:
        for i, (s, nv) in enumerate(pool.imap_unordered(convert, jobs, chunksize=16), 1):
            counts[s] += 1
            n_vis[nv] += 1
            if i % 1000 == 0:
                print(f"{i}/{len(jobs)}", flush=True)

    (OUT / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\n"
        "kpt_shape: [14, 3]\n"
        f"flip_idx: {FLIP_IDX}\n"
        "nc: 1\nnames:\n  0: court\n")
    print("images par split :", dict(counts), "| videos :", Counter(split_of.values()))
    print("points visibles par image :", dict(sorted(n_vis.items())))

    zpath = ROOT / "court_pose.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:  # JPG deja compresses
        for p in OUT.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(ROOT))
    print(f"-> {zpath} ({zpath.stat().st_size / 1e9:.2f} Go)")


if __name__ == "__main__":      # obligatoire sous Windows (multiprocessing)
    main()
