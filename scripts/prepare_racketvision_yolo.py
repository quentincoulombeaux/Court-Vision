"""Convertit le dataset tennis RacketVision (videos + CSV de centres de balle) en dataset YOLO.

Lancer depuis la racine du projet :  python scripts\\prepare_racketvision_yolo.py
Sortie : yolo_rv\\ (le dossier yolo\\ d'origine n'est pas touche).

- Seules les frames annotees sont extraites (~21 500 sur ~150 000), redimensionnees en 1280x720.
- Splits officiels RacketVision (info/train.json, val.json, test.json), un clip = un rallye.
- Les annotations ne donnent que le centre de la balle : on cree une boite carree de BOX px
  (mesuree en 1920x1080). Visibility=0 -> fichier label vide (image negative).
"""
import argparse
import json
import shutil
from multiprocessing import Pool
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "RacketVision" / "source" / "data" / "tennis"
OUT = ROOT / "yolo_rv"
OUT_W, OUT_H = 1280, 720


def convert_clip(job):
    split, match, rally, box = job
    df = pd.read_csv(SRC / "all" / match / "csv" / f"{rally}_ball.csv").fillna(0)
    labels = {int(r.Frame): r for r in df.itertuples()}
    cap = cv2.VideoCapture(str(SRC / "videos" / f"{match}_{rally}.mp4"))
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    last, fid, n = max(labels), 0, 0
    while fid <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if fid in labels:
            r = labels[fid]
            name = f"{match}_{rally}_{fid:04d}"
            cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"),
                        cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            line = ""
            if r.Visibility == 1 and (r.X > 0 or r.Y > 0):
                line = f"0 {r.X / vw:.6f} {r.Y / vh:.6f} {box / vw:.6f} {box / vh:.6f}\n"
            (OUT / "labels" / split / f"{name}.txt").write_text(line)
            n += 1
        fid += 1
    cap.release()
    return split, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=int, default=16, help="cote de la boite en px (sur 1920x1080)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if OUT.exists():
        shutil.rmtree(OUT)
    jobs = []
    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
        for match, rally in json.load(open(SRC / "info" / f"{split}.json")):
            jobs.append((split, match, rally, args.box))

    counts = {"train": 0, "val": 0, "test": 0}
    with Pool(args.workers) as pool:
        for i, (split, n) in enumerate(pool.imap_unordered(convert_clip, jobs), 1):
            counts[split] += n
            if i % 25 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} clips -- {counts}", flush=True)

    (OUT / "data.yaml").write_text("train: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames:\n  0: ball\n")
    print("images par split :", counts)


if __name__ == "__main__":      # obligatoire sous Windows (multiprocessing)
    main()
