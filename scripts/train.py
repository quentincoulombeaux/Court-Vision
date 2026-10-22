"""Entrainement du detecteur de balle de tennis (YOLO, une classe) pour un PC sans GPU.

Installation (une fois), depuis la racine du projet, dans PowerShell :
    python -m venv .venv
    .venv\\Scripts\\activate
    pip install ultralytics

Usage :
    python scripts\\train.py --smoke      # test rapide : 1 epoque sur 10 % des images, sert a mesurer la vitesse
    python scripts\\train.py              # entrainement complet (nano, 1280 px, 50 epoques)
    python scripts\\train.py --resume     # reprend apres une interruption (Ctrl+C, veille, coupure)

Conseils : branche sur secteur, mode d'alimentation "Performances" Windows, veille desactivee,
et ferme le navigateur : le processeur est limite thermiquement, la RAM compte pour le cache.
Licence : Ultralytics est sous AGPL-3.0 (sans consequence pour un usage personnel).
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "yolo" / "data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt", help="poids de depart (yolo11n.pt, yolov8n.pt, yolo11s.pt...)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=1280, help="ne pas descendre sous 1280 : la balle ferait ~6 px")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="ball_v1")
    ap.add_argument("--data", default=str(DATA), help="yolo\\data.yaml (Roboflow) ou yolo_rv\\data.yaml (RacketVision)")
    ap.add_argument("--cache", default="ram", help="ram (petit dataset), disk ou none (yolo_rv : 17 500 images, trop gros pour la RAM)")
    ap.add_argument("--fraction", type=float, default=1.0, help="part du train utilisee (ex. 0.3 pour aller plus vite)")
    ap.add_argument("--smoke", action="store_true", help="1 epoque sur 10 %% du train pour mesurer le temps")
    ap.add_argument("--resume", action="store_true", help="reprend le dernier entrainement de --name")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Appareil : {device} | torch {torch.__version__} | threads CPU : {torch.get_num_threads()}")

    runs = ROOT / "runs"
    if args.resume:
        last = runs / args.name / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"Rien a reprendre : {last} n'existe pas.")
        YOLO(str(last)).train(resume=True)
        best = runs / args.name / "weights" / "best.pt"
    else:
        model = YOLO(args.model)
        t0 = time.time()
        model.train(
            data=args.data,
            imgsz=args.imgsz,
            epochs=1 if args.smoke else args.epochs,
            fraction=0.1 if args.smoke else args.fraction,
            batch=args.batch,
            workers=args.workers,
            device=device,
            cache=False if args.cache == "none" else args.cache,  # yolo\ : 384 images a 1280 px = ~1 Go de RAM
            project=str(runs),
            name=("smoke" if args.smoke else args.name),
            exist_ok=args.smoke,
            patience=15,          # arret anticipe si la validation ne progresse plus
            # Reglages pour un objet minuscule :
            scale=0.25,           # defaut 0.5 : rapetisserait une balle de 12 px a 6 px
            mosaic=1.0,
            close_mosaic=10,      # derniere phase sans mosaique
            degrees=0.0,          # pas de rotation : la camera TV est horizontale
            flipud=0.0,
            fliplr=0.5,
            plots=True,
        )
        dt = time.time() - t0
        print(f"\nDuree : {dt/60:.1f} min")
        if args.smoke:
            print("Estimation grossiere d'une epoque complete : ~%.0f min (la 1re epoque inclut du chargement)" % (dt / 60 * 10))
            return
        best = runs / args.name / "weights" / "best.pt"

    # Evaluation finale : val (meme scene que le train, optimiste) et test (terre battue + gazon, jamais vus)
    final = YOLO(str(best))
    out = {}
    for split in ("val", "test"):
        m = final.val(data=args.data, split=split, imgsz=args.imgsz, batch=args.batch, device=device,
                      plots=False, name=f"{args.name}_eval_{split}", project=str(runs), exist_ok=True)
        out[split] = dict(precision=float(m.box.mp), recall=float(m.box.mr),
                          mAP50=float(m.box.map50), mAP50_95=float(m.box.map))
    print(json.dumps(out, indent=2))
    (runs / args.name / "final_metrics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":      # obligatoire sous Windows (workers DataLoader)
    main()
