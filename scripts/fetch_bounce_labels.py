"""Telecharge les etiquettes de rebonds du dataset tennis TrackNet (Huang et al., 2019) dans bounce_eval\\.

    python scripts\\fetch_bounce_labels.py

Seuls les 95 fichiers Label.csv (positions de balle + statut : 1 = frappe, 2 = rebond) sont lus a
distance dans l'archive tennis.zip (2,5 Go) hebergee sur Hugging Face : les images ne sont pas
telechargees. Necessite huggingface_hub (pip install huggingface_hub).
Ces etiquettes servent a mesurer la detection des rebonds avec scripts\\eval_bounces.py.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bounce_eval" / "tracknet_labels"
REMOTE = "datasets/oweng/tracknet_preprocessd_data/tennis.zip"


def main():
    from huggingface_hub import HfFileSystem
    OUT.mkdir(parents=True, exist_ok=True)
    with HfFileSystem().open(REMOTE, "rb", block_size=2 ** 20) as f:
        z = zipfile.ZipFile(f)
        names = [n for n in z.namelist() if n.lower().endswith("label.csv")]
        for n in names:
            # tennis/Dataset/game1/Clip1/Label.csv -> game1__Clip1__Label.csv
            parts = n.split("/")
            (OUT / "__".join(parts[-3:])).write_bytes(z.read(n))
    print(f"{len(names)} fichiers -> {OUT}")


if __name__ == "__main__":
    main()
