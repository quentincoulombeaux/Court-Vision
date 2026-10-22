# CourtVision

**Automatic tennis video analysis: ball tracking, bounce detection, in/out line calls, shot speed,
and a top-down bounce map, all served by a local web interface.**

CourtVision takes an ordinary broadcast-style tennis video. It finds the court, tracks the ball,
detects every bounce, judges each bounce **in**, **out** or **close call** against the official
court dimensions, and estimates the speed of each shot. Results are drawn live on top of the
original video in the browser, and the annotated video can be exported as MP4 with the original
sound.

It runs on an ordinary laptop **without a GPU**. Only model training needed a GPU, which was done
on free Google Colab.

---

## Table of contents

- [Features](#features)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Using the web interface](#using-the-web-interface)
- [Command-line tools](#command-line-tools)
- [Models](#models)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation and accuracy](#evaluation-and-accuracy)
- [Configuration reference](#configuration-reference)
- [Project structure](#project-structure)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Credits and licenses](#credits-and-licenses)

---

## Features

| | |
|---|---|
| **Court detection** | 14 court keypoints (YOLO11s-pose), fitted to the official court with a RANSAC homography. All lines are redrawn, including those hidden by players. Camera cuts (close-ups, replays) are detected to the exact frame. |
| **Ball tracking** | YOLO11s at 1280 px with a velocity-predicted tracker that rejects detections jumping to other objects. Raw detections are cached, so re-tracking takes seconds. |
| **Bounce detection** | Physics-based (no extra training): a bounce is a sudden *upward* kick in the ball's image velocity, framed by shots. Measured **F1 = 0.888** on 523 hand-labelled bounces. |
| **Line calls** | Each bounce is converted to metres on the court. Singles (default) or doubles, with an adjustable orange "close call" zone (±5 cm by default). |
| **Shot speed** | Estimated from the court's official dimensions, flight time and air drag. Serves are detected automatically. Typical results: rally shots 100–160 km/h, serves 180–200 km/h. |
| **Bounce map** | Top-down court view; bounces appear as the video plays. |
| **Web interface** | Video library, drag-and-drop import, background analysis with progress and ETA, live overlay layers, timeline of bounces and shots, statistics, English / French. |
| **Export** | Annotated MP4 (court lines, ball trail, animated impact markers, speed labels) with the original audio. |

The impact animation shows two ripples spreading **flat on the ground, in the court's perspective**,
a flash, and an IN / OUT / CLOSE label. A coloured mark then stays on the court: green for in,
red for out, amber for a close call.

---

## How it works

```
video ─► 1. Court   YOLO11s-pose every 15 frames ─► RANSAC homography (court metres → image)
                     + binary search for the exact frame of each camera cut
      ─► 2. Ball    YOLO11s on every frame ─► all candidates cached ─► velocity-predicted linking
      ─► 3. Bounces trajectory cleaning ─► motion fits before/after each point ─► "upward kick"
                     + shot detection (3 rules) ─► at most one bounce between two shots
                     ─► contact point refined with parabolas
      ─► 4. Calls   contact point → court metres (ball radius taken into account)
                     ─► shot speeds ─► analysis.json
      ─► web UI (live overlays, map, stats)  /  MP4 export
```

Each step is cached in `runs/analysis/<video>/`. Re-running only recomputes what is missing.

### 1. Court

- A YOLO11s-pose model finds 14 points: the 4 doubles corners, the ends of the 4 singles lines,
  the 4 service-line/singles intersections, and the 2 "T"s.
- A homography is fitted with RANSAC between these points and the **regulation court in metres**:
  23.77 m × 10.97 m, singles 8.23 m, service lines 6.40 m from the net.
- On ground-truth keypoints, the median reprojection error is **0.36 px**.
- The model runs every 15 frames and the homography is smoothed between runs. When the court
  appears or disappears between two checks, a binary search finds the exact frame of the cut.

### 2. Ball

- **Detection (slow, cached).** YOLO11s returns every candidate above confidence 0.10 on each
  frame.
- **Linking (fast).** The ball's next position is predicted from its last two positions. Only
  candidates inside a window are accepted: 3 % of the frame width plus twice the speed, widening
  with every missed frame. The candidate that best balances confidence and closeness to the
  prediction is kept. After 5 lost frames, the tracker re-acquires the most confident candidate.

### 3. Bounces

All pixel thresholds are expressed for 720p and scaled to the video's height.

1. **Cleaning.** Duplicated frames are merged (some broadcasts show each position twice).
   Isolated outliers far from the midpoint of their neighbours are removed.
2. **Motion fits.** For every position, a constant-velocity motion is fitted on the 5 positions
   before and the 5 after. Both segments must be geometrically straight and must not contain a
   sudden jump.
3. **The bounce signature.** The ground always pushes the ball **up** the image, while gravity
   always pulls it down. A candidate needs `kick = −Δvy / speed ≥ 0.4`. This also catches bounces
   along the camera axis, where the direction barely changes but the vertical speed drops
   sharply.
4. **Shots are boundaries**, since a ball bounces at most once between two shots. A shot is:
   - a large vertical U-turn of the ball in the image;
   - a sharp *downward* kick (only a racket can do that);
   - or a fast, accelerating *upward* departure (near-side player).

   A near-side U-turn where the ball *slows down* is a high bounce, not a shot. A bounce absorbs
   energy; a shot adds it.
5. **Contact point.** Parabolas (gravity) are fitted on each side of the best descent/ascent
   split. The contact is where the two vertical trajectories meet, usually between two frames.

### 4. Line calls and speed

- **Ground contact.** The detector returns the ball's centre. It is moved down by one ball radius
  (3.3 cm, converted with the local vertical scale) and projected onto the court.
- **Impossible bounces.** Anything more than 3 m outside the lines is discarded (a serve toss, or
  a tracking error).
- **The call is made in the browser** from the bounce's court coordinates. Switching
  singles/doubles or the close-call margin is therefore instant.
- **Speed.** The ball at the moment of the hit is placed at a known height: 1 m for groundstrokes.
  For serves, the height is solved so that the server stands 40 cm inside the baseline.
  - The height is converted to pixels with a local camera approximation (vertical px/m = lateral
    scale × cos(viewing angle), where sin(viewing angle) = depth scale / lateral scale). Only the
    homography is needed.
  - The **hit instant** is the moment of sudden acceleration. The top of the ball's curve comes
    1–4 frames too early.
  - Average speed = 3D distance from hit to bounce / flight time. The racket speed then corrects
    for air drag: `v0 = (e^{kD} − 1) / (k·t)`, with k = 0.019 m⁻¹.
  - The hit must be on the other side of the net from the bounce.

---

## Quick start

**Requirements:**
- Windows, macOS or Linux, with Python 3.11;
- [ffmpeg](https://ffmpeg.org/) on the `PATH`;
- the two trained weight files (see [Models](#models)).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

The trained weights are **not stored in the git repository** (≈115 MB). Download them from the
[Releases page](https://github.com/quentincoulombeaux/Court-Vision/releases), or train your own
(see [Training](#training)), and put them here:

```
runs/ball_rv/weights/best.pt       # ball detector (YOLO11s)
runs/court_pose/weights/best.pt    # court keypoints (YOLO11s-pose)
```

Start the interface:

```bash
python app/server.py                 # opens http://127.0.0.1:8000
python app/server.py --host 0.0.0.0  # reachable from other devices on your network
```

---

## Using the web interface

**Three columns:**
- **Left:** video library, rules (singles / doubles, close-call zone) and export.
- **Centre:** the player and the timeline.
- **Right:** layers, statistics and the bounce map.

1. **Add a video.** Click **Import**, or drop a file anywhere in the window. Videos that are not
   H.264 are converted automatically, and the audio is kept.
2. **Analyse.** Click **Run analysis**. The four steps (court, ball, bounces, line call) show their
   progress and an ETA. Expect about 1 s per frame on a laptop CPU; ball detection is 85 % of the
   time. Only one job runs at a time, and you can keep watching other videos meanwhile.
3. **Watch.**
   - Toggle the **layers**: court, ball, trail, bounces, speed.
   - Bounce markers on the timeline are coloured by verdict. Click one to jump just before it,
     in slow motion.
   - The bounce list and the map are clickable too.
4. **Export.** **Export video** renders an MP4 with the current rules and language, with the
   original sound. The download link appears in the Export panel.
5. **Re-run.** The ↻ icon next to a video recomputes its whole analysis.

**Keyboard shortcuts**

| Key | Action |
|---|---|
| `Space` | play / pause |
| `←` `→` | previous / next frame |
| `B` / `Shift+B` | next / previous bounce |
| `M` | sound on / off |
| `S` | singles ↔ doubles |
| `1`–`5` | toggle the court, ball, trail, bounces and speed layers |

**Colours:** green = in, red = out, amber = close call, grey = not judgeable (court not visible).

---

## Command-line tools

All scripts live in `scripts/` and should be run with the project's virtual environment.

| Script | Purpose |
|---|---|
| `pipeline.py <video> [--force]` | Full analysis (court → ball → bounces → calls → speeds). Writes `runs/analysis/<name>/analysis.json`. |
| `render_overlay.py <video> [--doubles] [--close-cm 5] [--lang en\|fr]` | Annotated MP4 export from an existing analysis. |
| `predict_video_yolo.py <video> --weights …` | Ball tracking only: CSV + video with trail. |
| `predict_court.py <video or image>` | Court detection only: lines drawn + per-frame homographies. |
| `detect_bounces.py <video> [--csv …] [--debug]` | Bounce detection on an existing tracking CSV (YOLO or TrackNet format). |
| `eval_bounces.py [--noise] [detection options]` | Measures bounce detection on the 523 labelled bounces. |
| `fetch_bounce_labels.py` | Downloads the TrackNet bounce labels used by `eval_bounces.py` into `bounce_eval/` (labels only, no images; needs `huggingface_hub`). |
| `prepare_racketvision_yolo.py` | Builds the ball dataset `yolo_rv/` from RacketVision. |
| `prepare_court_pose.py` | Builds the court dataset `court_pose/` + `court_pose.zip` for Colab. |
| `train.py --data … --cache none` | Local YOLO training (CPU-only machines: use Colab instead). |

Example:

```bash
python scripts/pipeline.py videos/match.mp4
python scripts/render_overlay.py videos/match.mp4 --lang en
```

---

## Models

| Model | Architecture | Trained on | Input | Validation |
|---|---|---|---|---|
| Ball | YOLO11s, 1 class | RacketVision tennis frames (17 494 train images) | 1280 px | mAP50 0.94, mAP50-95 0.61, recall 0.87 |
| Court | YOLO11s-pose, 14 keypoints | Court keypoint dataset, re-split by video (6 802 train images) | 960 px | 14/14 points, 0.3 px reprojection error on a 1080p US Open frame |

**Why YOLO11s.**
- The nano model misses more small balls.
- Medium and large models are 2–4× slower at inference for a small gain.
- The small model keeps CPU inference practical.

**Alternative evaluated: TrackNetV3** (RacketVision checkpoint).
- It sees 4 consecutive frames and the median background, and found the ball on 96 % of the
  frames of a test video, with a 1.4 px median error.
- At 1.1–1.4 s per frame on CPU, it is too slow to be the default.
- The comparison script was removed during cleanup; the RacketVision repository provides its own
  inference code. Note that its `inference.py` aligns frames `t-4…t-1` instead of `t-3…t` as
  used in training.

---

## Datasets

| Dataset | Use | Notes |
|---|---|---|
| [RacketVision](https://github.com/OrcustD/RacketVision) (tennis subset: 431 rally videos, ball centres) | Ball detector training | Only the annotated frames were extracted (21 544), resized to 1280×720. 16 px boxes were created around the ball centres. The official clip splits were kept. Frames without a visible ball are used as negatives. |
| Court keypoint dataset (8 841 broadcast frames, 498 videos, 14 points) | Court detector training | The original split leaked videos between train and val, so it was re-split **by video** (80/10/10). Out-of-frame points get visibility 0. Left/right flip mapping is provided. |
| TrackNet tennis dataset (Huang et al., 2019) — labels only | Bounce detection evaluation | 95 clips with 523 bounces and 516 shots, labelled per frame. Only the `Label.csv` files are read, from a remote zip, with `scripts/fetch_bounce_labels.py`. Not included in the repository. |
| Roboflow "tennis ball" dataset (384 images) | First prototype only | Too small: recall 0.26 on unseen scenes. Superseded. |

---

## Training

Both detectors were trained on **free Google Colab (Tesla T4)**, with the notebooks in `colab/`.
Upload the notebook, select a T4 GPU runtime and choose *Run all*.

- **`train_yolo11s_colab.ipynb`** (ball).
  - Rebuilds the dataset inside Colab from Hugging Face, with retries on rate limits.
  - Training settings: `yolo11s.pt`, `imgsz=1280`, a 10-hour time cap, `batch=16` (Ultralytics
    fell back to 8 on the T4).
  - Small-object augmentations: `scale=0.25`, no rotation, no vertical flip.
- **`train_court_pose_colab.ipynb`** (court).
  - Expects `court_pose.zip` in `My Drive/CourtVision_court/`.
  - Training settings: `yolo11s-pose.pt`, `imgsz=960`, an 8-hour cap.
  - `mosaic=0`, because mosaic cuts the court apart. Left/right flip is enabled with `flip_idx`.

Both notebooks save to Google Drive after every epoch and **resume automatically** after a
disconnection. Free Colab usage limits can still cut a session after about 5 hours.

---

## Evaluation and accuracy

**Bounce detection**, measured with `scripts/eval_bounces.py` on 523 labelled bounces (±2 frames):

| Version | F1 | Axial bounces found | Near-net bounces found |
|---|---|---|---|
| First version: angle between segments | 0.655 | 32 % | 73 % |
| + velocity "upward kick" | 0.79 | 47 % | 79 % |
| + high-bounce vs shot (speed ratio), 8 px tolerance | 0.844 | 81 % | 76 % |
| + downward-kick shots, interval bug fixed | 0.870 | 91 % | 88 % |
| **Current** (+ fast-upward shots, kick 0.4) | **0.888** | **91 %** | **90 %** |

With simulated detector noise (1.5 px jitter, 8 % missed detections): F1 0.868.

**Line calls.**
- At 30 fps the ball travels up to ~1 m between frames. The refined contact point reduces this
  error, but calls near the lines remain **indicative (±10–30 cm)**. This is not Hawk-Eye; the
  close-call zone is there to flag those cases.
- At the far baseline of a 1080p broadcast, 1 pixel is about 8 cm of court depth.

**Shot speed.**
- Accuracy is about **±10 %**: the hit height is assumed, spin is not modelled, and the hit and
  bounce are located to the nearest frame or so.
- Videos with duplicated frames (≈15 distinct positions per second) add ~±7 %.
- Reference values on the test videos: Alcaraz serve at Wimbledon **197 km/h**, Shelton serve at
  the US Open **183 km/h**, rally shots 100–162 km/h.

---

## Configuration reference

**Bounce detection** (`scripts/detect_bounces.py`; pixel values are for 720p and scaled
automatically):

| Option | Default | Meaning |
|---|---|---|
| `--n` | 5 | positions per segment, before and after |
| `--kick` | 0.4 | minimum upward kick for a bounce |
| `--max-dev` / `--max-rel-dev` | 8 px / 0.08 | straightness tolerance |
| `--min-len` | 15 px | minimum segment length |
| `--shot-prom` | 0.16 | vertical U-turn amplitude for a shot (fraction of frame height) |
| `--shot-ratio` | 0.8 | below this after/before speed ratio, a near-side U-turn is a bounce |
| `--shot-kick` | 0.8 | downward kick that marks a shot |
| `--hit-speed` / `--hit-ratio` | 0.8 H/s / 1.2 | fast accelerating upward departure = shot |
| `--min-gap` | 4 | minimum frames between a shot and a bounce |
| `--refine` | 1 | refine the contact point with parabolas |

**Pipeline constants** (`scripts/pipeline.py`):

| Constant | Value |
|---|---|
| `BALL_R` | 0.033 m |
| `MAX_OUT` | 3.0 m |
| `K_DRAG` | 0.019 m⁻¹ |
| `HIT_HEIGHT` | 1.0 m |
| `SERVE_INSIDE` | 0.4 m |

**Court** (`scripts/predict_court.py`): `--imgsz 960`, `--kp-conf 0.5`, `--every 15`,
`--smooth 0.5`.

---

## Project structure

```
app/
  server.py                  FastAPI server: library, background jobs, upload, export, static files
  static/                    index.html, style.css, app.js (no build step)
scripts/
  pipeline.py                full analysis, line-call geometry, shot speeds
  predict_video_yolo.py      ball candidates + velocity-predicted linking
  predict_court.py           court keypoints, homography, camera-cut detection
  detect_bounces.py          bounce and shot detection
  render_overlay.py          annotated MP4 export (with audio)
  eval_bounces.py            bounce evaluation on labelled data
  prepare_*.py, train.py     dataset building and local training
colab/                       training notebooks (ball, court)
bounce_eval/                 TrackNet bounce labels (fetched, not versioned)
videos/                      web UI library
runs/
  ball_rv/weights/best.pt    ball model
  court_pose/weights/best.pt court model
  analysis/<video>/          cached analysis (court.json, candidates.json, ball.csv, analysis.json, exports)
requirements.txt
JOURNAL_DU_PROJET.md         full project log (French)
```

Large training data (`data/`, `court_pose/`, `yolo_rv/`, `RacketVision/`), videos, analysis
results and model weights are not versioned (see `.gitignore`). They are only needed to retrain
the models, or are produced when you run the system.

---

## Known limitations

- **Ball tracking is the weakest link.** The ball can be lost exactly at a bounce (fast winners),
  or the tracker can briefly follow the racket head.
- **Serves are not judged against the service box yet**: every bounce is judged against the full
  court.
- **Slow volleys** are not always marked as shots.
- **Broadcast camera assumed**: an elevated view from behind the baseline. Unusual angles have not
  been tested.
- **CPU analysis is slow** (about 1 s per frame). Analyse demo videos in advance; the results are
  cached.

## Roadmap

- Service-box rule for serves.
- Player detection to anchor shots (a shot always happens next to a player).
- Kalman filtering / TrackNet fallback for difficult ball segments.
- Speed calibration against official on-screen serve speeds.
- Rally statistics per player (depth, direction, winners, errors).

---

## Credits and licenses

- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)**: YOLO11 and YOLO11-pose
  (AGPL-3.0). Commercial use of the trained models requires an Ultralytics licence.
- **[RacketVision](https://github.com/OrcustD/RacketVision)**: ball dataset and TrackNetV3
  checkpoint (MIT). Dong et al., *RacketVision: A Multiple Racket Sports Benchmark for Unified
  Ball and Racket Analysis*, AAAI 2026.
- **TrackNet tennis dataset**: Huang et al., *TrackNet: A Deep Learning Network for Tracking
  High-speed and Tiny Objects in Sports Applications*, 2019. Used for evaluation labels only.
- **Court keypoint dataset**: broadcast frames with 14 court keypoints, in the format of the
  open-source *TennisCourtDetector* project. Check its licence before redistributing.
- **Roboflow tennis-ball dataset**: first prototype only.

The test videos are broadcast excerpts used for personal research. They are not covered by this
project and should not be redistributed.
