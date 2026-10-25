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
import detect_players  # noqa: E402
import predict_court  # noqa: E402
import predict_video_yolo  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "runs" / "analysis"
# A augmenter quand le calcul change : l'interface signale les analyses faites avec une version plus ancienne
PIPELINE_VERSION = 22
BALL_WEIGHTS = ROOT / "runs" / "ball_rv" / "weights" / "best.pt"
COURT_WEIGHTS = ROOT / "runs" / "court_pose" / "weights" / "best.pt"
STEPS = [("court", "Terrain", 0.08), ("players", "Joueurs", 0.12), ("ball", "Balle", 0.75), ("bounces", "Rebonds", 0.03),
         ("judge", "Jugement", 0.02)]

_models = {}


def model(path):
    """Les modeles sont charges une seule fois (l'interface enchaine plusieurs analyses)."""
    from ultralytics import YOLO
    if str(path) not in _models:
        _models[str(path)] = YOLO(str(path))
    return _models[str(path)]


def detect_players_weights():
    """Poids YOLO11n "person" (telecharges une fois dans runs/models)."""
    detect_players.person_model()
    return detect_players.PERSON_WEIGHTS


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


# ---------------------------------------------------------------- vitesse : trajectoire 3D ajustee
G = 9.81
H_SIGMA = 0.1   # confiance (m) dans la hauteur de frappe de depart (mesure sur le joueur, ou hypothese)


def project3d(H, X, Y, Z):
    """Point 3D du court (m, Z vers le haut) -> pixel. Meme approximation locale de la camera que
    vertical_scale : le point au sol par l'homographie, puis Z metres plus haut a l'echelle verticale locale."""
    H = np.asarray(H, float)

    def img(x, y):
        q = H @ np.array([x, y, 1.0])
        return q[:2] / q[2]
    p = img(X, Y)
    s_lat = np.linalg.norm(img(X + 0.5, Y) - img(X - 0.5, Y))
    s_dep = np.linalg.norm(img(X, Y + 0.5) - img(X, Y - 0.5))
    vs = s_lat * math.sqrt(max(0.02, 1 - min(0.98, (s_dep / max(1e-6, s_lat)) ** 2)))
    return np.array([p[0], p[1] - Z * vs])


def simulate(theta, times, dt=0.002):
    """Vol de la balle : gravite + frottement de l'air (quadratique, K_DRAG). theta = (X, Y, Z, vx, vy, vz)
    au depart (t = 0). Renvoie la position 3D aux instants demandes (s)."""
    p, v = np.array(theta[:3], float), np.array(theta[3:], float)
    t = 0.0
    res = np.zeros((len(times), 3))
    for idx in np.argsort(times):
        while t + dt <= times[idx]:
            a = np.array([0.0, 0.0, -G]) - K_DRAG * np.linalg.norm(v) * v
            v = v + a * dt
            p = p + v * dt
            t += dt
        res[idx] = p + v * (times[idx] - t)
    return res


def levenberg(fun, x0, iters=40):
    """Moindres carres non lineaires (Levenberg-Marquardt, jacobienne numerique) : 6 inconnues seulement."""
    x = np.array(x0, float)
    r = fun(x)
    cost, lam = r @ r, 1e-2
    for _ in range(iters):
        J = np.empty((len(r), len(x)))
        for j in range(len(x)):
            dx = 1e-4 * max(1.0, abs(x[j]))
            xp = x.copy()
            xp[j] += dx
            J[:, j] = (fun(xp) - r) / dx
        A, g = J.T @ J, J.T @ r
        while True:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
            xn = x + step
            rn = fun(xn)
            if rn @ rn < cost:
                x, r, cost, lam = xn, rn, rn @ rn, max(lam / 3, 1e-7)
                break
            lam *= 4
            if lam > 1e7:
                return x, r
        if np.linalg.norm(step) < 1e-5:
            break
    return x, r


def fit_shot_3d(track, homs, t_hit, t_bounce, bounce_xy, start_xy, h0, fps, img_h):
    """Reconstruit le vol en 3D : on cherche la position et la vitesse de depart telles que la trajectoire
    physique simulee, projetee dans l'image, colle a TOUTES les positions observees entre la frappe et le
    rebond, et touche le sol au point de rebond. Renvoie dict(kmh, z0, rms_px, traj) ou None si l'ajustement
    ne colle pas (on garde alors la methode simple)."""
    k = img_h / 720
    tau_b = (t_bounce - t_hit) / fps
    frames = [f for f in range(int(math.floor(t_hit)) + 1, int(math.ceil(t_bounce))) if f in track and f in homs
              and 0.5 / fps < (f - t_hit) / fps < tau_b - 0.5 / fps]
    if len(frames) < 5 or tau_b <= 0.15:
        return None
    taus = np.array([(f - t_hit) / fps for f in frames])
    obs = np.array([track[f] for f in frames], float)
    Hs = [homs[f] for f in frames]
    sigma = 2.0 * k

    def residuals(th):
        P = simulate(th, np.append(taus, tau_b))
        r = [(project3d(H, *P[i]) - obs[i]) / sigma for i, H in enumerate(Hs)]
        r.append((P[-1, :2] - bounce_xy) / 0.15)                     # touche le sol au point de rebond
        r.append(np.array([P[-1, 2] / 0.05]))
        r.append(np.array([(th[2] - h0) / H_SIGMA]))                  # hauteur de frappe (mesuree ou estimee)
        r.append((np.array(th[:2]) - start_xy) / 2.0)
        return np.concatenate(r)

    # depart : ligne droite frappe -> rebond, avec la composante verticale qui compense la gravite
    d = np.array([bounce_xy[0] - start_xy[0], bounce_xy[1] - start_xy[1]])
    v_h = d / tau_b * (1 + K_DRAG * np.linalg.norm(d) / 2)
    vz = (0 - h0 + 0.5 * G * tau_b ** 2) / tau_b
    th, r = levenberg(residuals, [start_xy[0], start_xy[1], h0, v_h[0], v_h[1], vz])
    n = len(frames)
    rms = float(np.sqrt(np.mean(r[:2 * n] ** 2)) * sigma)
    kmh = float(np.linalg.norm(th[3:]) * 3.6)
    if rms > 4 * k or not 30 <= kmh <= 260 or not 0.2 <= th[2] <= 3.8:
        return None
    ts = np.linspace(0, tau_b, 24)
    traj = simulate(th, ts)
    # apres le rebond (pour la relecture) : rebond amorti, environ 0,35 s
    vb = (simulate(th, np.array([tau_b]))[0] - simulate(th, np.array([tau_b - 0.01]))[0]) / 0.01
    after = [traj[-1] + np.array([vb[0] * 0.8 * t, vb[1] * 0.8 * t, -vb[2] * 0.7 * t - 0.5 * G * t * t])
             for t in np.linspace(0.03, 0.35, 8)]
    pts = np.vstack([traj] + [a[None] for a in after if a[2] >= 0])
    return dict(kmh=round(kmh), z0=round(float(th[2]), 2), rms_px=round(rms, 1),
                traj=[[round(float(v), 3) for v in q] for q in pts], traj_bounce=len(traj) - 1)


def hidden_bounces(bounces, shots, track, homs, players, fps, img_h):
    """Rebond cache : la balle passe d'un joueur a l'autre (frappes de part et d'autre du filet) sans
    rebond detecte, et le receveur n'est pas au filet (ce n'est pas une volee). Le rebond a eu lieu sans
    etre vu (balle cachee par le joueur, suivi accroche a autre chose). On reconstruit le vol en 3D
    (gravite + frottement) sur les positions vues apres la frappe, et on cherche ou la balle touche le sol.
    Renvoie des rebonds au format du detecteur (point au sol dans l'image), marques "hidden"."""
    k = img_h / 720
    who = serve_zone(players)
    out = []
    for s0, s1 in zip(shots, shots[1:]):
        if any(s0 < b["frame"] < s1 for b in bounces) or s1 - s0 < 0.6 * fps or s0 not in track or s1 not in track:
            continue
        a, b = who(s0, *track[s0]), who(s1, *track[s1])
        if not a or not b or a == b:
            continue
        hitter, receiver = (detect_players.player_near(players, a, s0), detect_players.player_near(players, b, s1))
        if not hitter or not receiver or abs(receiver[5] - NET_Y) < 4.5:
            continue  # volee au filet : pas de rebond
        rx1, ry1, rx2, ry2 = receiver[:4]
        frames = []
        for f in range(s0 + 4, s1):  # (les toutes premieres positions apres la frappe sont souvent sur la raquette)
            if f not in track or f not in homs or track.get(f - 1) == track[f]:
                continue  # image dupliquee (meme position que la precedente) : instant incertain, on l'ignore
            q = detect_players.player_near(players, b, f, 3)
            if q and q[0] <= track[f][0] <= q[2] and q[1] <= track[f][1] <= q[3]:
                break  # la balle "entre" dans le receveur : suivi accroche au joueur, on s'arrete
            frames.append(f)
        if len(frames) < 8:
            continue
        # instants regularises (video a images dupliquees : 20 img/s converties en 30, etc.)
        st, step = detect_bounces.sample_times([(f, *track[f]) for f in frames])
        taus = np.array([(frames[0] - s0 + t * step) / fps for t in st])
        obs = np.array([track[f] for f in frames], float)
        Hs = [homs[f] for f in frames]
        sigma, h0 = 2.0 * k, 1.0
        start = np.array(hitter[4:6], float)
        far = start[1] < NET_Y

        def residuals(th):
            P = simulate(th, taus)
            r = [(project3d(H, *P[i]) - obs[i]) / sigma for i, H in enumerate(Hs)]
            r.append(np.array([(th[2] - h0) / 0.4]))
            r.append((np.array(th[:2]) - start) / 2.0)
            return np.concatenate(r)
        tgt = np.array([start[0], predict_court.L - 4 if far else 4.0])
        tau_g = (s1 - s0) / fps * 0.75
        v_h = (tgt - start) / tau_g
        th, r = levenberg(residuals, [start[0], start[1], h0, v_h[0], v_h[1], 0.5 * G * tau_g - h0 / tau_g])
        rms = float(np.sqrt(np.mean(r[:2 * len(frames)] ** 2)) * sigma)
        if rms > 4 * k:
            continue
        ts = np.arange(taus[0], (s1 - s0) / fps, 0.002)
        P = simulate(th, ts)
        hit = np.nonzero(P[:, 2] <= 0)[0]
        if not len(hit):
            continue
        t_b, g = ts[hit[0]], P[hit[0]]
        f_b = s0 + t_b * fps
        # le sol est touche apres la derniere position vue, de l'autre cote du filet, pas trop loin des lignes
        last = s0 + taus[-1] * fps
        if f_b < last - 2 or (g[1] < NET_Y) == far or signed_distance(g[:2], True) > 1.0:
            continue
        f_b = max(f_b, last + 1)  # la balle etait encore vue : le contact a lieu juste apres
        H = homs.get(int(round(f_b))) or Hs[-1]
        px = project3d(H, g[0], g[1], BALL_R)
        out.append(dict(frame=int(round(f_b)), t=round(float(f_b), 2), x=float(px[0]), y=float(px[1]),
                        angle=0.0, kick=0.0, hidden=True, hidden_rms=round(rms, 1)))
    return out


PLAYER_HEIGHT = 1.85   # taille moyenne d'un joueur professionnel (m)


def player_hit_height(players, side, frame, ball_xy):
    """Hauteur de la balle a la frappe, mesuree sur le joueur : la boite du joueur (pieds -> tete) fait
    environ 1,85 m, donc hauteur = 1,85 x (pieds - balle) / (pieds - tete). Plus fiable que l'echelle du
    court pres du joueur proche. Pas au service : la boite englobe alors le bras leve et la raquette.
    None si pas de joueur a cote de la balle."""
    p = detect_players.player_near(players, side, frame) if players else None
    if not p:
        return None
    x1, y1, x2, y2, X, Y = p
    w = x2 - x1
    if not (x1 - 1.5 * w <= ball_xy[0] <= x2 + 1.5 * w) or y2 - y1 < 20:
        return None
    h = PLAYER_HEIGHT * (y2 - ball_xy[1]) / (y2 - y1)
    return h if 0.3 <= h <= 2.8 else None


def shot_speeds(bounces, shots, track, homs, info, players=None):
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
            # (ou lancer reconnu : la balle redescend du lancer et accelere deja un peu avant l'impact)
            serve = first_of_point and (slow or detect_bounces.is_toss(track, fb, info["width"], H_img, fps))
            side = "far" if b["court"][1] > NET_Y else "near"     # le frappeur est de l'autre cote du rebond
            h_player = None if serve else player_hit_height(players, side, fb, track[fb])
            if h_player is not None:
                h = h_player                                       # hauteur mesuree sur le joueur
            else:
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
                              hit_height=round(h, 2), h_prior=round(h, 2),
                              height_from="player" if h_player is not None else "assumed",
                              hit_court=[round(g[0], 2), round(g[1], 2)], method="2pts", kmh_2pts=round(v0 * 3.6))
            fit = fit_shot_3d(track, homs, t_hit, b.get("t", b["frame"]), np.array(b["court"], float),
                              np.array(g, float), h, fps, H_img)
            if fit:  # trajectoire 3D ajustee sur tout le vol : remplace l'estimation a deux points
                b["speed"].update(kmh=fit["kmh"], hit_height=fit["z0"], method="3d", rms_px=fit["rms_px"],
                                  traj=fit["traj"], traj_bounce=fit["traj_bounce"])
            break


def continuity_violations(bounces, shots):
    """Rebonds juges faute (en simple) alors que l'echange continue ensuite : un rebond bon plus tard, avec
    au moins une frappe entre les deux. Exception : faute au service suivie d'un nouveau service."""
    bad = []
    for i, b in enumerate(bounces):
        if judge(b) != "out":
            continue
        later = bounces[i + 1:]
        resumed = [x for x in later if judge(x) in ("in", "close") and any(b["frame"] < s < x["frame"] for s in shots)]
        if not resumed:
            continue  # vraie faute : le point s'arrete (les rebonds d'apres-point ne comptent pas)
        if (b.get("speed") or {}).get("serve") and any((x.get("speed") or {}).get("serve") for x in later):
            continue  # premiere balle faute, puis seconde balle
        bad.append(b["frame"])
    return bad


def fault_then_good(bounces):
    """Regle 1 : un rebond ne peut pas etre faute si le rebond suivant est bon (le point aurait ete fini).
    Exception : premiere balle de service faute, puis seconde balle."""
    bad = []
    for b, nxt in zip(bounces, bounces[1:]):
        if judge(b) == "out" and judge(nxt) in ("in", "close"):
            if (b.get("speed") or {}).get("serve") and (nxt.get("speed") or {}).get("serve"):
                continue
            bad.append(b["frame"])
    return bad


def same_side_pairs(bounces, shots=(), in_racket=None):
    """Regle 2 : pas deux rebonds d'affilee du meme cote du filet si un rebond de l'autre cote suit dans le
    point (un double rebond termine le point). On ecarte le moins fiable des deux : celui de la seconde
    passe, sinon celui dont la cassure ("coup vers le haut") est la plus faible.
    Exception : une VOLEE entre les deux (frappe de chaque joueur, la balle ne touche pas le sol chez le
    voleyeur) -> au moins deux frappes entre les deux rebonds : c'est normal."""
    side = lambda b: None if b.get("court") is None else (b["court"][1] > NET_Y)  # noqa: E731
    drop = []
    for i in range(len(bounces) - 1):
        a, b = bounces[i], bounces[i + 1]
        if side(a) is None or side(a) != side(b):
            continue
        if not any(side(x) is not None and side(x) != side(a) for x in bounces[i + 2:]):
            continue  # pas de rebond de l'autre cote ensuite : double rebond, fin du point
        between = sum(a["frame"] < s < b["frame"] for s in shots)
        if between >= 2:
            continue  # volee entre les deux
        if between == 0:
            # aucune frappe entre les deux : la balle n'a pas pu changer de cote, un seul est le vrai rebond.
            # Le vrai est le point le plus bas a l'image (la balle au sol) ; l'autre est une cassure sur la
            # montee ou la descente.
            drop.append(min((a, b), key=lambda x: x["y"])["frame"])
            continue
        # on ne retire que celui qui est dans la zone de frappe d'un joueur (une frappe mal classee) ; si
        # aucun ne l'est, un rebond a sans doute ete manque entre les deux : en supprimer un serait pire
        suspects = [x for x in (a, b) if in_racket and in_racket(x["frame"], x["x"], x["y"])]
        if not suspects:
            continue
        drop.append(min(suspects, key=lambda x: (not x.get("relaxed"), x.get("kick", 0)))["frame"])
    return drop


def serve_boxes(bounces, players):
    """Regle du service : le premier rebond doit tomber dans le carre de service en diagonale du serveur.
    Cote du serveur = cote de la frappe ; sa position laterale = celle du joueur detecte (sinon celle de
    la frappe). Ajoute bounce["box"] = [x0, x1, y0, y1] (m) aux rebonds de service."""
    AL, DW, SV, CX = predict_court.AL, predict_court.DW, predict_court.SV, predict_court.CX
    for b in bounces:
        sp = b.get("speed")
        if not sp or not sp.get("serve") or b.get("court") is None:
            continue
        far_server = sp["hit_court"][1] < NET_Y
        p = detect_players.player_near(players, "far" if far_server else "near", sp["hit_frame"]) if players else None
        x_server = p[4] if p else sp["hit_court"][0]
        x0, x1 = (AL, CX) if x_server > CX else (CX, DW - AL)      # en diagonale
        y0, y1 = (NET_Y, predict_court.L - SV) if far_server else (SV, NET_Y)
        if edge_distance(b["court"], [x0, x1, y0, y1])[0] > 2.5:
            # un service retombe pres de son carre : sinon ce n'etait pas un service (l'echange continue)
            sp["serve"] = False
            continue
        b["box"] = [round(x0, 3), round(x1, 3), round(y0, 3), round(y1, 3)]


def signed_distance(c, doubles):
    """Distance (m) du rebond a la limite du terrain : < 0 dedans, > 0 dehors."""
    L, DW, AL = predict_court.L, predict_court.DW, predict_court.AL
    x0, x1 = (0.0, DW) if doubles else (AL, DW - AL)
    dx = max(x0 - c[0], c[0] - x1)
    dy = max(-c[1], c[1] - L)
    if dx > 0 and dy > 0:
        return float(np.hypot(dx, dy))
    return float(max(dx, dy))


PIXEL_SIGMA = 1.5   # precision du centre de la balle dans l'image (px, pour une video 720p)


def position_error(H, c, img_h):
    """Incertitude (m) de la position du rebond sur le court, en largeur et en profondeur : la precision
    du detecteur (~1,5 px) convertie avec l'echelle locale du court. Cote eloigne d'une video en basse
    definition, 1 px peut valoir 10 cm en profondeur : un verdict a 5 cm pres n'y a pas de sens."""
    p = cv2.perspectiveTransform(np.float32([[[c[0] - 0.5, c[1]]], [[c[0] + 0.5, c[1]]],
                                             [[c[0], c[1] - 0.5]], [[c[0], c[1] + 0.5]]]), np.array(H)).reshape(4, 2)
    s_lat, s_dep = float(np.linalg.norm(p[1] - p[0])), float(np.linalg.norm(p[3] - p[2]))
    sig = PIXEL_SIGMA * max(1.0, img_h / 720)
    return [round(sig / max(1e-6, s_lat), 3), round(sig / max(1e-6, s_dep), 3)]


def edge_distance(c, rect):
    """Distance signee (m) au bord du rectangle [x0, x1, y0, y1] (< 0 dedans) et l'axe du bord le plus proche."""
    x0, x1, y0, y1 = rect
    dx, dy = max(x0 - c[0], c[0] - x1), max(y0 - c[1], c[1] - y1)
    if dx > 0 and dy > 0:
        return float(np.hypot(dx, dy)), ("x" if dx >= dy else "y")
    return float(max(dx, dy)), ("x" if dx >= dy else "y")


def box_distance(c, box):
    x0, x1, y0, y1 = box
    dx, dy = max(x0 - c[0], c[0] - x1), max(y0 - c[1], c[1] - y1)
    return float(np.hypot(dx, dy)) if dx > 0 and dy > 0 else float(max(dx, dy))


def judge(bounce, doubles=False, close_m=0.05):
    """'in', 'out', 'close' (trop pres de la ligne pour trancher) ou 'unknown' (terrain non visible).
    Un rebond de service est juge sur le carre de service (bounce["box"])."""
    if bounce.get("court") is None:
        return "unknown"
    L_, DW_, AL_ = predict_court.L, predict_court.DW, predict_court.AL
    rect = bounce.get("box") or ([0.0, DW_, 0.0, L_] if doubles else [AL_, DW_ - AL_, 0.0, L_])
    s, axis = edge_distance(bounce["court"], rect)
    err = bounce.get("err") or [0.0, 0.0]
    margin = max(close_m, err[0] if axis == "x" else err[1])   # zone "limite" = au moins l'incertitude locale
    if abs(s) <= margin:
        return "close"
    return "in" if s < 0 else "out"


def racket_zone(players):
    """Test 'la balle est dans la zone de frappe d'un joueur' : autour de sa boite (bras + raquette),
    au-dessus de ses pieds (un rebond a ses pieds reste un rebond)."""
    def test(f, x, y):
        for side in ("far", "near"):
            p = detect_players.player_near(players, side, f, 3)
            if not p:
                continue
            x1, y1, x2, y2 = p[:4]
            w, h = x2 - x1, y2 - y1
            if x1 - 0.8 * w <= x <= x2 + 0.8 * w and y1 - 0.25 * h <= y <= y2 - 0.12 * h:
                return True
        return False
    return test


def serve_zone(players):
    """Test 'la balle est au-dessus ou autour d'un joueur' (lancer de service, rebond de preparation) :
    largement autour de sa boite, jusqu'a 1,6 fois sa taille au-dessus de sa tete."""
    def test(f, x, y):
        for side in ("far", "near"):
            p = detect_players.player_near(players, side, f, 4)
            if not p:
                continue
            x1, y1, x2, y2 = p[:4]
            w, h = x2 - x1, y2 - y1
            if x1 - w <= x <= x2 + w and y1 - 1.6 * h <= y <= y2:
                return side  # "far" / "near" (vrai)
        return None
    return test


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

    # 2. Joueurs (YOLO11n "person", une image sur deux) : positions sur le court
    players_path = out / "players.json"
    if not players_path.exists():
        res_p = detect_players.track_players(video, model(detect_players_weights()), homs,
                                             progress=lambda i, n: report("players", i / max(1, n), f"frame {i}/{n}"))
        players_path.write_text(json.dumps(res_p, separators=(",", ":")))
    players = json.loads(players_path.read_text())
    report("players", 1.0, f"joueurs suivis sur {len(players['frames'])} images")

    # 3. Balle (l'etape longue : ~1 s par frame sur CPU)
    # candidates.json : toutes les detections YOLO (lent) ; ball.csv : la balle choisie parmi elles (rapide)
    ball_path, cand_path = out / "ball.csv", out / "candidates.json"
    if not cand_path.exists() and not ball_path.exists():
        cands = predict_video_yolo.detect_candidates(video, model(BALL_WEIGHTS),
                                                     progress=lambda i, n: report("ball", i / max(1, n), f"frame {i}/{n}"))
        cand_path.write_text(json.dumps(cands, separators=(",", ":")))
    if not ball_path.exists():  # (une ancienne analyse peut n'avoir que ball.csv : on le garde)
        rows = predict_video_yolo.link_track(json.loads(cand_path.read_text()), info["width"])
        predict_video_yolo.write_track_csv(rows, ball_path)
    # trous du suivi combles (detections ecartees recuperees, sinon paraboles / rebond reconstitue)
    raw = [dict(Frame=int(r["Frame"]), X=float(r["X"]), Y=float(r["Y"]), Visibility=int(r["Visibility"]),
                Confidence=float(r["Confidence"])) for r in csv.DictReader(open(ball_path, newline=""))]
    cands = json.loads(cand_path.read_text()) if cand_path.exists() else None
    filled = predict_video_yolo.fill_gaps(raw, cands, info["height"])
    track = {r["Frame"]: (r["X"], r["Y"]) for r in filled if r["Visibility"]}
    n_seen = sum(r["Visibility"] == 1 for r in raw)
    report("ball", 1.0, f"balle vue sur {n_seen} frames, +{len(track) - n_seen} reconstituees")

    # 3. Rebonds et frappes (reglages par defaut de detect_bounces.py)
    report("bounces", 0.0, "analyse de la trajectoire")
    args = detect_bounces.make_parser().parse_args([])
    samples = detect_bounces.to_samples(track, args.dedup, args.outlier, info["height"])

    def place(bounces):
        """Position de chaque rebond sur le court (en metres) pour le jugement in / out."""
        placed, dropped = [], 0

        def locate(b):
            f = b["frame"]
            near = [g for g in range(f - 3, f + 4) if g in homs]  # terrain visible a ce moment-la ? (pas un gros plan)
            H = homs[min(near, key=lambda g: abs(g - f))] if near else None
            return H, (contact_coords(H, b["x"], b["y"]) if H else None)
        for b in bounces:
            H, c = locate(b)
            if b.get("first") and c is not None and signed_distance(c, True) > 0:
                # rebond "faute" alors qu'une cassure nette plus tot dans le meme echange tombe dans le
                # terrain : celle-ci etait le rebond, la suivante le second rebond (coup gagnant)
                H1, c1 = locate(b["first"])
                if c1 is not None and signed_distance(c1, True) <= 0:
                    b, H, c = b["first"], H1, c1
            f = b["frame"]
            if c is not None and signed_distance(c, True) > MAX_OUT:
                dropped += 1  # ex. lancer de balle au service : projetee au sol, la balle en l'air tombe tres loin
                continue
            err = position_error(H, c, info["height"]) if H else None
            if b.get("hidden") and err:
                err = [max(err[0], 0.4), max(err[1], 0.6)]  # rebond reconstitue : position bien moins sure
            placed.append(dict(frame=f, t=round(b.get("t", f), 2), x=round(b["x"], 1), y=round(b["y"], 1), court=c,
                               err=err, kick=round(b["kick"], 2), angle=round(b["angle"], 1),
                               relaxed=bool(b.get("relaxed")), **({"hidden": True} if b.get("hidden") else {})))
        return placed, dropped

    # Continuite du point (videos d'UN point) : une balle faute termine l'echange. Si l'echange continue
    # apres un "rebond faute", ce n'etait pas un rebond (souvent la frappe du joueur) : on l'impose comme
    # frappe et on relance la detection, qui trouve alors le vrai rebond de cet intervalle.
    forced, excluded = [], []
    for _ in range(8):
        bounces, shots = detect_bounces.find_bounces(samples, args, info["height"], info["fps"], racket_zone(players),
                                                     extra_shots=forced, exclude=excluded, at_player=serve_zone(players))
        bounces, shots = detect_bounces.serve_impacts(bounces, shots, track, info["width"], info["height"],
                                                      info["fps"], serve_zone(players))
        report("bounces", 1.0, f"{len(bounces)} rebonds, {len(shots)} frappes")
        report("judge", 0.0, "projection sur le court")
        bounces = sorted(bounces + hidden_bounces(bounces, shots, track, homs, players, info["fps"], info["height"]),
                         key=lambda b: b["frame"])
        out_bounces, dropped = place(bounces)
        shot_speeds(out_bounces, shots, track, homs, info, players)
        serve_boxes(out_bounces, players)
        bad = continuity_violations(out_bounces, shots) + fault_then_good(out_bounces)
        same = same_side_pairs(out_bounces, shots, racket_zone(players))
        if not bad and not same:
            break
        forced += sorted(set(bad) - set(forced))
        excluded += [f for f in same if f not in excluded]
    n_judged = sum(b["court"] is not None for b in out_bounces)
    report("judge", 1.0, f"{n_judged}/{len(out_bounces)} rebonds jugeables ({dropped} ecartes : trop loin du terrain)")

    result = dict(
        video=video.name, **info, version=PIPELINE_VERSION,
        created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 1),
        ball=[[f, round(x, 1), round(y, 1)] for f, (x, y) in sorted(track.items())],
        shots=shots, bounces=out_bounces,
        players={str(f): e for f, e in players["frames"].items()}, players_every=players["every"],
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
