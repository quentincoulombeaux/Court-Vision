"""Detecte les rebonds de la balle a partir du CSV de suivi (YOLO ou TrackNet) et les marque sur la video.

    .venv\\Scripts\\python.exe scripts\\detect_bounces.py video_test.mp4
    .venv\\Scripts\\python.exe scripts\\detect_bounces.py video_test.mp4 --csv runs\\tracknet\\video_test_ball.csv
    .venv\\Scripts\\python.exe scripts\\detect_bounces.py video_test.mp4 --amin 20 --amax 150 --debug

Par defaut le CSV lu est runs\\yolo_video\\<nom>_ball.csv (sortie de predict_video_yolo.py).

Methode (purement geometrique, aucun apprentissage) :
  1. Nettoyage : on fusionne les positions identiques consecutives (frames dupliquees de la video)
     et on retire les points isoles aberrants. Chaque position distincte = un "echantillon".
  2. Pour chaque echantillon k, on ajuste une droite sur les N echantillons AVANT (finissant en k)
     et une droite sur les N echantillons APRES (commencant en k).
  3. Les deux segments doivent etre RECTILIGNES (ecart a la droite <= --max-dev px ou --max-rel-dev x longueur),
     assez LONGS (>= --min-len px) et sans trou de plus de --max-frame-gap frames.
  4. Angle entre les deux directions : 0 = tout droit, 180 = demi-tour.
     --amin <= angle <= --amax  -> rebond ; au-dessus -> frappe ; en dessous -> trajectoire normale.
  5. Frappes = demi-tours verticaux francs de la balle (amplitude >= --shot-prom x hauteur d'image).
     Au tennis la balle rebondit au plus une fois entre deux frappes : dans chaque intervalle on garde
     la cassure la plus marquee (angle max), a au moins --min-gap frames des frappes.
  6. Point exact = intersection des deux droites (le rebond peut tomber entre deux frames).

Sortie : runs\\bounces\\<nom>_bounces.mp4 (trainee + rebonds marques en permanence) et <nom>_bounces.csv.
"""
import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_track(path):
    rows = list(csv.DictReader(open(path, newline="")))
    track = {}
    for r in rows:
        if int(r["Visibility"]) in (1, 2):  # 2 = position reconstituee (trou du suivi comble)
            track[int(r["Frame"])] = (float(r["X"]), float(r["Y"]))
    return track, len(rows)


def to_samples(track, dedup, outlier, frame_h=720):
    """Liste de (frame, x, y) sans doublons consecutifs ni points isoles aberrants."""
    k = frame_h / 720  # les seuils en pixels sont donnes pour du 720p
    samples = []
    for f in sorted(track):
        x, y = track[f]
        if samples and math.dist(samples[-1][1:], (x, y)) <= dedup:
            continue  # frame dupliquee : meme position, on garde la premiere
        samples.append((f, x, y))
    clean = []
    for i, s in enumerate(samples):
        if 0 < i < len(samples) - 1:
            prev, nxt = samples[i - 1], samples[i + 1]
            # loin de ses deux voisins alors que ceux-ci sont proches entre eux -> fausse detection
            if (math.dist(s[1:], prev[1:]) > outlier * k and math.dist(s[1:], nxt[1:]) > outlier * k
                    and math.dist(prev[1:], nxt[1:]) < outlier * k):
                continue
            # tres loin du milieu de ses voisins (plus que l'ecart entre eux) : le detecteur a vu autre chose
            mid = ((prev[1] + nxt[1]) / 2, (prev[2] + nxt[2]) / 2)
            if math.dist(s[1:], mid) > max(25 * k, 2 * math.dist(prev[1:], nxt[1:])):
                continue
        clean.append(s)
    return clean


def fit_line(pts):
    """Droite des moindres carres (point moyen, direction unitaire orientee du 1er au dernier point)
    et ecart max des points a cette droite."""
    pts = np.asarray(pts, dtype=float)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c)
    d = vt[0]
    if np.dot(pts[-1] - pts[0], d) < 0:
        d = -d
    dev = np.abs((pts - c) @ np.array([-d[1], d[0]])).max()
    return c, d, dev


def intersect(c1, d1, c2, d2):
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-6:
        return None
    t = ((c2[0] - c1[0]) * d2[1] - (c2[1] - c1[1]) * d2[0]) / den
    return c1 + t * d1


def sample_times(samples):
    """Temps de chaque echantillon, en 'pas' de la camera d'origine. Une video dont les frames sont
    dupliquees (ex. 18 img/s converties en 30) a des ecarts de 1 a 3 frames entre positions distinctes :
    on ramene ces ecarts a un pas regulier, sinon les vitesses sont fausses."""
    gaps = [q[0] - p[0] for p, q in zip(samples, samples[1:])]
    small = [g for g in gaps if g <= 3]
    r = max(1.0, sum(small) / len(small)) if small else 1.0
    t = [0.0]
    for g in gaps:
        t.append(t[-1] + max(1, round(g / r)))
    return t, r


def fit_motion(ts, pts, quad):
    """Mouvement p(t) = p0 + v.t (+ a.t^2/2 si quad : la gravite courbe la trajectoire) par moindres
    carres, avec t = 0 au point de cassure. Renvoie la position et la vitesse en t = 0, et l'ecart max."""
    ts = np.asarray(ts, dtype=float)
    T = np.c_[np.ones(len(ts)), ts, ts ** 2] if quad else np.c_[np.ones(len(ts)), ts]
    P = np.asarray(pts, dtype=float)
    coef, *_ = np.linalg.lstsq(T, P, rcond=None)
    dev = np.sqrt(((P - T @ coef) ** 2).sum(1)).max()
    return coef[0], coef[1], dev


def has_jump(ts, pts, tol):
    """Vrai si un pas du segment est bien plus rapide que les autres (une balle ne double pas sa vitesse en vol)."""
    steps = [math.dist(p, q) / max(1e-6, t1 - t0) for (t0, p), (t1, q) in zip(zip(ts, pts), zip(ts[1:], pts[1:]))]
    return max(steps) > 3 * float(np.median(steps)) + tol


def lowest_point(samples, ts, k):
    """Sommet (point le plus bas dans l'image) d'une descente-remontee autour de l'echantillon k."""
    for i in sorted(range(max(1, k - 2), min(len(samples) - 1, k + 3)), key=lambda j: abs(j - k)):
        (f0, x0, y0), (f1, x1, y1), (f2, x2, y2) = samples[i - 1], samples[i], samples[i + 1]
        # vraie descente PUIS vraie remontee (un palier, comme un rebond dans l'axe cote proche ou la balle
        # file a plat apres le contact, n'est pas un sommet : la parabole y placerait le contact trop bas)
        if not (y1 - y0 >= 1.0 and y1 - y2 >= 1.0):
            continue
        t0, t1, t2 = ts[i - 1], ts[i], ts[i + 1]
        c = np.linalg.solve(np.array([[t0 * t0, t0, 1], [t1 * t1, t1, 1], [t2 * t2, t2, 1]], float), [y0, y1, y2])
        if c[0] >= 0:
            continue  # pas un vrai sommet
        tv = float(np.clip(-c[1] / (2 * c[0]), t0, t2))
        if tv <= t1:
            u = (tv - t0) / max(1e-6, t1 - t0)
            xv, fa, fb = x0 + u * (x1 - x0), f0, f1
        else:
            u = (tv - t1) / max(1e-6, t2 - t1)
            xv, fa, fb = x1 + u * (x2 - x1), f1, f2
        return float(xv), float(np.polyval(c, tv)), float(fa + (fb - fa) * u)
    return None


def refine_contact(samples, ts, k, n=4):
    """Point de contact au sol : on cherche la meilleure coupure descente / remontee autour de k (la
    cassure n'est comptee que d'un cote), on ajuste une parabole de chaque cote (la gravite courbe la
    trajectoire : des droites croiseraient trop haut) et on prend l'instant ou les deux trajectoires
    verticales se rejoignent -- le contact tombe souvent entre deux images.
    Priorite a un point le plus bas net (descente puis remontee, cas du rebond cote eloigne) : le contact
    est alors le sommet de la parabole passant par ce point et ses deux voisins, et la position laterale
    est interpolee au meme instant. Plus robuste que la coupure entre deux paraboles, qui glissait d'un
    cran trop tard sur les videos a images dupliquees (balle mesuree apres avoir franchi une ligne)."""
    if lowest_point is not None:
        lp = lowest_point(samples, ts, k)
        if lp is not None:
            return lp
    best = None
    for s in (k - 1, k):
        if s - n + 1 < 0 or s + n >= len(samples):
            continue
        fits = []
        for idx in (range(s - n + 1, s + 1), range(s + 1, s + n + 1)):
            tt = np.array([ts[i] - ts[s] for i in idx])
            pp = np.array([samples[i][1:] for i in idx], float)
            T = np.c_[np.ones(len(tt)), tt, tt ** 2]
            coef, *_ = np.linalg.lstsq(T, pp, rcond=None)
            fits.append((coef, float(((pp - T @ coef) ** 2).sum())))
        err = fits[0][1] + fits[1][1]
        if best is None or err < best[0]:
            best = (err, s, fits[0][0], fits[1][0])
    if best is None:
        return None
    _, s, cb, ca = best
    gap = ts[s + 1] - ts[s]
    tt = np.linspace(-0.5, gap + 0.5, 61)
    T = np.c_[np.ones(len(tt)), tt, tt ** 2]
    pb, pa = T @ cb, T @ ca
    i = int(np.argmin(np.abs(pb[:, 1] - pa[:, 1])))
    p = (pb[i] + pa[i]) / 2
    frame = samples[s][0] + (samples[s + 1][0] - samples[s][0]) * tt[i] / max(1e-6, gap)
    return float(p[0]), float(p[1]), float(frame)  # frame fractionnaire : instant du contact


def find_kinks(samples, a):
    """Cassures candidates. On ajuste un mouvement rectiligne uniforme sur les N echantillons avant et
    les N apres, puis on compare les deux vitesses :
      - angle entre les directions (la balle tourne), et
      - 'coup vers le haut' : le sol renvoie toujours la balle vers le HAUT de l'image (vy diminue),
        alors que la gravite la tire vers le bas. Ce critere voit aussi les rebonds dans l'axe de la
        camera, ou la direction change a peine mais la vitesse verticale chute d'un coup."""
    n = a.n
    ts, r = sample_times(samples)
    kinks, downs = [], []
    for k in range(n - 1, len(samples) - n + 1):
        ib, ia = range(k - n + 1, k + 1), range(k, k + n)
        seg = samples[k - n + 1:k + n]
        if any(q[0] - p[0] > a.max_frame_gap for p, q in zip(seg, seg[1:])):
            continue
        tb, ta = [ts[i] - ts[k] for i in ib], [ts[i] - ts[k] for i in ia]
        pb, pa = [samples[i][1:] for i in ib], [samples[i][1:] for i in ia]
        lb, la = math.dist(pb[0], pb[-1]), math.dist(pa[0], pa[-1])
        if max(lb, la) < a.min_len:
            continue  # balle quasi immobile
        if has_jump(tb, pb, a.max_dev) or has_jump(ta, pa, a.max_dev):
            continue  # saut brusque : le suivi a change d'objet en cours de segment
        p0b, vb, devb = fit_motion(tb, pb, a.quad)
        p0a, va, deva = fit_motion(ta, pa, a.quad)
        if not a.quad:
            # rectitude jugee sur la geometrie seule : insensible au rythme irregulier des frames dupliquees
            devb, deva = fit_line(pb)[2], fit_line(pa)[2]
        # rectiligne : ecart a la droite <= max_dev px, ou max_rel_dev x longueur pour les longs segments
        if devb > max(a.max_dev, a.max_rel_dev * lb) or deva > max(a.max_dev, a.max_rel_dev * la):
            continue  # un des segments contient deja une cassure
        speed = max(np.linalg.norm(vb), np.linalg.norm(va), 1e-6)
        angle = math.degrees(math.acos(float(np.clip(np.dot(vb, va) / max(1e-9, np.linalg.norm(vb) * np.linalg.norm(va)), -1, 1))))
        kick = -(va[1] - vb[1]) / speed  # > 0 : la balle est renvoyee vers le haut de l'image
        if a.debug:
            print(f"  frame {samples[k][0]:4d} ({samples[k][1]:.0f},{samples[k][2]:.0f}) angle {angle:5.1f} kick {kick:+.2f}")
        if kick <= -a.shot_kick:
            downs.append((samples[k][0], kick))  # renvoyee brutalement vers le bas : seule une raquette le peut
        if kick < a.kick or angle > a.amax:
            continue
        # instant ou les deux mouvements (lineaires au voisinage) se rejoignent : marche aussi quand les
        # droites sont paralleles (balle dans l'axe de la camera)
        dv, d0 = va - vb, p0a - p0b
        tstar = float(np.clip(-np.dot(d0, dv) / max(1e-9, np.dot(dv, dv)), tb[-2], ta[1]))
        p = (p0b + vb * tstar + p0a + va * tstar) / 2
        contact = refine_contact(samples, ts, k) if a.refine else None
        t_contact = float(samples[k][0])
        if contact is not None and math.dist(contact[:2], samples[k][1:]) < 4 * a.max_dev:
            p, t_contact = contact[:2], contact[2]
        kinks.append(dict(frame=samples[k][0], t=t_contact, x=float(p[0]), y=float(p[1]), angle=angle, kick=kick,
                          vy_after=float(va[1]) / r, speed_after=float(np.linalg.norm(va)) / r,  # px / frame
                          ratio=float(np.linalg.norm(va) / max(1e-6, np.linalg.norm(vb)))))
    # une meme frappe donne plusieurs echantillons voisins : on garde le plus marque de chaque groupe
    shots = []
    for f, kk in downs:
        if shots and f - shots[-1][0] <= a.min_gap:
            if kk < shots[-1][1]:
                shots[-1] = (f, kk)
        else:
            shots.append((f, kk))
    return kinks, [f for f, _ in shots]


def speed_ratio(samples, ts, k, n, quad):
    """Vitesse apres / vitesse avant l'echantillon k (None si trop pres du debut ou de la fin)."""
    if k < n - 1 or k > len(samples) - n:
        return None
    ib, ia = range(k - n + 1, k + 1), range(k, k + n)
    _, vb, _ = fit_motion([ts[i] - ts[k] for i in ib], [samples[i][1:] for i in ib], quad)
    _, va, _ = fit_motion([ts[i] - ts[k] for i in ia], [samples[i][1:] for i in ia], quad)
    return np.linalg.norm(va) / max(1e-6, np.linalg.norm(vb))


def find_shots(samples, min_prom, a):
    """Frappes = demi-tours verticaux francs : extremums de Y dont la prominence >= min_prom px.
    (Camera TV : la balle monte dans l'image vers le joueur du fond, descend vers le joueur proche.)
    Cote joueur proche, un rebond haut dans l'axe de la camera fait aussi remonter la balle dans
    l'image : on le reconnait a ce que la balle RALENTIT (le rebond absorbe de l'energie) alors
    qu'une frappe l'accelere (vitesse apres / avant < --shot-ratio -> rebond, pas frappe)."""
    ys = [s[2] for s in samples]
    ts, _ = sample_times(samples)
    shots, slowed = [], []
    for sign in (1, -1):  # +1 : Y max (joueur proche), -1 : Y min (joueur du fond)
        v = [sign * y for y in ys]
        for k in range(1, len(v) - 1):
            if not (v[k] >= v[k - 1] and v[k] > v[k + 1]):
                continue
            bases = []
            for step in (-1, 1):  # descend de chaque cote jusqu'a un point plus haut que le pic
                j, lo = k + step, v[k]
                while 0 <= j < len(v) and v[j] <= v[k]:
                    lo = min(lo, v[j])
                    j += step
                if 0 <= j < len(v):
                    bases.append(lo)
                else:
                    bases.append(None)  # cote qui touche le debut/la fin de la video : non significatif
            known = [b for b in bases if b is not None] or [min(v)]
            if v[k] - max(known) < min_prom:
                continue
            if sign == 1:
                r = speed_ratio(samples, ts, k, a.n, a.quad)
                # aussi juste apres le point bas (2 positions) : si la frappe suit le rebond de pres,
                # la fenetre longue englobe deja la balle frappee et masque le ralentissement du rebond
                if a.short_ratio and 2 <= k <= len(samples) - 3:
                    vb = math.dist(samples[k - 2][1:], samples[k][1:]) / max(1e-6, ts[k] - ts[k - 2])
                    va = math.dist(samples[k][1:], samples[k + 2][1:]) / max(1e-6, ts[k + 2] - ts[k])
                    if va / max(1e-6, vb) < a.short_slow:  # ralentissement franc : rebond, meme si la frappe suit de pres
                        r = 0.0
                if r is not None and r < a.shot_ratio:
                    slowed.append(samples[k])  # la balle ralentit : c'est un rebond, pas une frappe
                    continue
            shots.append((samples[k][0], sign, ys[k]))
    shots.sort()
    # Les frappes alternent : joueur proche (Y max) puis joueur du fond (Y min). Deux du meme type
    # d'affilee : on ne garde que la plus extreme (Y max : la plus basse ; l'autre est le rebond d'un
    # lob cote eloigne. Y min : la plus haute ; l'autre est typiquement le lancer du service).
    kept = []
    for s in shots:
        if kept and kept[-1][1] == s[1]:
            if s[1] * s[2] > kept[-1][1] * kept[-1][2]:
                kept[-1] = s
        else:
            kept.append(s)
    return [s[0] for s in kept], slowed


def find_bounces(samples, a, frame_h, fps=30, in_racket=None, extra_shots=(), exclude=(), at_player=None):
    """in_racket(frame, x, y) -> bool (optionnel) : la balle est-elle dans la zone de frappe d'un joueur ?
    Fourni par le pipeline a partir de la detection des joueurs."""
    # les seuils en pixels sont donnes pour une video 720p : on les met a l'echelle
    a = argparse.Namespace(**vars(a))
    a.max_dev *= frame_h / 720
    a.min_len *= frame_h / 720
    kinks, down_shots = find_kinks(samples, a)
    # Frappe du joueur proche : la balle repart vers le HAUT de l'image, vite et en accelerant. Un rebond
    # la renvoie aussi vers le haut, mais nettement moins vite. Ces cassures sont des frappes, pas des rebonds.
    hits = [k for k in kinks if k["vy_after"] < 0 and k["ratio"] > a.hit_ratio
            and k["speed_after"] * fps / frame_h > a.hit_speed]
    if in_racket:
        # demi-tour de la balle dans la zone de frappe d'un joueur : c'est sa frappe. Mesure sur les
        # annotations : 45 % des frappes cassent la trajectoire a plus de 150 deg, contre 7 % des rebonds.
        # On teste la position OBSERVEE de la balle (le point de contact recalcule peut deborder de la zone).
        # Et une frappe ACCELERE la balle : dans les 0,2 s qui suivent, elle repart nettement plus vite.
        # Un rebond haut a cote du joueur peut aussi faire demi-tour dans l'image, mais la balle y ralentit.
        seen = {f: (x, y) for f, x, y in samples}

        def accelerates(k):
            return any(0 <= kk["frame"] - k["frame"] <= 0.2 * fps and kk["ratio"] >= 1.5 for kk in kinks)
        hits += [k for k in kinks if k not in hits and k["angle"] >= a.racket_angle and accelerates(k)
                 and in_racket(k["frame"], *seen.get(k["frame"], (k["x"], k["y"])))]
        # La balle repart vers le HAUT de l'image en accelerant, depuis la zone d'un joueur : sa frappe,
        # meme sans demi-tour (ex. balle qui filait a plat apres le rebond puis renvoyee).
        hits += [k for k in kinks if k not in hits and k["vy_after"] < 0 and k["ratio"] >= a.depart_ratio
                 and in_racket(k["frame"], *seen.get(k["frame"], (k["x"], k["y"])))]
    kinks = [k for k in kinks if k not in hits]
    shots, slowed = find_shots(samples, a.shot_prom * frame_h, a)
    if exclude:  # candidats ecartes par les regles du point (ex. deux rebonds d'affilee du meme cote)
        kinks = [k for k in kinks if all(abs(k["frame"] - e) > 2 for e in exclude)]
        slowed = [sl for sl in slowed if all(abs(sl[0] - e) > 2 for e in exclude)]
    if extra_shots:  # frappes imposees (ex. "faute" suivie de la suite de l'echange : c'etait une frappe)
        kinks = [k for k in kinks if all(abs(k["frame"] - e) > 2 for e in extra_shots)]
        slowed = [sl for sl in slowed if all(abs(sl[0] - e) > 2 for e in extra_shots)]
        shots = sorted(set(shots) | set(extra_shots))
    if in_racket:  # un "rebond haut" dans la raquette d'un joueur est en fait sa frappe
        # ... seulement s'il y a aussi demi-tour : un rebond juste devant le joueur proche apparait
        # au-dessus de ses pieds dans l'image, sans pour autant etre dans sa raquette
        def u_turn(f):  # demi-tour ET la balle repart plus vite tout de suite (sinon : rebond, frappe juste apres)
            return any(abs(k["frame"] - f) <= 2 and k["angle"] >= a.racket_angle for k in kinks + hits) and                 any(0 <= k["frame"] - f <= 0.1 * fps and k["ratio"] >= 1.5 for k in kinks + hits)
        racket = [sl for sl in slowed if in_racket(*sl) and u_turn(sl[0])]
        slowed = [sl for sl in slowed if sl not in racket]
        shots = sorted(shots + [sl[0] for sl in racket if all(abs(sl[0] - s) > a.min_gap for s in shots)])
    # Frappe cachee dans un trou du suivi : la balle touche le sol (point bas, cassure franche) puis le
    # joueur la reprend pendant que le suivi la perd. Juste apres le trou elle file bien plus vite qu'avant :
    # c'est la que la raquette l'a touchee. Le point bas etait donc le rebond, pas la frappe.
    if a.hidden_hit:
        idx = {q[0]: i for i, q in enumerate(samples)}
        moved = []
        for sh in list(shots):
            i = idx.get(sh)
            if i is None or i < 1 or i + 2 >= len(samples) or sh in extra_shots:
                continue
            (f0, x0, y0), (f1, x1, y1), (f2, x2, y2), (f3, x3, y3) = samples[i - 1:i + 3]
            v_in = math.dist((x0, y0), (x1, y1)) / (f1 - f0)
            v_out = math.dist((x2, y2), (x3, y3)) / (f3 - f2)
            low = [k for k in kinks if abs(k["frame"] - sh) <= 1 and k["kick"] >= 1.0]
            # pas de rebond net juste avant : sinon la balle a deja rebondi et le point bas est la reprise
            before = [k for k in kinks if sh - 0.5 * fps <= k["frame"] < sh - 1 and k["kick"] >= 0.8]
            if f2 - f1 >= 3 and v_out >= a.hidden_ratio * v_in and y1 >= y0 and low and not before:
                k = max(low, key=lambda k: k["kick"])
                slowed.append((k["frame"], k["x"], k["y"]))
                moved.append((sh, (f1 + f2) // 2 + 1))
        for sh, new_f in moved:
            shots = sorted([x for x in shots if x != sh] + [new_f])
        slowed.sort()
    # Frappes du joueur du fond sans grand demi-tour (ex. apres un rebond court derriere le filet) :
    # reperees par un fort coup vers le BAS de la vitesse.
    shots = sorted(shots + [f for f in down_shots if all(abs(f - s) > a.min_gap for s in shots)])
    for h in hits:
        if all(abs(h["frame"] - s) > a.min_gap for s in shots):
            shots = sorted(shots + [h["frame"]])
    # Un joueur ne frappe pas deux fois de suite sans que la balle ait rebondi ou traverse : deux "frappes"
    # rapprochees autour du MEME joueur, sans rebond haut entre elles -> seule la derniere est sa frappe (les
    # precedentes : sommet d'un lob, balle qui ralentit en arrivant sur lui...). Le rebond devant lui,
    # cache jusque-la par la fausse frappe et son delai minimal, peut alors etre trouve.
    if at_player:
        def who(f):
            q = min(samples, key=lambda q: abs(q[0] - f))
            return at_player(f, q[1], q[2])
        drop = set()
        for s0, s1 in zip(shots, shots[1:]):
            if s1 - s0 <= a.same_player_gap * fps and s0 not in extra_shots and who(s0) and who(s0) == who(s1)                     and not any(s0 < sl[0] < s1 for sl in slowed):
                drop.add(s0)
        shots = [x for x in shots if x not in drop]
    if a.debug:
        print("  frappes (frames) :", shots, " rebonds hauts :", [sl[0] for sl in slowed])
    # Au tennis la balle rebondit au plus une fois entre deux frappes : on garde la cassure la plus nette.
    # Un rebond haut (demi-tour ou la balle ralentit) est retenu d'office et sert de frontiere : la frappe
    # qui le suit ne fait souvent pas de demi-tour visible (la balle remontait deja).
    def nearest_pos(f):
        return min(samples, key=lambda q: abs(q[0] - f))[1:]

    def earliest_bounce(s0, t0):
        """Apres une frappe : pas de rebond avant after_shot, ni avant after_racket si la frappe a eu lieu
        dans la raquette d'un joueur detecte (la balle doit traverser le court)."""
        start = s0 + (max(a.min_gap, a.after_shot * fps) if t0 == "frappe" else a.min_gap)
        if in_racket and t0 == "frappe" and s0 > 0 and in_racket(s0, *nearest_pos(s0)):
            start = max(start, s0 + a.after_racket * fps)
        return start

    bounces = []
    for f, x, y in slowed:
        near = [k for k in kinks if abs(k["frame"] - f) <= 2]
        bounces.append(min(near, key=lambda k: abs(k["frame"] - f)) if near
                       else dict(frame=f, x=x, y=y, angle=0.0, kick=0.0))
    events = sorted([(f, "frappe") for f in shots] + [(b["frame"], "rebond") for b in bounces])
    bounds = [(-10**9, "frappe")] + events + [(10**9, "frappe")]
    for (s0, t0), (s1, t1) in zip(bounds, bounds[1:]):
        if t0 == "rebond" and t1 == "rebond":
            continue  # deux rebonds hauts : rien a chercher entre eux
        # apres une frappe, la balle doit traverser le court : pas de rebond avant ~0,25 s (la balle qui
        # accelere encore juste apres la frappe n'est pas un rebond) ; avant la frappe suivante, l'ecart
        # rebond -> frappe peut etre tres court
        inside = [k for k in kinks if earliest_bounce(s0, t0) <= k["frame"] <= s1 - a.min_gap]
        if inside:
            best = dict(max(inside, key=lambda k: k["kick"]))
            # cassure nette bien plus tot : candidate "premier rebond", gardee pour le pipeline (si la plus
            # forte tombe faute et celle-ci dans le terrain, la plus forte etait le second rebond d'un coup
            # gagnant). Le choix se fait la-bas, une fois les rebonds places sur le court.
            early = [k for k in inside if k["frame"] <= best["frame"] - a.first_gap * fps
                     and k["kick"] >= max(a.first_kick, 0.7 * best["kick"])]
            if early:
                best["first"] = min(early, key=lambda k: k["frame"])
            bounces.append(best)
    # Deuxieme passe, plus souple : entre deux frappes il faut un rebond (sauf volee). Si un intervalle
    # n'en a aucun -- typiquement rebond et frappe tres proches, ou video a images dupliquees qui ne laisse
    # que 2-3 positions entre les deux -- on y cherche la cassure la plus nette avec des segments courts.
    if a.relaxed_n:
        r = argparse.Namespace(**vars(a))
        r.n = a.relaxed_n
        loose = None
        cuts = [-10**9] + shots + [10**9]
        for s0, s1 in zip(cuts, cuts[1:]):
            if s1 - s0 < a.relaxed_min_interval * fps or s0 < 0 or s1 > 10**8:
                continue
            if any(s0 < b["frame"] < s1 for b in bounces):
                continue
            if loose is None:
                loose, _ = find_kinks(samples, r)
            inside = [k for k in loose if earliest_bounce(s0, "frappe") <= k["frame"] <= s1 - 2 and k not in hits
                      and all(abs(k["frame"] - e) > 2 for e in exclude)
                      and not (in_racket and k["angle"] >= a.racket_angle
                               and in_racket(k["frame"], *nearest_pos(k["frame"])))]
            if inside:
                best = dict(max(inside, key=lambda k: k["kick"]))
                best["relaxed"] = True
                bounces.append(best)
    return sorted(bounces, key=lambda b: b["frame"]), shots


def is_toss(track, f, frame_w, frame_h, fps=30):
    """Juste avant la frame f, un lancer de service : la balle monte, s'arrete et redescend doucement,
    presque a la verticale (peu de deplacement horizontal)."""
    if f - int(0.7 * fps) < 0:
        return False  # pas assez d'images avant (la video commence en plein echange)
    win = [g for g in range(f - int(0.7 * fps), f) if g in track]
    if len(win) < 6:
        return False
    xs = [track[g][0] for g in win]
    ys = [track[g][1] for g in win]
    i = min(range(len(ys)), key=lambda j: ys[j])     # sommet du lancer (y minimal)
    descent = sum(math.dist(track[g], track[h]) for g, h in zip(win[i:], win[i + 1:]))
    return (max(xs) - min(xs) < 0.06 * frame_w and 0 < i < len(ys) - 2
            and ys[i] < ys[0] - 0.01 * frame_h and ys[-1] > ys[i] + 0.01 * frame_h
            and descent < 0.12 * frame_h)


def serve_impacts(bounces, shots, track, frame_w, frame_h, fps=30, at_player=None):
    """Impact du service pris pour un rebond. Au service, la balle est lancee : elle monte, s'arrete et
    redescend presque a la verticale, puis la raquette la renvoie -- une cassure qui ressemble a un
    rebond (et, projetee au sol, la balle a 3 m de haut tombe souvent dans le court). Un rebond, lui, est
    precede d'une balle qui arrive vite de l'autre cote. Regle : premier evenement du point (aucun rebond
    dans les 2,5 s avant), et juste avant un sommet de lancer (balle qui monte puis redescend) avec tres
    peu de deplacement horizontal -> c'est la frappe de service : retiree des rebonds, ajoutee aux frappes.
    (Retire aussi les rebonds de preparation, quand le serveur fait rebondir la balle avant de servir.)
    at_player(frame, x, y) -> bool (optionnel) : si fourni, la balle doit etre au-dessus ou autour d'un
    joueur, ce qui evite de retirer un vrai rebond de jeu loin des joueurs."""
    keep, prev = [], -10**9
    shots = list(shots)
    for b in sorted(bounces, key=lambda b: b["frame"]):
        f = b["frame"]
        first = f - prev > 2.5 * fps
        prev = f
        if first and is_toss(track, f, frame_w, frame_h, fps) and (at_player is None or at_player(f, b["x"], b["y"])):
            shots.append(f)
            continue
        keep.append(b)
    return keep, sorted(shots)


def make_parser():
    """Reglages de la detection (partages avec eval_bounces.py)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="echantillons par segment (avant et apres)")
    ap.add_argument("--kick", type=float, default=0.4, help="coup vers le haut min (variation de vy / vitesse)")
    ap.add_argument("--amax", type=float, default=180, help="angle max (deg) entre les directions avant/apres")
    ap.add_argument("--quad", type=int, default=0, help="1 = segments paraboliques (gravite), 0 = droites")
    ap.add_argument("--min-len", type=float, default=15, help="longueur min d'un segment (px, pour une video 720p)")
    ap.add_argument("--max-dev", type=float, default=8, help="ecart max a la droite (px, pour une video 720p) : segment rectiligne")
    ap.add_argument("--max-rel-dev", type=float, default=0.08, help="idem en fraction de la longueur du segment")
    ap.add_argument("--shot-prom", type=float, default=0.16, help="amplitude min d'un demi-tour vertical = frappe (fraction de la hauteur)")
    ap.add_argument("--shot-ratio", type=float, default=0.8, help="cote joueur proche : vitesse apres/avant sous ce seuil = rebond, pas frappe")
    ap.add_argument("--shot-kick", type=float, default=0.8, help="coup vers le bas min (variation de vy / vitesse) = frappe")
    ap.add_argument("--hit-speed", type=float, default=0.8, help="vitesse apres (hauteurs d'image / s) au-dela = frappe si la balle monte et accelere")
    ap.add_argument("--hit-ratio", type=float, default=1.2, help="acceleration min (vitesse apres / avant) pour cette regle")
    ap.add_argument("--refine", type=int, default=1, help="1 = point de contact recalcule (coupure descente/remontee)")
    ap.add_argument("--racket-angle", type=float, default=155, help="demi-tour (deg) dans la zone d'un joueur = frappe")
    ap.add_argument("--relaxed-n", type=int, default=3, help="2e passe : segments courts dans les intervalles sans rebond (0 = off)")
    ap.add_argument("--relaxed-min-interval", type=float, default=0.7, help="intervalle min (s) entre frappes pour la 2e passe")
    ap.add_argument("--short-ratio", type=int, default=1, help="1 = ralentissement aussi mesure sur 2 positions apres un point bas")
    ap.add_argument("--short-slow", type=float, default=0.5, help="ralentissement franc juste apres un point bas = rebond")
    ap.add_argument("--after-racket", type=float, default=0.35, help="s min entre une frappe dans la raquette d'un joueur et le rebond")
    ap.add_argument("--after-shot", type=float, default=0.15, help="s min entre une frappe et le rebond suivant")
    ap.add_argument("--depart-ratio", type=float, default=1.5, help="depart accelere (vitesse apres/avant) depuis un joueur = frappe")
    ap.add_argument("--hidden-hit", type=int, default=1, help="1 = frappe cachee dans un trou du suivi juste apres un point bas")
    ap.add_argument("--hidden-ratio", type=float, default=1.6, help="acceleration (vitesse apres le trou / avant) pour cette regle")
    ap.add_argument("--first-gap", type=float, default=0.3, help="s : une cassure nette au moins aussi tot avant la plus forte = le vrai rebond")
    ap.add_argument("--first-kick", type=float, default=0.8, help="kick min de cette cassure plus precoce")
    ap.add_argument("--same-player-gap", type=float, default=1.0, help="s : deux frappes du meme joueur sans rebond -> seule la derniere compte")
    ap.add_argument("--min-gap", type=int, default=4, help="frames min entre une frappe et un rebond")
    ap.add_argument("--max-frame-gap", type=int, default=6, help="trou max (frames) dans un segment")
    ap.add_argument("--dedup", type=float, default=2, help="px : positions plus proches = frame dupliquee")
    ap.add_argument("--outlier", type=float, default=120, help="px : saut d'un point isole juge aberrant")
    ap.add_argument("--trail", type=int, default=10, help="longueur de la trainee affichee")
    ap.add_argument("--debug", action="store_true", help="affiche l'angle mesure a chaque echantillon")
    return ap


def main():
    ap = make_parser()
    ap.add_argument("video")
    ap.add_argument("--csv", help="CSV de suivi (defaut : runs\\yolo_video\\<nom>_ball.csv)")
    a = ap.parse_args()

    stem = Path(a.video).stem
    csv_path = Path(a.csv) if a.csv else ROOT / "runs" / "yolo_video" / f"{stem}_ball.csv"
    track, n_rows = load_track(csv_path)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    samples = to_samples(track, a.dedup, a.outlier, vh)
    print(f"{csv_path.name} : {len(track)} detections -> {len(samples)} positions distinctes")
    bounces, shots = find_bounces(samples, a, vh, fps)
    print(f"{len(shots)} frappe(s) reperee(s) aux frames {shots}")
    for b in bounces:
        print(f"Rebond frame {b['frame']:4d} en ({b['x']:.0f}, {b['y']:.0f})  angle {b['angle']:3.0f} deg  kick {b['kick']:.2f}")
    print(f"{len(bounces)} rebond(s)")

    out_dir = ROOT / "runs" / "bounces"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{stem}_bounces.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "x", "y", "angle", "kick"])
        w.writeheader()
        w.writerows(dict(frame=b["frame"], x=round(b["x"]), y=round(b["y"]), angle=round(b["angle"], 1),
                         kick=round(b["kick"], 2)) for b in bounces)

    writer = cv2.VideoWriter(str(out_dir / f"{stem}_bounces.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))
    trail = []
    for fid in range(n_rows):
        ok, frame = cap.read()
        if not ok:
            break
        p = track.get(fid)
        trail.append((int(p[0]), int(p[1])) if p else None)
        del trail[:-a.trail]
        pts = [t for t in trail if t is not None]
        for p1, p2 in zip(pts, pts[1:]):
            cv2.line(frame, p1, p2, (0, 200, 255), 2)
        if trail[-1] is not None:
            cv2.circle(frame, trail[-1], 12, (0, 0, 255), 2)
        for b in bounces:  # chaque rebond reste affiche jusqu'a la fin
            if b["frame"] > fid:
                break
            c = (int(round(b["x"])), int(round(b["y"])))
            if fid - b["frame"] < int(fps):  # halo pendant 1 s pour voir le rebond arriver
                cv2.circle(frame, c, 22, (255, 255, 0), 2)
            cv2.circle(frame, c, 7, (0, 0, 0), -1)
            cv2.circle(frame, c, 5, (0, 255, 0), -1)
        writer.write(frame)
    cap.release()
    writer.release()
    print(f"-> {out_dir / (stem + '_bounces.mp4')}")


if __name__ == "__main__":
    main()
