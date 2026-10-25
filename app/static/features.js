// CourtVision -- fonctions avancees : joueurs, carte de chaleur, relecture de l'impact en 3D, rapport.
// Charge apres app.js (meme portee globale : S, t, verdict, project, COLORS...).

// ---------- joueurs ----------
const PLAYER_COLORS = { far: "#7dd3fc", near: "#c4b5fd" };
function playersAt(f) {
  const P = S.analysis && S.analysis.players;
  if (!P) return {};
  for (let d = 0; d <= 3; d++) {
    const e = P[String(f - d)] || P[String(f + d)];
    if (e) return e;
  }
  return {};
}
function drawPlayers(f, H, px) {
  const ps = playersAt(f);
  for (const side of ["far", "near"]) {
    const p = ps[side];
    if (!p) continue;
    const [, , , , X, Y] = p, col = PLAYER_COLORS[side];
    ctx.save();
    if (H) {                                               // anneau au sol, dans la perspective du court
      ctx.strokeStyle = col; ctx.lineWidth = 2 * px; ctx.globalAlpha = 0.9;
      ctx.shadowColor = col; ctx.shadowBlur = 8 * px;
      pathPoly(groundPoly(H, [X, Y], 0.55)); ctx.stroke();
    }
    ctx.restore();
  }
}
let mmPlayersGroup = null;
function updateMinimapPlayers(f) {               // positions en direct sur la mini-carte (pas de redessin complet)
  if (!mmPlayersGroup || !mmPlayersGroup.isConnected) {
    mmPlayersGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    mm.appendChild(mmPlayersGroup);
  }
  const ps = S.layers.players && S.mmMode !== "heat" ? playersAt(f) : {};
  for (const side of ["far", "near"]) {
    let c = mmPlayersGroup.querySelector(`.pl-${side}`);
    const p = ps[side];
    if (!p) { if (c) c.remove(); continue; }
    if (!c) {
      c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      c.setAttribute("class", `player-dot pl-${side}`); c.setAttribute("r", 0.42);
      mmPlayersGroup.appendChild(c);
    }
    c.setAttribute("cx", Math.min(DW + 2.8, Math.max(-2.8, p[4])));
    c.setAttribute("cy", Math.min(L + 2.8, Math.max(-2.8, p[5])));
  }
}

// ---------- carte de chaleur des rebonds ----------
function heatmapURL(bounces) {
  const x0 = -3, y0 = -3, W = DW + 6, Hh = L + 6, cell = 0.1, nx = Math.round(W / cell), ny = Math.round(Hh / cell);
  const grid = new Float32Array(nx * ny), sig = 0.9, R = Math.ceil((3 * sig) / cell);
  for (const b of bounces) {
    const cx = (b.court[0] - x0) / cell, cy = (b.court[1] - y0) / cell;
    for (let j = Math.max(0, Math.floor(cy - R)); j < Math.min(ny, cy + R); j++)
      for (let i = Math.max(0, Math.floor(cx - R)); i < Math.min(nx, cx + R); i++) {
        const d2 = ((i - cx) ** 2 + (j - cy) ** 2) * cell * cell;
        grid[j * nx + i] += Math.exp(-d2 / (2 * sig * sig));
      }
  }
  let max = 0; for (const v of grid) max = Math.max(max, v);
  const cv = document.createElement("canvas"); cv.width = nx; cv.height = ny;
  const g = cv.getContext("2d"), img = g.createImageData(nx, ny);
  // rampe : transparent -> jaune balle -> orange -> rouge
  const stops = [[0, [217, 242, 75, 0]], [0.25, [217, 242, 75, 110]], [0.55, [245, 158, 11, 175]], [1, [244, 63, 94, 220]]];
  for (let k = 0; k < grid.length; k++) {
    const v = max ? grid[k] / max : 0;
    let a = stops[0], c = stops[stops.length - 1];
    for (let s = 1; s < stops.length; s++) if (v <= stops[s][0]) { a = stops[s - 1]; c = stops[s]; break; }
    const u = (v - a[0]) / Math.max(1e-6, c[0] - a[0]);
    for (let ch = 0; ch < 4; ch++) img.data[4 * k + ch] = a[1][ch] + (c[1][ch] - a[1][ch]) * u;
  }
  g.putImageData(img, 0, 0);
  return cv.toDataURL();
}
function heatBounces() {
  return S.analysis ? S.analysis.bounces.filter((b) => b.court) : [];
}
function drawHeat(group) {
  const list = heatBounces();
  if (!list.length) return;
  const im = svgEl("image", { x: -3, y: -3, width: DW + 6, height: L + 6, preserveAspectRatio: "none", class: "heat" }, group);
  im.setAttributeNS("http://www.w3.org/1999/xlink", "href", heatmapURL(list));
  im.setAttribute("href", im.getAttributeNS("http://www.w3.org/1999/xlink", "href"));
}

// ---------- relecture de l'impact (vue 3D puis zoom au sol) ----------
const RP = { open: false, b: null, start: 0, raf: 0 };
function replayData(b) {
  const sp = b.speed;
  let pts, ib;
  if (sp && sp.traj) {
    // la trajectoire ajustee finit a quelques dizaines de cm du rebond mesure : on la recale
    // progressivement (0 a la frappe -> 100 % au rebond) pour que l'impact tombe pile sur la trace
    ib = sp.traj_bounce;
    const end = sp.traj[ib], dx = b.court[0] - end[0], dy = b.court[1] - end[1], dz = -end[2];
    pts = sp.traj.map((p, i) => { const w = Math.min(1, i / ib); return [p[0] + dx * w, p[1] + dy * w, Math.max(0, p[2] + dz * w)]; });
  }
  else {                                     // pas de trajectoire 3D : arc plausible depuis la ligne de fond opposee
    const hx = sp ? sp.hit_court[0] : CX, hy = sp ? sp.hit_court[1] : (b.court[1] > L / 2 ? -1.5 : L + 1.5);
    pts = [];
    for (let i = 0; i <= 23; i++) {
      const u = i / 23;
      pts.push([hx + (b.court[0] - hx) * u, hy + (b.court[1] - hy) * u, (1 - u) * 1.0 + 4 * 0.6 * u * (1 - u)]);
    }
    ib = 23;
    const dx = b.court[0] - hx, dy = b.court[1] - hy, n = Math.hypot(dx, dy) || 1;
    for (let i = 1; i <= 6; i++) { const s = i * 0.6; pts.push([b.court[0] + (dx / n) * s, b.court[1] + (dy / n) * s, 0.9 * Math.sin((i / 6) * Math.PI * 0.8)]); }
  }
  return { pts, ib };
}
// zone qui juge ce rebond : carre de service, ou court simple / double
function judgeRect(b) {
  if (b.box) return b.box;
  return S.doubles ? [0, DW, 0, L] : [AL, DW - AL, 0, L];
}
function nearestBoundary(c, r) {          // point le plus proche sur le bord du rectangle
  const [x0, x1, y0, y1] = r;
  const inside = c[0] >= x0 && c[0] <= x1 && c[1] >= y0 && c[1] <= y1;
  if (!inside) return [Math.min(x1, Math.max(x0, c[0])), Math.min(y1, Math.max(y0, c[1]))];
  const d = [[c[0] - x0, [x0, c[1]]], [x1 - c[0], [x1, c[1]]], [c[1] - y0, [c[0], y0]], [y1 - c[1], [c[0], y1]]];
  return d.sort((a, b) => a[0] - b[0])[0][1];
}
// camera virtuelle (perspective) avec decoupage au plan proche : un polygone qui passe derriere la camera
// est coupe proprement au lieu d'etre abandonne (sinon la surface du court disparait en vue rapprochee)
function camera(Cp, T, upHint, fov, w, h) {
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const nrm = (a) => { const n = Math.hypot(...a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };
  const f = nrm(sub(T, Cp));
  let r = cross(f, upHint);
  if (Math.hypot(...r) < 1e-3) r = cross(f, [0, 1, 0]);
  r = nrm(r);
  const u = cross(r, f), F = (0.5 * h) / Math.tan((fov * Math.PI) / 360), NEAR = 0.1;
  const toCam = (P) => { const d = sub(P, Cp); return [dot(d, r), dot(d, u), dot(d, f)]; };
  // le repere du court (x a droite, y vers la camera TV, z en haut) est indirect : d'ou le signe moins
  // sur l'axe horizontal (sans lui, la scene s'affichait en miroir)
  const toScr = (c) => [w / 2 - (F * c[0]) / c[2], h / 2 - (F * c[1]) / c[2], F / c[2]];
  const cut = (a, b) => { const k = (NEAR - a[2]) / (b[2] - a[2]); return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, NEAR]; };
  return {
    proj: (P) => { const c = toCam(P); return c[2] < NEAR ? null : toScr(c); },
    poly: (P3) => {                       // Sutherland-Hodgman sur le plan proche
      const C = P3.map(toCam), out = [];
      for (let i = 0; i < C.length; i++) {
        const a = C[i], b = C[(i + 1) % C.length], ia = a[2] >= NEAR, ib = b[2] >= NEAR;
        if (ia) out.push(a);
        if (ia !== ib) out.push(cut(a, b));
      }
      return out.length >= 3 ? out.map(toScr) : null;
    },
    seg: (A, B) => {
      let a = toCam(A), b = toCam(B);
      if (a[2] < NEAR && b[2] < NEAR) return null;
      if (a[2] < NEAR) a = cut(a, b); else if (b[2] < NEAR) b = cut(b, a);
      return [toScr(a), toScr(b)];
    },
  };
}
function openReplay(b) {
  if (!b || !b.court) return;
  video.pause();
  RP.open = true; RP.b = b; RP.start = performance.now();
  $("replay").classList.remove("hidden");
  const v = verdict(b);
  $("replayVerdict").textContent = label(v); $("replayVerdict").className = `rp-verdict ${v}`;
  $("replayInfo").textContent = `${fmtTime(b.frame / fps())} · ${describe(b)}${b.speed ? ` · ${b.speed.kmh} km/h` : ""}`;
  cancelAnimationFrame(RP.raf);
  RP.raf = requestAnimationFrame(drawReplay);
}
function closeReplay() { RP.open = false; cancelAnimationFrame(RP.raf); $("replay").classList.add("hidden"); }
function drawReplay(now) {
  if (!RP.open) return;
  const cv = $("replayCanvas"), dpr = window.devicePixelRatio || 1, rect = cv.getBoundingClientRect();
  cv.width = Math.round(rect.width * dpr); cv.height = Math.round(rect.height * dpr);
  const g = cv.getContext("2d"), w = cv.width, h = cv.height, s = dpr;
  const b = RP.b, { pts, ib } = replayData(b), v = verdict(b), col = COLORS[v];
  const T = (now - RP.start) / 1000;
  const FLY = 2.4, MOVE = 1.4;                     // vol au ralenti, puis descente de la camera
  const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
  // position de la balle le long de la trajectoire (temps uniforme entre les echantillons)
  const uFly = Math.min(1, T / FLY), idxF = uFly * (pts.length - 1);
  const ballIdx = Math.min(pts.length - 1, idxF);
  // cameras : derriere le frappeur, puis a la verticale du rebond
  const hit = pts[0], bc = [b.court[0], b.court[1], 0];
  let dir = [bc[0] - hit[0], bc[1] - hit[1]]; const dn = Math.hypot(...dir) || 1; dir = [dir[0] / dn, dir[1] / dn];
  // vol filme de biais et en hauteur (l'arc de la balle est visible), puis vue plongeante sur la trace,
  // orientee dans le sens du coup, avec 3 a 4 m de court autour
  const perp = [-dir[1], dir[0]];
  const camA = [hit[0] - dir[0] * 5 + perp[0] * 8, hit[1] - dir[1] * 5 + perp[1] * 8, 4.5];
  const tgtA = [(hit[0] + bc[0]) / 2, (hit[1] + bc[1]) / 2, 0.8];
  const camB = [bc[0] - dir[0] * 1.2 + perp[0] * 0.8, bc[1] - dir[1] * 1.2 + perp[1] * 0.8, 4.2], tgtB = bc;
  const m = ease(Math.min(1, Math.max(0, (T - FLY + 0.3) / MOVE)));
  const lerp = (a, c) => a.map((x, i) => x + (c[i] - x) * m);
  const cam = camera(lerp(camA, camB), lerp(tgtA, tgtB), lerp([0, 0, 1], [dir[0], dir[1], 0]), 55 - 7 * m, w, h);
  const proj = cam.proj;

  // fond et sol
  const bg = g.createLinearGradient(0, 0, 0, h); bg.addColorStop(0, "#0b0f15"); bg.addColorStop(1, "#111a24");
  g.fillStyle = bg; g.fillRect(0, 0, w, h);
  const poly = (P3, fill, stroke, lw) => {
    const P = cam.poly(P3); if (!P) return;
    g.beginPath(); P.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]))); g.closePath();
    if (fill) { g.fillStyle = fill; g.fill(); }
    if (stroke) { g.strokeStyle = stroke; g.lineWidth = lw * s; g.stroke(); }
  };
  const quad = (xa, ya, xb, yb, z = 0) => [[xa, ya, z], [xb, ya, z], [xb, yb, z], [xa, yb, z]];
  poly(quad(-6, -8, DW + 6, L + 8), "#0f1d2a");
  poly(quad(0, 0, DW, L), "#1b2c40");
  const R = judgeRect(b);                                        // zone jugee, legerement eclairee
  poly(quad(R[0], R[2], R[1], R[3]), "rgba(217,242,75,0.05)");
  for (const [a, c] of COURT_LINES) {                            // lignes avec leur vraie largeur
    const wl = (a[1] === c[1] && (a[1] === 0 || a[1] === L)) ? 0.10 : 0.05;
    const dx = c[0] - a[0], dy = c[1] - a[1], n = Math.hypot(dx, dy), ux = dx / n, uy = dy / n, px = -uy * wl / 2, py = ux * wl / 2;
    const e = wl / 2;
    poly([[a[0] - ux * e + px, a[1] - uy * e + py, 0], [c[0] + ux * e + px, c[1] + uy * e + py, 0],
          [c[0] + ux * e - px, c[1] + uy * e - py, 0], [a[0] - ux * e - px, a[1] - uy * e - py, 0]], "rgba(240,244,248,0.92)");
  }
  const net = []; for (let i = 0; i <= 10; i++) { const x = -0.914 + (i / 10) * (DW + 1.828); net.push([x, L / 2, 0.914 + 0.156 * Math.abs(x - CX) / (CX + 0.914)]); }
  poly([...net, [DW + 0.914, L / 2, 0], [-0.914, L / 2, 0]], "rgba(200,210,220,0.18)", "rgba(240,244,248,0.7)", 1.2);

  // trajectoire parcourue + ombre + balle
  const upto = Math.floor(ballIdx);
  g.lineCap = "round";
  for (let i = 1; i <= upto; i++) {
    const sgm = cam.seg(pts[i - 1], pts[i]);
    if (!sgm) continue;
    const [p, q] = sgm;
    g.strokeStyle = BALL; g.globalAlpha = 0.25 + 0.6 * (i / Math.max(1, upto)); g.lineWidth = 2.2 * s;
    g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke();
  }
  g.globalAlpha = 1;
  const i0 = Math.floor(ballIdx), fr = ballIdx - i0, a = pts[i0], c = pts[Math.min(pts.length - 1, i0 + 1)];
  const bp = [a[0] + (c[0] - a[0]) * fr, a[1] + (c[1] - a[1]) * fr, a[2] + (c[2] - a[2]) * fr];
  const sh = proj([bp[0], bp[1], 0]), bb = proj(bp);
  if (sh) { g.fillStyle = "rgba(0,0,0,0.35)"; g.beginPath(); g.ellipse(sh[0], sh[1], Math.max(2 * s, 0.05 * sh[2]), Math.max(1 * s, 0.025 * sh[2]), 0, 0, 7); g.fill(); }
  // trace de la balle au sol (une fois le rebond passe)
  if (ballIdx >= ib) {
    const ang = Math.atan2(dir[1], dir[0]), mark = [];
    for (let k = 0; k < 32; k++) { const q = (2 * Math.PI * k) / 32, lx = 0.065 * Math.cos(q), ly = 0.034 * Math.sin(q);
      mark.push([bc[0] + lx * Math.cos(ang) - ly * Math.sin(ang), bc[1] + lx * Math.sin(ang) + ly * Math.cos(ang), 0.002]); }
    g.globalAlpha = Math.min(1, (ballIdx - ib) / 2 + 0.4); poly(mark, col, "rgba(255,255,255,0.8)", 1); g.globalAlpha = 1;
    // zone d'incertitude de la mesure (a l'echelle reelle) : si elle chevauche la ligne, le verdict est "limite"
    if (b.err && m > 0.3) {
      const zone = [];
      for (let k = 0; k < 48; k++) { const q = (2 * Math.PI * k) / 48; zone.push([bc[0] + b.err[0] * Math.cos(q), bc[1] + b.err[1] * Math.sin(q), 0.003]); }
      const a = Math.min(1, (m - 0.3) / 0.4);
      g.globalAlpha = 0.22 * a; poly(zone, COLORS.close);
      g.globalAlpha = 0.9 * a; g.setLineDash([5 * s, 4 * s]); poly(zone, null, COLORS.close, 1.4); g.setLineDash([]);
      g.globalAlpha = 1;
    }
  }
  if (bb) { g.fillStyle = BALL; g.shadowColor = BALL; g.shadowBlur = 14 * s; g.beginPath(); g.arc(bb[0], bb[1], Math.max(3.5 * s, 0.033 * bb[2]), 0, 7); g.fill(); g.shadowBlur = 0; }

  // mesure finale : distance de la trace a la ligne
  if (m > 0.85) {
    const nb = nearestBoundary(b.court, R), p = proj([b.court[0], b.court[1], 0]), q = proj([nb[0], nb[1], 0]);
    if (p && q) {
      g.globalAlpha = (m - 0.85) / 0.15;
      g.strokeStyle = "#ffffff"; g.setLineDash([6 * s, 5 * s]); g.lineWidth = 1.6 * s;
      g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke(); g.setLineDash([]);
      // meme verdict et meme distance que la liste (zone jugee + incertitude locale)
      const [sd, axis] = edgeDistance(b.court, judgedRect(b)), d = Math.abs(sd);
      const u = b.err ? ` ±${Math.max(1, Math.round(b.err[axis] * 100))} cm` : "";
      const txt = `${label(v)} · ${d >= 1 ? `${num(d, 2)} m` : `${Math.round(d * 100)} cm`}${u}`;
      g.font = `700 ${22 * s}px "JetBrains Mono", monospace`; g.textAlign = "center"; g.textBaseline = "bottom";
      const mx = (p[0] + q[0]) / 2, my = Math.min(p[1], q[1]) - 14 * s;
      const tw = g.measureText(txt).width + 22 * s;
      g.fillStyle = "rgba(13,17,23,0.85)"; g.beginPath(); g.roundRect(mx - tw / 2, my - 34 * s, tw, 36 * s, 10 * s); g.fill();
      g.strokeStyle = col; g.lineWidth = 1.5 * s; g.stroke();
      g.fillStyle = col; g.fillText(txt, mx, my - 6 * s);
      g.globalAlpha = 1;
    }
  }
  if (T < FLY + MOVE + 30) RP.raf = requestAnimationFrame(drawReplay);
}
function boxDistance(c, box) {
  const [x0, x1, y0, y1] = box, dx = Math.max(x0 - c[0], c[0] - x1), dy = Math.max(y0 - c[1], c[1] - y1);
  return dx > 0 && dy > 0 ? Math.hypot(dx, dy) : Math.max(dx, dy);
}
function replayNearest() {          // R : relecture du dernier rebond passe (ou du prochain)
  if (!S.analysis || !S.analysis.bounces.length) return;
  const f = frameAt(video.currentTime), list = S.analysis.bounces.filter((b) => b.court);
  const past = list.filter((b) => b.frame <= f + 2);
  openReplay(past.length ? past[past.length - 1] : list[0]);
}

// ---------- rapport du match (page imprimable / PDF) et CSV ----------
function csvBounces() {
  const head = ["time_s", "frame", "verdict", "x_m", "y_m", "distance_to_line_m", "serve", "speed_kmh"];
  const rows = S.analysis.bounces.map((b) => {
    const d = b.court ? (b.box ? boxDistance(b.court, b.box) : signedDistance(b.court, S.doubles)) : "";
    return [(b.frame / fps()).toFixed(2), b.frame, verdict(b), b.court ? b.court[0] : "", b.court ? b.court[1] : "",
            d === "" ? "" : d.toFixed(3), b.speed && b.speed.serve ? 1 : 0, b.speed ? b.speed.kmh : ""].join(",");
  });
  return [head.join(","), ...rows].join("\n");
}
function downloadCSV() {
  if (!S.analysis) return toast(t("tAnalyzeFirst"));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csvBounces()], { type: "text/csv" }));
  a.download = `${S.current.stem}_bounces.csv`; a.click();
}
function courtSVG(bounces, heat) {
  const lines = COURT_LINES.map(([a, c]) => `<line x1="${a[0]}" y1="${a[1]}" x2="${c[0]}" y2="${c[1]}" stroke="#334155" stroke-width=".08"/>`).join("");
  const dots = heat ? "" : bounces.map((b) => `<circle cx="${b.court[0]}" cy="${b.court[1]}" r=".32" fill="${COLORS[verdict(b)]}" stroke="#fff" stroke-width=".06"/>`).join("");
  const img = heat && bounces.length ? `<image href="${heatmapURL(bounces)}" x="-3" y="-3" width="${DW + 6}" height="${L + 6}" preserveAspectRatio="none"/>` : "";
  return `<svg viewBox="-3 -3 ${DW + 6} ${L + 6}" class="court"><rect x="-1.5" y="-2" width="${DW + 3}" height="${L + 4}" fill="#eef2f6" rx=".4"/>${img}${lines}
    <line x1="-0.914" y1="${L / 2}" x2="${DW + 0.914}" y2="${L / 2}" stroke="#64748b" stroke-width=".1" stroke-dasharray=".3 .2"/>${dots}</svg>`;
}
function openReport() {
  if (!S.analysis) return toast(t("tAnalyzeFirst"));
  const A = S.analysis, bs = A.bounces, judged = bs.filter((b) => b.court);
  const c = { in: 0, out: 0, close: 0, unknown: 0 }; bs.forEach((b) => c[verdict(b)]++);
  const sp = bs.filter((b) => b.speed), rally = sp.filter((b) => !b.speed.serve).map((b) => b.speed.kmh), srv = sp.filter((b) => b.speed.serve).map((b) => b.speed.kmh);
  const avg = (l) => (l.length ? Math.round(l.reduce((a, x) => a + x, 0) / l.length) : "–"), mx = (l) => (l.length ? Math.max(...l) : "–");
  const maxS = Math.max(1, ...sp.map((b) => b.speed.kmh));
  const bars = sp.map((b, i) => { const hgt = (b.speed.kmh / maxS) * 120, x = 10 + i * (560 / Math.max(1, sp.length));
    return `<rect x="${x}" y="${140 - hgt}" width="${Math.max(6, 560 / Math.max(1, sp.length) - 6)}" height="${hgt}" rx="3" fill="${b.speed.serve ? "#0f172a" : "#a3b81f"}"/>
      <text x="${x + 3}" y="${135 - hgt}" font-size="10" fill="#334155">${b.speed.kmh}</text>`; }).join("");
  const rows = bs.map((b) => { const v = verdict(b);
    return `<tr><td>${fmtTime(b.frame / fps())}</td><td><span class="v ${v}">${label(v)}</span></td><td>${describe(b)}</td><td>${b.speed ? b.speed.kmh + " km/h" + (b.speed.serve ? " · " + t("serveTag") : "") : "–"}</td></tr>`; }).join("");
  const R = (k) => t("rp")[k];
  const html = `<!doctype html><html lang="${LANG}"><head><meta charset="utf-8"><title>${R("title")} — ${S.current.stem}</title>
<style>
  body{font:14px/1.5 Inter,system-ui,sans-serif;color:#0f172a;margin:32px auto;max-width:820px;padding:0 16px}
  h1{font-size:24px;margin:0} .sub{color:#64748b;margin:4px 0 24px} h2{font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:#64748b;margin:28px 0 10px}
  .kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px} .k{border:1px solid #e2e8f0;border-radius:10px;padding:10px;text-align:center}
  .k b{display:block;font:600 22px "JetBrains Mono",monospace} .k span{font-size:12px;color:#64748b}
  .in b{color:#059669}.out b{color:#e11d48}.close b{color:#d97706}
  .maps{display:grid;grid-template-columns:1fr 1fr;gap:16px} .maps figure{margin:0;text-align:center} .court{width:100%;max-height:420px}
  figcaption{font-size:12px;color:#64748b;margin-top:6px} table{width:100%;border-collapse:collapse;font-size:13px}
  td,th{border-bottom:1px solid #e2e8f0;padding:6px 8px;text-align:left} th{color:#64748b;font-weight:600}
  .v{font-size:11px;font-weight:700;padding:2px 8px;border-radius:6px}.v.in{background:#d1fae5;color:#065f46}.v.out{background:#ffe4e6;color:#9f1239}
  .v.close{background:#fef3c7;color:#92400e}.v.unknown{background:#f1f5f9;color:#475569}
  .actions{margin:20px 0} button{font:600 13px Inter,sans-serif;padding:8px 14px;border-radius:8px;border:1px solid #cbd5e1;background:#fff;cursor:pointer;margin-right:8px}
  @media print{.actions{display:none} body{margin:0}}
</style></head><body>
<h1>${R("title")}</h1><div class="sub">${S.current.stem} · ${fmtTime(A.frames / A.fps).replace(/\.\d+$/, "")} · ${S.doubles ? t("doubles") : t("singles")} · ${new Date().toLocaleDateString(LANG)}</div>
<div class="actions"><button onclick="print()">${R("print")}</button><button onclick="opener.downloadCSV()">${R("csv")}</button></div>
<h2>${R("summary")}</h2>
<div class="kpis"><div class="k in"><b>${c.in}</b><span>${t("stIn")}</span></div><div class="k out"><b>${c.out}</b><span>${t("stOut")}</span></div>
<div class="k close"><b>${c.close}</b><span>${t("stClose")}</span></div><div class="k"><b>${A.shots.length}</b><span>${t("stShots")}</span></div>
<div class="k"><b>${mx(sp.map((b) => b.speed.kmh))}</b><span>${t("stMax")} (km/h)</span></div><div class="k"><b>${avg(rally)}</b><span>${t("stAvg")} (km/h)</span></div>
<div class="k"><b>${mx(srv)}</b><span>${t("stServe")} (km/h)</span></div><div class="k"><b>${judged.length}/${bs.length}</b><span>${R("judged")}</span></div></div>
<h2>${R("maps")}</h2><div class="maps"><figure>${courtSVG(judged, false)}<figcaption>${R("mapPoints")}</figcaption></figure>
<figure>${courtSVG(judged, true)}<figcaption>${R("mapHeat")}</figcaption></figure></div>
<h2>${R("speeds")}</h2><svg viewBox="0 0 580 150" width="100%">${bars}<line x1="0" y1="140" x2="580" y2="140" stroke="#cbd5e1"/></svg>
<div class="sub">${R("speedNote")}</div>
<h2>${R("bounces")}</h2><table><tr><th>${R("time")}</th><th>${R("call")}</th><th>${R("dist")}</th><th>${R("speed")}</th></tr>${rows}</table>
<div class="sub" style="margin-top:24px">${R("disclaimer")}</div>
</body></html>`;
  const w = window.open(URL.createObjectURL(new Blob([html], { type: "text/html" })), "_blank");
  if (!w) toast(R("popup"));
}
