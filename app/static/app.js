// CourtVision -- interface de demo. Les calques sont dessines en direct sur la video d'origine a
// partir de runs/analysis/<nom>/analysis.json : changer un reglage ne demande aucun recalcul.

// ---------- textes (anglais par defaut, francais au choix) ----------
const I18N = {
  en: {
    subtitle: "Match analysis", videos: "Videos", import: "Import", dropHint: "Drop a video anywhere in the window to add it.",
    rules: "Rules", singles: "Singles", doubles: "Doubles", closeZone: "Close-call zone",
    export: "Export", exportHint: "Annotated video (MP4) with the current rules and language, original sound included.",
    exportBtn: "Export video", pickVideo: "Pick a video from the library.", runAnalysis: "Run analysis",
    tPlay: "Play / pause (Space)", tPrevFrame: "Previous frame (←)", tNextFrame: "Next frame (→)",
    tPrevBounce: "Previous bounce (Shift+B)", tNextBounce: "Next bounce (B)", tMute: "Sound on / off (M)", tRerun: "Re-run the full analysis",
    rate05: "0.5×", rate025: "0.25×",
    lgIn: "In", lgOut: "Out", lgClose: "Close call", lgUnknown: "Not judgeable", lgShot: "Shot",
    layers: "Layers", lyCourt: "Court", lyBall: "Ball", lyTrail: "Trail", lyBounces: "Bounces", lySpeed: "Speed",
    stats: "Statistics", stIn: "In", stOut: "Out", stClose: "Close", stShots: "Shots",
    stMax: "Top speed", stAvg: "Avg speed", stServe: "Top serve", serveTag: "SERVE",
    minimap: "Bounce map", noBounce: "No bounce yet",
    dropHere: "Drop the video to add it to the library",
    analyzed: "Analyzed", toAnalyze: "Not analyzed", queued: "Queued",
    notAnalyzed: "Video not analyzed", analysisQueued: "Analysis queued…", analysisRunning: "Analysis running…",
    keepWatching: "You can keep watching other videos meanwhile.",
    analyzeText: "Court, ball, bounces, then in / out call. About {m} min on this PC ({n} frames).",
    left: "~{t} left", min: "{n} min", sec: "{n} s",
    steps: { court: "Court", ball: "Ball", bounces: "Bounces", judge: "Line call" },
    jobAnalyze: "Analysis", jobExport: "Export",
    inside: "{d} inside", outside: "{d} out", noCourt: "court not visible",
    frame: "frame {n}", exportLink: "↓ {mode} export (MP4)", modeSimple: "Singles", modeDouble: "Doubles",
    tAnalysisStarted: "Analysis started", tAnalysisDone: "Analysis complete", tExportStarted: "Export started (about 1–2 min)",
    tExportDone: "Export complete: download link in the Export panel", tError: "Error: {m}", tAnalyzeFirst: "Analyze the video first.",
    tImport: "Import: {p} %", tAdded: "Video added", tImportFail: "Import failed: {m}", confirmRerun: "Re-run the full analysis of “{v}”?",
    labels: { in: "IN", out: "OUT", close: "CLOSE", unknown: "?" },
  },
  fr: {
    subtitle: "Analyse de match", videos: "Vidéos", import: "Importer", dropHint: "Glisse une vidéo n'importe où dans la fenêtre pour l'ajouter.",
    rules: "Règles", singles: "Simple", doubles: "Double", closeZone: "Zone « limite »",
    export: "Export", exportHint: "Vidéo annotée (MP4) avec les règles et la langue choisies, son d'origine inclus.",
    exportBtn: "Exporter la vidéo", pickVideo: "Choisis une vidéo dans la bibliothèque.", runAnalysis: "Lancer l'analyse",
    tPlay: "Lecture / pause (Espace)", tPrevFrame: "Image précédente (←)", tNextFrame: "Image suivante (→)",
    tPrevBounce: "Rebond précédent (Maj+B)", tNextBounce: "Rebond suivant (B)", tMute: "Son activé / coupé (M)", tRerun: "Relancer toute l'analyse",
    rate05: "0,5×", rate025: "0,25×",
    lgIn: "Bonne", lgOut: "Faute", lgClose: "Limite", lgUnknown: "Non jugeable", lgShot: "Frappe",
    layers: "Calques", lyCourt: "Terrain", lyBall: "Balle", lyTrail: "Traînée", lyBounces: "Rebonds", lySpeed: "Vitesse",
    stats: "Statistiques", stIn: "Bonnes", stOut: "Fautes", stClose: "Limites", stShots: "Frappes",
    stMax: "Vitesse max", stAvg: "Vitesse moy.", stServe: "Service max", serveTag: "SERVICE",
    minimap: "Carte des rebonds", noBounce: "Pas encore de rebond",
    dropHere: "Dépose la vidéo pour l'ajouter à la bibliothèque",
    analyzed: "Analysée", toAnalyze: "À analyser", queued: "En attente",
    notAnalyzed: "Vidéo non analysée", analysisQueued: "Analyse en attente…", analysisRunning: "Analyse en cours…",
    keepWatching: "Tu peux continuer à regarder les autres vidéos pendant ce temps.",
    analyzeText: "Terrain, balle, rebonds puis jugement in / out. Compte environ {m} min sur ce PC ({n} images).",
    left: "encore ~{t}", min: "{n} min", sec: "{n} s",
    steps: { court: "Terrain", ball: "Balle", bounces: "Rebonds", judge: "Jugement" },
    jobAnalyze: "Analyse", jobExport: "Export",
    inside: "{d} dans le terrain", outside: "{d} dehors", noCourt: "terrain non visible",
    frame: "image {n}", exportLink: "↓ Export {mode} (MP4)", modeSimple: "simple", modeDouble: "double",
    tAnalysisStarted: "Analyse lancée", tAnalysisDone: "Analyse terminée", tExportStarted: "Export lancé (environ 1 à 2 min)",
    tExportDone: "Export terminé : lien de téléchargement dans le panneau Export", tError: "Erreur : {m}", tAnalyzeFirst: "Analyse d'abord la vidéo.",
    tImport: "Import : {p} %", tAdded: "Vidéo ajoutée", tImportFail: "Import impossible : {m}", confirmRerun: "Relancer toute l'analyse de « {v} » ?",
    labels: { in: "IN", out: "OUT", close: "LIMITE", unknown: "?" },
  },
};
let LANG = "en";
try { LANG = localStorage.getItem("cv-lang") === "fr" ? "fr" : "en"; } catch { /* stockage indisponible */ }
function t(key, params = {}) {
  const s = I18N[LANG][key] ?? I18N.en[key] ?? key;
  return typeof s === "string" ? s.replace(/\{(\w+)\}/g, (_, k) => params[k] ?? "") : s;
}
function applyLang() {
  document.documentElement.lang = LANG;
  document.querySelectorAll("[data-i18n]").forEach((el) => (el.textContent = t(el.dataset.i18n)));
  document.querySelectorAll("[data-i18n-title]").forEach((el) => (el.title = t(el.dataset.i18nTitle)));
  document.querySelectorAll("#langSwitch button").forEach((b) => b.classList.toggle("active", b.dataset.lang === LANG));
  updateCloseLabel();
}

// ---------- court reglementaire (metres), identique a scripts/predict_court.py ----------
const L = 23.77, DW = 10.97, SW = 8.23, AL = (DW - SW) / 2, SV = L / 2 - 6.40, CX = DW / 2;
const COURT_LINES = [
  [[0, 0], [DW, 0]], [[0, L], [DW, L]], [[0, 0], [0, L]], [[DW, 0], [DW, L]],
  [[AL, 0], [AL, L]], [[DW - AL, 0], [DW - AL, L]],
  [[AL, SV], [DW - AL, SV]], [[AL, L - SV], [DW - AL, L - SV]], [[CX, SV], [CX, L - SV]],
];
const NET = [[-0.914, L / 2], [DW + 0.914, L / 2]];
const COLORS = { in: "#34d399", out: "#f43f5e", close: "#f59e0b", unknown: "#94a3b8" };
const BALL = "#d9f24b", TRAIL = 12, ANIM = 1.4;

const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("overlay"), ctx = canvas.getContext("2d");
const S = {
  videos: [], current: null, analysis: null, ball: new Map(), homs: new Map(),
  layers: { court: true, ball: true, trail: true, bounces: true, speed: true },
  doubles: false, closeCm: 5, jobs: null, slowmoUntil: null,
};

// ---------- utilitaires ----------
const fps = () => (S.analysis ? S.analysis.fps : (S.current ? S.current.fps : 30));
const frameAt = (time) => Math.floor(time * fps() + 1e-3);
const fmtTime = (s) => { s = Math.max(0, s); const m = Math.floor(s / 60); return `${m}:${(s - 60 * m).toFixed(2).padStart(5, "0")}`; };
const easeOut = (x) => 1 - Math.pow(1 - x, 3);
const num = (x, d) => (LANG === "fr" ? x.toFixed(d).replace(".", ",") : x.toFixed(d));
function toast(msg, ms = 2600) { const el = $("toast"); el.textContent = msg; el.classList.remove("hidden"); clearTimeout(el._h); el._h = setTimeout(() => el.classList.add("hidden"), ms); }
// cache: "no-store" : les analyses changent sur le disque (reanalyse, nouvelles fonctions), le navigateur
// ne doit jamais resservir une ancienne copie
async function api(path, opts) { const r = await fetch(path, { cache: "no-store", ...opts }); if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText); return r.json(); }

function project(H, [X, Y]) {
  const w = H[2][0] * X + H[2][1] * Y + H[2][2];
  return [(H[0][0] * X + H[0][1] * Y + H[0][2]) / w, (H[1][0] * X + H[1][1] * Y + H[1][2]) / w];
}
function signedDistance([x, y], doubles) {          // < 0 dedans, > 0 dehors (metres)
  const x0 = doubles ? 0 : AL, x1 = doubles ? DW : DW - AL;
  const dx = Math.max(x0 - x, x - x1), dy = Math.max(-y, y - L);
  return dx > 0 && dy > 0 ? Math.hypot(dx, dy) : Math.max(dx, dy);
}
function verdict(b) {
  if (!b.court) return "unknown";
  const s = signedDistance(b.court, S.doubles);
  if (Math.abs(s) <= S.closeCm / 100) return "close";
  return s < 0 ? "in" : "out";
}
const label = (v) => t("labels")[v];
function describe(b) {
  if (!b.court) return t("noCourt");
  const s = signedDistance(b.court, S.doubles), a = Math.abs(s);
  const d = a >= 1 ? `${num(a, 2)} m` : `${Math.round(a * 100)} cm`;
  return t(s < 0 ? "inside" : "outside", { d });
}

// ---------- canvas : taille et repere (pixels de la video) ----------
let view = { scale: 1, ox: 0, oy: 0, dpr: 1 };
function resizeCanvas() {
  const r = $("player").getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr);
  const vw = video.videoWidth || 16, vh = video.videoHeight || 9;
  const scale = Math.min(r.width / vw, r.height / vh);
  view = { scale, ox: (r.width - vw * scale) / 2, oy: (r.height - vh * scale) / 2, dpr };
}
new ResizeObserver(resizeCanvas).observe($("player"));
video.addEventListener("loadedmetadata", resizeCanvas);

// ---------- dessin sur la video ----------
function groundPoly(H, c, r, n = 48) {
  const pts = [];
  for (let i = 0; i < n; i++) { const a = (2 * Math.PI * i) / n; pts.push(project(H, [c[0] + r * Math.cos(a), c[1] + r * Math.sin(a)])); }
  return pts;
}
function screenPoly(c, r) {        // repli sans terrain : ellipse dans le plan de l'image
  const k = (S.analysis ? S.analysis.height : 720) / 720, pts = [];
  for (let i = 0; i < 40; i++) { const a = (2 * Math.PI * i) / 40; pts.push([c[0] + r * 40 * k * Math.cos(a), c[1] + r * 16 * k * Math.sin(a)]); }
  return pts;
}
function pathPoly(pts) { ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath(); }

function drawCourt(H, px) {
  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.6)"; ctx.lineWidth = 1.6 * px; ctx.lineCap = "round";
  ctx.shadowColor = "rgba(255,255,255,0.35)"; ctx.shadowBlur = 6 * px;
  for (const [a, b] of COURT_LINES) { const p = project(H, a), q = project(H, b); ctx.beginPath(); ctx.moveTo(...p); ctx.lineTo(...q); ctx.stroke(); }
  ctx.setLineDash([6 * px, 6 * px]); ctx.strokeStyle = "rgba(255,255,255,0.35)"; ctx.lineWidth = 1.2 * px;
  const p = project(H, NET[0]), q = project(H, NET[1]); ctx.beginPath(); ctx.moveTo(...p); ctx.lineTo(...q); ctx.stroke();
  ctx.restore();
}

function drawBall(f, px) {
  if (S.layers.trail) {
    ctx.save(); ctx.lineCap = "round"; ctx.strokeStyle = BALL; ctx.shadowColor = BALL; ctx.shadowBlur = 8 * px;
    for (let g = f - TRAIL + 1; g <= f; g++) {
      const a = S.ball.get(g - 1), b = S.ball.get(g);
      if (!a || !b) continue;
      const k = (g - (f - TRAIL)) / TRAIL;
      ctx.globalAlpha = k * 0.9; ctx.lineWidth = (1 + 3 * k) * px;
      ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
    }
    ctx.restore();
  }
  const p = S.ball.get(f);
  if (S.layers.ball && p) {
    ctx.save(); ctx.strokeStyle = BALL; ctx.lineWidth = 2.2 * px; ctx.shadowColor = BALL; ctx.shadowBlur = 12 * px;
    ctx.beginPath(); ctx.arc(p[0], p[1], 10 * px, 0, 2 * Math.PI); ctx.stroke();
    ctx.fillStyle = BALL; ctx.beginPath(); ctx.arc(p[0], p[1], 2 * px, 0, 2 * Math.PI); ctx.fill();
    ctx.restore();
  }
}

function drawLabel(text, x, y, color, alpha, px) {
  ctx.save(); ctx.globalAlpha = alpha;
  ctx.font = `700 ${13 * px}px Inter, system-ui, sans-serif`;
  const w = ctx.measureText(text).width + 20 * px, h = 24 * px;
  ctx.fillStyle = "rgba(13,17,23,0.85)"; ctx.strokeStyle = color; ctx.lineWidth = 1.2 * px;
  ctx.beginPath(); ctx.roundRect(x - w / 2, y - h / 2, w, h, 7 * px); ctx.fill(); ctx.stroke();
  ctx.fillStyle = color; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(text, x, y + 0.5 * px);
  ctx.restore();
}

function drawBounce(b, time, H, px) {
  const v = verdict(b), color = COLORS[v];
  const onGround = b.court && H;
  const poly = (r) => (onGround ? groundPoly(H, b.court, r) : screenPoly([b.x, b.y], r));
  const center = onGround ? project(H, b.court) : [b.x, b.y];

  if (time < ANIM) {
    ctx.save(); ctx.strokeStyle = color; ctx.shadowColor = color; ctx.shadowBlur = 10 * px;
    for (const [delay, rmax, lw] of [[0, 1.1, 2.4], [0.15, 0.7, 1.6]]) {     // deux ondes au sol
      const u = (time - delay) / 0.9;
      if (u < 0 || u > 1) continue;
      ctx.globalAlpha = 1 - u; ctx.lineWidth = lw * px;
      pathPoly(poly(0.12 + rmax * easeOut(u))); ctx.stroke();
    }
    if (time < 0.25) {                                                   // flash
      ctx.globalAlpha = 0.65 * (1 - time / 0.25); ctx.fillStyle = "#fff"; ctx.shadowColor = "#fff"; ctx.shadowBlur = 25 * px;
      pathPoly(poly(0.12 + 0.35 * (time / 0.25))); ctx.fill();
    }
    ctx.restore();
    const a = Math.min(1, time / 0.15) * (1 - Math.max(0, (time - 1.0) / (ANIM - 1.0)));
    drawLabel(label(v), center[0], center[1] - (30 + 16 * easeOut(Math.min(1, time / 0.6))) * px, color, a, px);
  }
  ctx.save();                                                            // marque au sol persistante
  ctx.globalAlpha = 0.92; ctx.fillStyle = color; ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.lineWidth = 1 * px;
  ctx.shadowColor = color; ctx.shadowBlur = 8 * px;
  pathPoly(poly(0.11)); ctx.fill(); ctx.shadowBlur = 0; ctx.stroke();
  ctx.restore();
}

// vitesse de la frappe, affichee au depart de la trajectoire jusqu'au rebond
function drawSpeed(b, f, px) {
  const sp = b.speed;
  if (!sp || f < sp.hit_frame || f > b.frame + 0.4 * fps()) return;
  const tIn = (f - sp.hit_frame) / fps(), tOut = (f - b.frame) / fps();
  const alpha = Math.min(1, tIn / 0.15) * (tOut > 0 ? Math.max(0, 1 - tOut / 0.4) : 1);
  const x = sp.hit_x;
  let y = sp.hit_y - (34 + 8 * easeOut(Math.min(1, tIn / 0.4))) * px;
  ctx.save(); ctx.globalAlpha = alpha;
  const big = `700 ${15 * px}px "JetBrains Mono", Consolas, monospace`, small = `600 ${10 * px}px Inter, sans-serif`;
  const tag = sp.serve ? t("serveTag") : "", value = `${sp.kmh}`, unit = " km/h";
  ctx.font = big; const wv = ctx.measureText(value).width;
  ctx.font = small; const wu = ctx.measureText(unit).width, wt = tag ? ctx.measureText(tag + "  ").width : 0;
  const w = wt + wv + wu + 22 * px, h = 26 * px, vw = S.analysis.width;
  const x0 = Math.min(Math.max(x - w / 2, 4 * px), vw - w - 4 * px);   // reste dans l'image
  y = Math.max(y, h / 2 + 4 * px);
  ctx.fillStyle = "rgba(13,17,23,0.82)"; ctx.strokeStyle = "rgba(217,242,75,0.55)"; ctx.lineWidth = 1 * px;
  ctx.beginPath(); ctx.roundRect(x0, y - h / 2, w, h, 13 * px); ctx.fill(); ctx.stroke();
  ctx.textBaseline = "middle"; let cx = x0 + 11 * px;
  if (tag) { ctx.font = small; ctx.fillStyle = "#8b96a5"; ctx.fillText(tag, cx, y + 0.5 * px); cx += wt; }
  ctx.font = big; ctx.fillStyle = BALL; ctx.fillText(value, cx, y + 1 * px); cx += wv;
  ctx.font = small; ctx.fillStyle = "#c9d3a0"; ctx.fillText(unit, cx, y + 1.5 * px);
  ctx.restore();
}

function render() {
  const { scale, ox, oy, dpr } = view;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!S.analysis || !video.videoWidth) return;
  ctx.setTransform(dpr * scale, 0, 0, dpr * scale, dpr * ox, dpr * oy);
  const px = 1 / scale;   // 1 pixel ecran, en pixels video
  const f = frameAt(video.currentTime);
  const H = S.homs.get(f);
  if (S.layers.court && H) drawCourt(H, px);
  if (S.layers.bounces) {
    for (const b of S.analysis.bounces) {
      if (b.frame > f) break;
      drawBounce(b, (f - b.frame) / fps(), H || S.homs.get(b.frame), px);
    }
  }
  drawBall(f, px);
  if (S.layers.speed) for (const b of S.analysis.bounces) drawSpeed(b, f, px);
}

// ---------- mini-carte : le court vu de dessus, cote eloigne en haut (comme la camera) ----------
const SVGNS = "http://www.w3.org/2000/svg";
const mm = $("minimap");
const svgEl = (tag, attrs, parent = mm) => { const e = document.createElementNS(SVGNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };
let mmState = { key: null };
function drawMinimapBase() {
  mm.innerHTML = "";
  svgEl("rect", { class: "surface", x: -1.5, y: -2, width: DW + 3, height: L + 4, rx: 0.4 });
  if (!S.doubles) {                         // en simple, les couloirs sont fautes
    svgEl("rect", { class: "alley-out", x: 0, y: 0, width: AL, height: L });
    svgEl("rect", { class: "alley-out", x: DW - AL, y: 0, width: AL, height: L });
  }
  for (const [a, b] of COURT_LINES) svgEl("line", { class: "cline", x1: a[0], y1: a[1], x2: b[0], y2: b[1] });
  svgEl("line", { class: "net", x1: NET[0][0], y1: NET[0][1], x2: NET[1][0], y2: NET[1][1] });
  svgEl("circle", { class: "post", cx: NET[0][0], cy: L / 2, r: 0.12 });
  svgEl("circle", { class: "post", cx: NET[1][0], cy: L / 2, r: 0.12 });
  return svgEl("g", {});
}
function updateMinimap(force = false) {
  const f = S.analysis ? frameAt(video.currentTime) : -1;
  const shown = S.analysis ? S.analysis.bounces.filter((b) => b.frame <= f && b.court) : [];
  const key = `${S.current && S.current.name}|${shown.length}|${S.doubles}|${S.closeCm}|${LANG}`;
  if (!force && key === mmState.key) return;
  const grew = mmState.key && key.split("|")[0] === mmState.key.split("|")[0] && shown.length > (mmState.count || 0);
  mmState = { key, count: shown.length };
  const g = drawMinimapBase();
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  shown.forEach((b, i) => {
    const v = verdict(b), cx = clamp(b.court[0], -2.8, DW + 2.8), cy = clamp(b.court[1], -2.8, L + 2.8);
    const last = i === shown.length - 1;
    if (last) {
      svgEl("circle", { class: "latest", cx, cy, r: 0.6 }, g);
      if (grew) svgEl("circle", { class: `ripple ${v}`, cx, cy, r: 0.3 }, g);   // le rebond qui vient d'arriver
    }
    const c = svgEl("circle", { class: `bounce ${v}`, cx, cy, r: last ? 0.36 : 0.3 }, g);
    c.addEventListener("click", () => jumpToBounce(b));
    c.addEventListener("mouseenter", () => {
      const tip = $("minimapTip"), r = mm.getBoundingClientRect(), wr = mm.parentElement.getBoundingClientRect();
      const ctm = c.getScreenCTM(), x = ctm.e + ctm.a * cx - wr.left, y = ctm.f + ctm.d * cy - wr.top;
      tip.innerHTML = `${fmtTime(b.frame / fps())} · <b style="color:${COLORS[v]}">${label(v)}</b> · ${describe(b)}`;
      tip.style.left = x + "px"; tip.style.top = (y - 38) + "px"; tip.classList.remove("hidden");
      void r;
    });
    c.addEventListener("mouseleave", () => $("minimapTip").classList.add("hidden"));
  });
  if (!shown.length) {
    const e = svgEl("text", { class: "side-label", x: CX, y: L / 2 - 0.8 }); e.textContent = S.analysis ? t("noBounce") : "";
  }
}

function tick() {
  render(); updateTimeUI(); updateMinimap();
  if (S.slowmoUntil !== null && video.currentTime > S.slowmoUntil) { video.playbackRate = currentRate(); S.slowmoUntil = null; }
  requestAnimationFrame(tick);
}

// ---------- lecture et son ----------
const currentRate = () => +document.querySelector("#speeds .active").dataset.rate;
function setPlayIcon() { $("playIcon").innerHTML = video.paused ? '<path d="M7 4.5v15l13-7.5z" class="fill"/>' : '<path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z" class="fill"/>'; }
function setMuteIcon() {
  const off = video.muted || video.volume === 0;
  $("muteIcon").innerHTML = '<path d="M4 9v6h4l5 4V5L8 9z" class="fill"/>' +
    (off ? '<path d="M17 9l5 6M22 9l-5 6"/>' : '<path d="M16.5 8.5a5 5 0 0 1 0 7M19 6a8.5 8.5 0 0 1 0 12"/>');
}
function togglePlay() { if (!S.current) return; video.paused ? video.play() : video.pause(); }
video.addEventListener("play", setPlayIcon); video.addEventListener("pause", setPlayIcon);
video.addEventListener("volumechange", setMuteIcon);
video.volume = 0.8;
$("muteBtn").onclick = () => { video.muted = !video.muted; if (!video.muted && video.volume === 0) video.volume = 0.8; $("volume").value = video.muted ? 0 : video.volume; };
$("volume").oninput = (e) => { video.volume = +e.target.value; video.muted = video.volume === 0; };
function seekFrame(f) { video.currentTime = (Math.max(0, f) + 0.1) / fps(); }
function stepFrame(d) { video.pause(); seekFrame(frameAt(video.currentTime) + d); }
function jumpToBounce(b) {          // saut juste avant le rebond, au ralenti
  video.currentTime = Math.max(0, (b.frame - 0.8 * fps()) / fps());
  video.playbackRate = 0.5; S.slowmoUntil = (b.frame + 0.9 * fps()) / fps();
  video.play();
}
function neighbourBounce(dir) {
  if (!S.analysis) return;
  const f = frameAt(video.currentTime), list = S.analysis.bounces;
  const b = dir > 0 ? list.find((x) => x.frame > f + 1) : [...list].reverse().find((x) => x.frame < f - 0.9 * fps());
  if (b) jumpToBounce(b);
}

$("playBtn").onclick = togglePlay;
$("prevFrame").onclick = () => stepFrame(-1);
$("nextFrame").onclick = () => stepFrame(1);
$("prevBounce").onclick = () => neighbourBounce(-1);
$("nextBounce").onclick = () => neighbourBounce(1);
$("player").addEventListener("click", togglePlay);
document.querySelectorAll("#speeds .chip").forEach((c) => (c.onclick = () => {
  document.querySelectorAll("#speeds .chip").forEach((x) => x.classList.remove("active"));
  c.classList.add("active"); video.playbackRate = +c.dataset.rate; S.slowmoUntil = null;
}));

// ---------- frise ----------
function updateTimeUI() {
  const d = video.duration || 0, tm = video.currentTime || 0, pct = d ? (100 * tm) / d : 0;
  $("played").style.width = pct + "%"; $("playhead").style.left = pct + "%";
  $("timeCur").textContent = fmtTime(tm); $("timeTot").textContent = fmtTime(d);
  $("frameNo").textContent = S.current ? t("frame", { n: frameAt(tm) }) : "";
  const f = frameAt(tm);
  document.querySelectorAll("#bounceList li").forEach((li) => li.classList.toggle("current", Math.abs(+li.dataset.frame - f) < fps() * 0.6));
}
function buildTimeline() {
  const m = $("markers"); m.innerHTML = "";
  if (!S.analysis) return;
  const n = S.analysis.frames;
  for (const s of S.analysis.shots) {
    const e = document.createElement("div"); e.className = "shot-tick"; e.style.left = (100 * s) / n + "%"; m.appendChild(e);
  }
  for (const b of S.analysis.bounces) {
    const e = document.createElement("div"), v = verdict(b);
    e.className = `marker ${v}`; e.style.left = (100 * b.frame) / n + "%";
    e.onmouseenter = () => showTip(e, `${fmtTime(b.frame / fps())} · <b style="color:${COLORS[v]}">${label(v)}</b> · ${describe(b)}`);
    e.onmouseleave = hideTip;
    e.onclick = (ev) => { ev.stopPropagation(); jumpToBounce(b); };
    m.appendChild(e);
  }
}
function showTip(el, html) { const tip = $("tooltip"); tip.innerHTML = html; tip.style.left = el.style.left; tip.classList.remove("hidden"); }
function hideTip() { $("tooltip").classList.add("hidden"); }
(function timelineDrag() {
  const tl = $("timeline"); let drag = false;
  const seek = (ev) => { const r = tl.getBoundingClientRect(); video.currentTime = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)) * (video.duration || 0); };
  tl.addEventListener("pointerdown", (ev) => { if (ev.target.classList.contains("marker")) return; drag = true; tl.setPointerCapture(ev.pointerId); seek(ev); });
  tl.addEventListener("pointermove", (ev) => drag && seek(ev));
  tl.addEventListener("pointerup", () => (drag = false));
})();

// ---------- statistiques et liste des rebonds ----------
function refreshVerdicts() {
  if (!S.analysis) {
    ["stIn", "stOut", "stClose", "stShots", "stMax", "stAvg", "stServe"].forEach((id) => ($(id).textContent = "–"));
    $("bounceList").innerHTML = ""; $("markers").innerHTML = ""; updateMinimap(true); return;
  }
  const c = { in: 0, out: 0, close: 0, unknown: 0 };
  const list = $("bounceList"); list.innerHTML = "";
  for (const b of S.analysis.bounces) {
    const v = verdict(b); c[v]++;
    const li = document.createElement("li"); li.dataset.frame = b.frame;
    li.innerHTML = `<span class="mono muted">${fmtTime(b.frame / fps())}</span><span class="verdict ${v}">${label(v)}</span><span class="muted">${describe(b)}</span><span class="mono spd">${b.speed ? b.speed.kmh + " km/h" : ""}</span>`;
    li.onclick = () => jumpToBounce(b);
    list.appendChild(li);
  }
  $("stIn").textContent = c.in; $("stOut").textContent = c.out; $("stClose").textContent = c.close;
  $("stShots").textContent = S.analysis.shots.length;
  const speeds = S.analysis.bounces.filter((b) => b.speed).map((b) => b.speed);
  const rally = speeds.filter((s) => !s.serve).map((s) => s.kmh), serves = speeds.filter((s) => s.serve).map((s) => s.kmh);
  const kmh = (v) => (v == null ? "–" : `${Math.round(v)}<small>km/h</small>`);
  $("stMax").innerHTML = kmh(speeds.length ? Math.max(...speeds.map((s) => s.kmh)) : null);
  $("stAvg").innerHTML = kmh(rally.length ? rally.reduce((a, c) => a + c, 0) / rally.length : null);
  $("stServe").innerHTML = kmh(serves.length ? Math.max(...serves) : null);
  buildTimeline(); updateMinimap(true);
}
document.querySelectorAll("[data-layer]").forEach((cb) => (cb.onchange = () => (S.layers[cb.dataset.layer] = cb.checked)));
document.querySelectorAll("#modeSeg button").forEach((b) => (b.onclick = () => setMode(b.dataset.mode === "double")));
function setMode(doubles) {
  S.doubles = doubles;
  document.querySelectorAll("#modeSeg button").forEach((x) => x.classList.toggle("active", (x.dataset.mode === "double") === doubles));
  refreshVerdicts();
}
function updateCloseLabel() { $("closeVal").textContent = `± ${S.closeCm} cm`; }
$("closeRange").oninput = (e) => { S.closeCm = +e.target.value; updateCloseLabel(); refreshVerdicts(); };

// ---------- bibliotheque ----------
function jobFor(name, type = "analyze") {
  const J = S.jobs; if (!J) return null;
  return [J.current, ...J.queued].find((j) => j && j.video === name && j.type === type) || null;
}
const REDO_ICON = '<svg viewBox="0 0 24 24"><path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7"/></svg>';
function renderLibrary() {
  const ul = $("videoList"); ul.innerHTML = "";
  for (const v of S.videos) {
    const job = jobFor(v.name);
    const li = document.createElement("li");
    li.className = "video-item" + (S.current && S.current.name === v.name ? " active" : "");
    const dur = fmtTime(v.frames / v.fps).replace(/\.\d+$/, "");
    const badge = job ? `<span class="badge run">${job.status === "queued" ? t("queued") : Math.round(100 * job.progress) + " %"}</span>`
      : v.analyzed ? `<span class="badge ok">${t("analyzed")}</span>` : `<span class="badge todo">${t("toAnalyze")}</span>`;
    li.innerHTML = `<div class="name" title="${v.name}">${v.stem}</div>
      ${v.analyzed && !job ? `<button class="redo" title="${t("tRerun")}">${REDO_ICON}</button>` : "<span></span>"}
      <div class="meta">${badge}<span>${dur}</span><span>${v.width}×${v.height}</span></div>
      ${job ? `<div class="mini-bar"><div style="width:${100 * job.progress}%"></div></div>` : ""}`;
    li.onclick = () => selectVideo(v.name);
    const redo = li.querySelector(".redo");
    if (redo) redo.onclick = (ev) => { ev.stopPropagation(); if (confirm(t("confirmRerun", { v: v.stem }))) startAnalysis(v.name, true); };
    ul.appendChild(li);
  }
}
async function loadVideos() { S.videos = await api("/api/videos"); renderLibrary(); }

async function selectVideo(name) {
  const v = S.videos.find((x) => x.name === name); if (!v) return;
  S.current = v; S.analysis = null; S.ball = new Map(); S.homs = new Map();
  video.pause(); video.playbackRate = currentRate();
  video.src = `/api/videos/${encodeURIComponent(name)}/file`;
  $("emptyState").classList.add("hidden");
  renderLibrary(); refreshVerdicts(); renderExports();
  if (v.analyzed) await loadAnalysis(); else showAnalyzeCard();
}
async function loadAnalysis() {
  const a = await api(`/api/videos/${encodeURIComponent(S.current.name)}/analysis`);
  S.analysis = a;
  S.ball = new Map(a.ball.map(([f, x, y]) => [f, [x, y]]));
  S.homs = new Map(Object.entries(a.homographies).map(([k, v]) => [+k, v]));
  $("analyzeCard").classList.add("hidden");
  refreshVerdicts();
}

// ---------- analyse ----------
function showAnalyzeCard() {
  const v = S.current, job = jobFor(v.name), card = $("analyzeCard");
  card.classList.remove("hidden");
  const steps = (S.jobs && S.jobs.steps) || [];
  const mins = Math.max(1, Math.round((v.frames * 1.05) / 60));
  $("analyzeTitle").textContent = job ? t(job.status === "queued" ? "analysisQueued" : "analysisRunning") : t("notAnalyzed");
  $("analyzeText").textContent = job ? t("keepWatching") : t("analyzeText", { m: mins, n: v.frames });
  const cur = job && job.step ? steps.findIndex((s) => s.key === job.step) : -1;
  $("stepList").innerHTML = steps.map((s, i) => `<li class="${i < cur ? "done" : i === cur ? "active" : ""}">${i + 1}. ${t("steps")[s.key] || s.label}</li>`).join("");
  $("analyzeBar").style.width = job ? 100 * job.progress + "%" : "0%";
  $("analyzeMsg").textContent = job ? etaText(job) : "";
  $("analyzeBtn").classList.toggle("hidden", !!job);
  card.querySelector(".progress").classList.toggle("hidden", !job);
}
function etaText(job) {
  const fr = /frame (\d+)\/(\d+)/.exec(job.message || "");
  const where = fr ? t("frame", { n: `${fr[1]} / ${fr[2]}` }) : "";
  if (job.status !== "running" || !job.started || job.progress < 0.03) return where;
  const el = Date.now() / 1000 - job.started, rest = (el / job.progress) * (1 - job.progress);
  const left = t("left", { t: rest > 90 ? t("min", { n: Math.round(rest / 60) }) : t("sec", { n: Math.round(rest) }) });
  return where ? `${where} · ${left}` : left;
}
async function startAnalysis(name, force = false) {
  await api(`/api/videos/${encodeURIComponent(name)}/analyze?force=${force}`, { method: "POST" });
  toast(t("tAnalysisStarted")); await pollJobs();
  if (S.current && S.current.name === name && force) { S.analysis = null; S.ball = new Map(); S.homs = new Map(); refreshVerdicts(); }
}
$("analyzeBtn").onclick = () => S.current && startAnalysis(S.current.name);

// ---------- export ----------
function renderExports() {
  const box = $("exportLinks"); box.innerHTML = "";
  if (!S.current) return;
  for (const f of S.current.exports || []) {
    const a = document.createElement("a");
    a.href = `/api/videos/${encodeURIComponent(S.current.name)}/export/${f}`;
    a.textContent = t("exportLink", { mode: t(f.includes("double") ? "modeDouble" : "modeSimple") });
    box.appendChild(a);
  }
}
$("exportBtn").onclick = async () => {
  if (!S.analysis) return toast(t("tAnalyzeFirst"));
  await api(`/api/videos/${encodeURIComponent(S.current.name)}/export?doubles=${S.doubles}&close_cm=${S.closeCm}&lang=${LANG}`, { method: "POST" });
  toast(t("tExportStarted")); pollJobs();
};

// ---------- suivi des taches ----------
let wasRunning = new Set();
async function pollJobs() {
  try { S.jobs = await api("/api/jobs"); } catch { return; }
  const J = S.jobs, active = [J.current, ...J.queued].filter(Boolean);
  const pill = $("jobPill");
  if (J.current) {
    pill.classList.remove("hidden");
    const what = t(J.current.type === "analyze" ? "jobAnalyze" : "jobExport");
    $("jobPillText").textContent = `${what} · ${J.current.video.replace(/\.\w+$/, "")} · ${Math.round(100 * J.current.progress)} %`;
  } else pill.classList.add("hidden");

  const now = new Set(active.map((j) => j.type + "|" + j.video));
  const finished = [...wasRunning].filter((k) => !now.has(k));
  wasRunning = now;
  if (finished.length) {
    await loadVideos();
    if (J.last && J.last.status === "error") toast(t("tError", { m: J.last.message }), 6000);
    if (S.current) {
      S.current = S.videos.find((v) => v.name === S.current.name) || S.current;
      if (finished.includes("analyze|" + S.current.name) && S.current.analyzed) { await loadAnalysis(); toast(t("tAnalysisDone")); }
      if (finished.includes("export|" + S.current.name)) toast(t("tExportDone"));
      renderExports();
    }
  }
  renderLibrary();
  if (S.current && !S.current.analyzed) showAnalyzeCard();
  const ex = S.current && jobFor(S.current.name, "export");
  $("exportProg").classList.toggle("hidden", !ex);
  if (ex) $("exportProg").firstElementChild.style.width = 100 * ex.progress + "%";
  $("exportBtn").disabled = !!ex;
}
setInterval(pollJobs, 1000);

// ---------- import (bouton ou glisser-deposer) ----------
function upload(file) {
  const xhr = new XMLHttpRequest(), fd = new FormData();
  fd.append("file", file);
  xhr.open("POST", "/api/upload");
  xhr.upload.onprogress = (e) => e.lengthComputable && toast(t("tImport", { p: Math.round((100 * e.loaded) / e.total) }), 60000);
  xhr.onload = async () => {
    if (xhr.status !== 200) return toast(t("tImportFail", { m: xhr.responseText }), 6000);
    const { name } = JSON.parse(xhr.responseText);
    toast(t("tAdded")); await loadVideos(); selectVideo(name);
  };
  xhr.send(fd);
}
$("fileInput").onchange = (e) => e.target.files[0] && upload(e.target.files[0]);
let dragDepth = 0;
window.addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; $("dropOverlay").classList.remove("hidden"); });
window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("dropOverlay").classList.add("hidden"); } });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault(); dragDepth = 0; $("dropOverlay").classList.add("hidden");
  const f = e.dataTransfer.files[0]; if (f) upload(f);
});

// ---------- langue et clavier ----------
document.querySelectorAll("#langSwitch button").forEach((b) => (b.onclick = () => {
  LANG = b.dataset.lang;
  try { localStorage.setItem("cv-lang", LANG); } catch { /* stockage indisponible */ }
  applyLang(); renderLibrary(); refreshVerdicts(); renderExports();
  if (S.current && !S.current.analyzed) showAnalyzeCard();
}));
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" && e.target.type !== "checkbox" && e.target.type !== "range") return;
  const k = e.key.toLowerCase();
  if (k === " ") { e.preventDefault(); togglePlay(); }
  else if (k === "arrowleft") { e.preventDefault(); stepFrame(-1); }
  else if (k === "arrowright") { e.preventDefault(); stepFrame(1); }
  else if (k === "b") neighbourBounce(e.shiftKey ? -1 : 1);
  else if (k === "m") $("muteBtn").click();
  else if (k === "s") setMode(!S.doubles);
  else if ("12345".includes(k) && k.length === 1) {
    const cb = document.querySelector(`[data-layer="${["court", "ball", "trail", "bounces", "speed"][+k - 1]}"]`);
    cb.checked = !cb.checked; cb.onchange();
  }
});

// ---------- demarrage ----------
(async function init() {
  applyLang(); setMode(false); setMuteIcon(); setPlayIcon();
  await pollJobs();
  await loadVideos();
  const first = S.videos.find((v) => v.analyzed) || S.videos[0];
  if (first) selectVideo(first.name);
  requestAnimationFrame(tick);
})();
