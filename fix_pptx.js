const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "TrueSkill Matchmaking — ECE Paris ING4 Groupe 02";
pres.author = "ECE Paris ING4 Groupe 02";

const C = {
  bg:       "0D1117",
  bg2:      "161B22",
  bg3:      "1F2937",
  purple:   "667EEA",
  purple2:  "764BA2",
  white:    "FFFFFF",
  gray:     "94A3B8",
  gray2:    "64748B",
  green:    "22C55E",
  orange:   "F97316",
  red:      "EF4444",
  cyan:     "06B6D4",
  yellow:   "EAB308",
  border:   "334155",
};

const makeShadow = () => ({ type: "outer", blur: 8, offset: 3, angle: 135, color: "000000", opacity: 0.3 });

function addBg(slide, color) { slide.background = { color: color || C.bg }; }

function addTitle(slide, text, y) {
  y = y !== undefined ? y : 0.25;
  slide.addText(text, { x: 0.4, y: y, w: 9.2, h: 0.65, fontSize: 26, bold: true, color: C.purple, fontFace: "Calibri", align: "left", margin: 0 });
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: y + 0.67, w: 1.2, h: 0.04, fill: { color: C.purple2 }, line: { color: C.purple2 } });
}

function addCard(slide, x, y, w, h, color) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: color || C.bg2 }, line: { color: C.border, width: 1 }, shadow: makeShadow() });
}

function addBadge(slide, text, x, y, color) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 1.6, h: 0.32, fill: { color }, line: { color }, rectRadius: 0.15 });
  slide.addText(text, { x, y, w: 1.6, h: 0.32, fontSize: 11, bold: true, color: C.white, fontFace: "Calibri", align: "center", margin: 0, valign: "middle" });
}

// ══════════════════════════════════════════════════
// SLIDE 1 — TITRE
// ══════════════════════════════════════════════════
{
  const s = pres.addSlide();
  addBg(s, C.bg);
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.18, h: 5.625, fill: { color: C.purple }, line: { color: C.purple } });
  s.addShape(pres.shapes.RECTANGLE, { x: 9.82, y: 0, w: 0.18, h: 5.625, fill: { color: C.purple2 }, line: { color: C.purple2 } });
  addCard(s, 1.2, 1.1, 7.6, 3.4, "161B22");
  s.addShape(pres.shapes.RECTANGLE, { x: 1.2, y: 1.1, w: 0.1, h: 3.4, fill: { color: C.purple }, line: { color: C.purple } });
  s.addText("TrueSkill Matchmaking", { x: 1.5, y: 1.4, w: 7.0, h: 1.0, fontSize: 42, bold: true, color: C.purple, fontFace: "Calibri", align: "left", margin: 0 });
  s.addText("Système de classement Bayésien compétitif", { x: 1.5, y: 2.45, w: 7.0, h: 0.6, fontSize: 20, color: C.white, fontFace: "Calibri", align: "left", margin: 0 });
  s.addShape(pres.shapes.LINE, { x: 1.5, y: 3.15, w: 6.0, h: 0, line: { color: C.border, width: 1 } });
  s.addText([
    { text: "ECE Paris", options: { bold: true, color: C.purple } },
    { text: "  ·  ING4  ·  Groupe 02", options: { color: C.gray } },
  ], { x: 1.5, y: 3.3, w: 7.0, h: 0.45, fontSize: 15, fontFace: "Calibri", align: "left", margin: 0 });
  const tags = ["Python", "TrueSkill", "Bayésien", "Streamlit"];
  tags.forEach((t, i) => {
    const tx = 1.5 + i * 1.65;
    s.addShape(pres.shapes.RECTANGLE, { x: tx, y: 3.95, w: 1.45, h: 0.28, fill: { color: "1E293B" }, line: { color: C.border, width: 1 } });
    s.addText(t, { x: tx, y: 3.95, w: 1.45, h: 0.28, fontSize: 10, color: C.cyan, fontFace: "Calibri", align: "center", margin: 0, valign: "middle" });
  });
}

// ══════════════════════════════════════════════════
// SLIDE 2 — PROBLÉMATIQUE
// ══════════════════════════════════════════════════
{
  const s = pres.addSlide();
  addBg(s);
  addTitle(s, "Pourquoi ELO ne suffit pas ?");
  addCard(s, 0.35, 1.1, 4.3, 4.1, "1A0A0A");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 1.1, w: 4.3, h: 0.05, fill: { color: C.red }, line: { color: C.red } });
  s.addText("❌  ELO  (système classique)", { x: 0.5, y: 1.2, w: 4.0, h: 0.45, fontSize: 15, bold: true, color: C.red, fontFace: "Calibri", align: "left", margin: 0 });
  const eloPoints = ["Un seul chiffre  (ex: 1200)", "Aucune mesure d'incertitude", "Mise à jour fixe  K = 32  par match", "Convergence lente vers le vrai niveau", "Non conçu pour les équipes"];
  s.addText(eloPoints.map((t, i) => ({ text: t, options: { bullet: true, color: i === 0 ? C.white : C.gray, breakLine: i < eloPoints.length - 1 } })), { x: 0.55, y: 1.75, w: 3.9, h: 3.0, fontSize: 13, fontFace: "Calibri", margin: 0, paraSpaceAfter: 8 });
  addCard(s, 5.35, 1.1, 4.3, 4.1, "0A1A0E");
  s.addShape(pres.shapes.RECTANGLE, { x: 5.35, y: 1.1, w: 4.3, h: 0.05, fill: { color: C.green }, line: { color: C.green } });
  s.addText("✅  TrueSkill  (notre solution)", { x: 5.5, y: 1.2, w: 4.0, h: 0.45, fontSize: 15, bold: true, color: C.green, fontFace: "Calibri", align: "left", margin: 0 });
  const tsPoints = ["Deux paramètres : µ (niveau) + σ (incertitude)", "Modélise la confiance dans l'estimation", "Mise à jour proportionnelle à σ²", "Convergence 3-4× plus rapide", "Équipes et matchs nuls natifs"];
  s.addText(tsPoints.map((t, i) => ({ text: t, options: { bullet: true, color: i === 0 ? C.white : C.gray, breakLine: i < tsPoints.length - 1 } })), { x: 5.5, y: 1.75, w: 3.9, h: 3.0, fontSize: 13, fontFace: "Calibri", margin: 0, paraSpaceAfter: 8 });
  s.addShape(pres.shapes.RECTANGLE, { x: 4.72, y: 2.7, w: 0.55, h: 0.3, fill: { color: C.purple }, line: { color: C.purple } });
  s.addText("VS", { x: 4.72, y: 2.7, w: 0.55, h: 0.3, fontSize: 12, bold: true, color: C.white, fontFace: "Calibri", align: "center", margin: 0, valign: "middle" });
}

// ══════════════════════════════════════════════════
// SLIDE 3 — PRINCIPE MATHÉMATIQUE (CORRIGÉ)
// ══════════════════════════════════════════════════
{
  const s = pres.addSlide();
  addBg(s);
  addTitle(s, "TrueSkill : La Gaussienne");

  addCard(s, 0.35, 1.1, 4.55, 4.1, C.bg2);
  s.addText("Chaque joueur = une courbe en cloche", { x: 0.5, y: 1.2, w: 4.2, h: 0.4, fontSize: 13, italic: true, color: C.gray, fontFace: "Calibri", align: "left", margin: 0 });
  const params = [
    { label: "µ  (mu)", desc: "Centre de la courbe = estimation du niveau", color: C.purple },
    { label: "σ  (sigma)", desc: "Largeur de la courbe = incertitude", color: C.cyan },
  ];
  params.forEach((p, i) => {
    const cy = 1.75 + i * 0.85;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: cy, w: 0.06, h: 0.55, fill: { color: p.color }, line: { color: p.color } });
    s.addText(p.label, { x: 0.7, y: cy, w: 3.9, h: 0.28, fontSize: 15, bold: true, color: p.color, fontFace: "Calibri", align: "left", margin: 0 });
    s.addText(p.desc, { x: 0.7, y: cy + 0.28, w: 3.9, h: 0.27, fontSize: 12, color: C.gray, fontFace: "Calibri", align: "left", margin: 0 });
  });

  addCard(s, 0.5, 3.55, 4.1, 1.4, "0A0F1E");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.55, w: 4.1, h: 0.04, fill: { color: C.purple }, line: { color: C.purple } });
  s.addText("Règle des 3 sigma", { x: 0.65, y: 3.62, w: 3.8, h: 0.3, fontSize: 12, bold: true, color: C.purple, fontFace: "Calibri", align: "left", margin: 0 });
  s.addText("µ ± 3σ  contient  99.7%  des valeurs", { x: 0.65, y: 3.95, w: 3.8, h: 0.35, fontSize: 13, color: C.white, fontFace: "Calibri", align: "left", margin: 0 });
  s.addText("→  Score conservateur  =  µ − 3σ", { x: 0.65, y: 4.3, w: 3.8, h: 0.35, fontSize: 13, bold: true, color: C.yellow, fontFace: "Calibri", align: "left", margin: 0 });

  // ─── EXEMPLES CORRIGÉS ───────────────────────────────────
  // Joueur A : µ=35, σ=5  →  Score = 35−15 = 20
  // Joueur B : µ=30, σ=1  →  Score = 30−3  = 27
  // B (27) > A (20)  ✓  B est bien classé devant A
  addCard(s, 5.1, 1.1, 4.55, 4.1, C.bg2);
  s.addText("Exemple concret", { x: 5.25, y: 1.2, w: 4.2, h: 0.35, fontSize: 14, bold: true, color: C.white, fontFace: "Calibri", align: "left", margin: 0 });

  const examples = [
    { label: "Joueur A", mu: "µ = 35,  σ = 5", score: "Score = 35 − 15 = 20", color: C.orange },
    { label: "Joueur B", mu: "µ = 30,  σ = 1", score: "Score = 30 − 3 = 27",  color: C.green  },
  ];
  examples.forEach((ex, i) => {
    const ey = 1.7 + i * 1.5;
    addCard(s, 5.25, ey, 4.2, 1.3, "1E293B");
    s.addShape(pres.shapes.RECTANGLE, { x: 5.25, y: ey, w: 0.07, h: 1.3, fill: { color: ex.color }, line: { color: ex.color } });
    s.addText(ex.label, { x: 5.4, y: ey + 0.05, w: 3.9, h: 0.3, fontSize: 14, bold: true, color: ex.color, fontFace: "Calibri", align: "left", margin: 0 });
    s.addText(ex.mu,    { x: 5.4, y: ey + 0.38, w: 3.9, h: 0.3, fontSize: 13, color: C.gray, fontFace: "Calibri", align: "left", margin: 0 });
    s.addText(ex.score, { x: 5.4, y: ey + 0.72, w: 3.9, h: 0.38, fontSize: 14, bold: true, color: C.white, fontFace: "Calibri", align: "left", margin: 0 });
  });

  s.addText("→  B est classé DEVANT A malgré un µ plus faible,\n   car σ_B (1) << σ_A (5) — B est bien plus certain !", {
    x: 5.25, y: 4.75, w: 4.4, h: 0.65,
    fontSize: 11, italic: true, bold: true, color: C.yellow,
    fontFace: "Calibri", align: "left", margin: 0,
  });
}

// ══════════════════════════════════════════════════
// SLIDE 4 — ARCHITECTURE
// ══════════════════════════════════════════════════
{
  const s = pres.addSlide();
  addBg(s);
  addTitle(s, "Architecture — 8 modules Python");
  const levels = [
    { title: "🎯 Niveau Minimum", color: C.green, files: [
      { name: "simulation.py", desc: "Joueurs + matchs 1v1" },
      { name: "matchmaking.py", desc: "Probabilités de victoire" },
      { name: "visualisation.py", desc: "Graphes PNG" },
      { name: "main.py", desc: "Orchestrateur" },
    ]},
    { title: "🏆 Niveau Bon", color: C.orange, files: [
      { name: "elo.py", desc: "Comparaison ELO" },
      { name: "equipes.py", desc: "Matchs 2v2 + nuls" },
    ]},
    { title: "🚀 Niveau Excellent", color: C.red, files: [
      { name: "dynamique.py", desc: "Saisons + decay σ" },
      { name: "trueskill2.py", desc: "Score composite TS2" },
      { name: "dashboard.py", desc: "Interface Streamlit" },
    ]},
  ];
  const colX = [0.35, 3.6, 6.85];
  const colW = 3.0;
  levels.forEach((lev, ci) => {
    const x = colX[ci];
    const totalH = 0.55 + lev.files.length * 0.75 + 0.2;
    addCard(s, x, 1.05, colW, totalH, C.bg2);
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.05, w: colW, h: 0.04, fill: { color: lev.color }, line: { color: lev.color } });
    s.addText(lev.title, { x: x + 0.1, y: 1.12, w: colW - 0.2, h: 0.38, fontSize: 13, bold: true, color: lev.color, fontFace: "Calibri", align: "left", margin: 0 });
    lev.files.forEach((f, fi) => {
      const fy = 1.62 + fi * 0.76;
      addCard(s, x + 0.12, fy, colW - 0.24, 0.62, "1E293B");
      s.addShape(pres.shapes.RECTANGLE, { x: x + 0.12, y: fy, w: 0.06, h: 0.62, fill: { color: lev.color }, line: { color: lev.color } });
      s.addText(f.name, { x: x + 0.25, y: fy + 0.04, w: colW - 0.45, h: 0.28, fontSize: 12, bold: true, color: C.white, fontFace: "Consolas", align: "left", margin: 0 });
      s.addText(f.desc, { x: x + 0.25, y: fy + 0.32, w: colW - 0.45, h: 0.24, fontSize: 10, color: C.gray, fontFace: "Calibri", align: "left", margin: 0 });
    });
  });
}

// ══════════════════════════════════════════════════
// SLIDE 5 — SIMULATION 1v1
// ══════════════════════════════════════════════════
{
  const s = pres.addSlide();
  addBg(s);
  addTitle(s, "🎯 Niveau Minimum — Simulation 1v1");
  addBadge(s, "Niveau Minimum", 0.4, 0.0, C.green);

  const steps = [
    { label: "Créer 10 joueurs", sub: "µ=25, σ=8.33" },
    { label: "200 matchs 1v1", sub: "aléatoires" },
    { label: "Mise à jour\nTrueSkill", sub: "µ, σ recalculés" },
    { label: "Classement\nµ − 3σ", sub: "score conservateur" },
  ];
  const sw = 2.0, sh = 0.75, sy = 1.15;
  steps.forEach((st, i) => {
    const sx = 0.4 + i * 2.4;
    s.addShape(pres.shapes.RECTANGLE, { x: sx, y: sy, w: sw, h: sh, fill: { color: "1E3A5F" }, line: { color: C.purple, width: 1.5 }, shadow: makeShadow() });
    s.addText(st.label, { x: sx, y: sy + 0.02, w: sw, h: 0.42, fontSize: 12, bold: true, color: C.white, fontFace: "Calibri", align: "center", margin: 0, valign: "middle" });
    s.addText(st.sub,   { x: sx, y: sy + 0.46, w: sw, h: 0.24, fontSize: 9, color: C.gray, fontFace: "Calibri", align: "center", margin: 0 });
    if (i < steps.length - 1) {
      s.addShape(pres.shapes.LINE, { x: sx + sw, y: sy + sh / 2, w: 0.38, h: 0, line: { color: C.purple, width: 2 } });
    }
  });

  addCard(s, 0.4, 2.15, 4.55, 1.05, "0A0F1E");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 2.15, w: 4.55, h: 0.04, fill: { color: C.cyan }, line: { color: C.cyan } });
  s.addText("Formule du résultat du match", { x: 0.55, y: 2.22, w: 4.2, h: 0.3, fontSize: 11, bold: true, color: C.cyan, fontFace: "Calibri", align: "left", margin: 0 });
  s.addText("P(j1 gagne) = 1 / (1 + 10^(-diff / 10))", { x: 0.55, y: 2.52, w: 4.2, h: 0.35, fontSize: 13, bold: true, color: C.white, fontFace: "Consolas", align: "left", margin: 0 });
  s.addText("diff = compétence(j1) - compétence(j2)", { x: 0.55, y: 2.88, w: 4.2, h: 0.26, fontSize: 10, color: C.gray, fontFace: "Consolas", align: "left", margin: 0 });

  addCard(s, 5.1, 2.15, 4.55, 3.1, C.bg2);
  s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y: 2.15, w: 4.55, h: 0.04, fill: { color: C.purple }, line: { color: C.purple } });
  s.addText("Mise à jour après le match", { x: 5.25, y: 2.22, w: 4.2, h: 0.3, fontSize: 12, bold: true, color: C.purple, fontFace: "Calibri", align: "left", margin: 0 });
  const updates = [
    { who: "Vainqueur :", what: "µ monte  ↑    σ diminue  ↓", color: C.green },
    { who: "Perdant :",   what: "µ descend ↓    σ diminue aussi ↓", color: C.red },
  ];
  updates.forEach((u, i) => {
    const uy = 2.65 + i * 0.9;
    addCard(s, 5.25, uy, 4.2, 0.75, "1E293B");
    s.addShape(pres.shapes.RECTANGLE, { x: 5.25, y: uy, w: 0.06, h: 0.75, fill: { color: u.color }, line: { color: u.color } });
    s.addText(u.who,  { x: 5.38, y: uy + 0.04, w: 3.9, h: 0.28, fontSize: 12, bold: true, color: u.color, fontFace: "Calibri", align: "left", margin: 0 });
    s.addText(u.what, { x: 5.38, y: uy + 0.34, w: 3.9, h: 0.28, fontSize: 12, color: C.white, fontFace: "Calibri", align: "left", margin: 0 });
  });
  s.addText("Les deux joueurs apprennent de chaque match", { x: 5.25, y: 4.55, w: 4.2, h: 0.35, fontSize: 11, italic: true, color: C.yellow, fontFace: "Calibri", align: "center", margin: 0 });

  addCard(s, 0.4, 3.3, 4.55, 2.0, C.bg2);
  s.addText("La compétence est CACHÉE", { x: 0.55, y: 3.38, w: 4.2, h: 0.35, fontSize: 12, bold: true, color: C.yellow, fontFace: "Calibri", align: "left", margin: 0 });
  s.addText([
    { text: "TrueSkill ne voit que ", options: { color: C.gray } },
    { text: "\"Joueur A a gagné\"", options: { color: C.white, bold: true } },
    { text: "\nIl estime le niveau par ", options: { color: C.gray } },
    { text: "inférence Bayésienne", options: { color: C.purple, bold: true } },
  ], { x: 0.55, y: 3.8, w: 4.2, h: 0.9, fontSize: 12, fontFace: "Calibri", align: "left", margin: 0 });
}

// ══════════════════════════════════════════════════
// SLIDE 6 — CONVERGENCE (CORRIGÉE)
// ══════════════════════════════════════════════════
{
  const s = pres.addSlide();
  addBg(s);
  addTitle(s, "Convergence de µ et σ — Preuves visuelles");
  addBadge(s, "Niveau Minimum", 0.4, 0.0, C.green);

  // ─── GRAPHE µ (gauche) ──────────────────────────
  addCard(s, 0.35, 1.05, 4.55, 4.2, C.bg2);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 1.05, w: 4.55, h: 0.04, fill: { color: C.purple }, line: { color: C.purple } });
  s.addText("Graphe µ — Convergence", { x: 0.5, y: 1.12, w: 4.2, h: 0.35, fontSize: 13, bold: true, color: C.purple, fontFace: "Calibri", align: "left", margin: 0 });

  // Zone graphique µ
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.55, w: 4.2, h: 2.6, fill: { color: "0A0F1E" }, line: { color: C.border, width: 1 } });
  s.addText("Matchs joués →", { x: 0.5, y: 4.2, w: 4.2, h: 0.22, fontSize: 9, color: C.gray2, fontFace: "Calibri", align: "center", margin: 0 });

  // Ligne de départ µ=25 (milieu du graphe)
  const midY = 2.85;  // milieu de la zone = µ=25 de départ
  s.addShape(pres.shapes.LINE, { x: 0.55, y: midY, w: 4.1, h: 0, line: { color: "334155", width: 0.75, dashType: "dash" } });
  s.addText("µ=25 (départ)", { x: 0.52, y: midY - 0.2, w: 1.0, h: 0.2, fontSize: 7, color: C.gray2, fontFace: "Calibri", align: "left", margin: 0 });

  // Courbes µ : TOUTES partent de midY et divergent vers leur compétence
  // Vers le haut (joueurs forts, compétence > 25)
  // Vers le bas  (joueurs faibles, compétence < 25)
  const muCurves = [
    { endY: 1.70, color: "4ADE80" },   // compétence ≈ 45 → monte fort
    { endY: 2.05, color: "60A5FA" },   // compétence ≈ 37 → monte
    { endY: 2.45, color: "A78BFA" },   // compétence ≈ 29 → monte légèrement
    { endY: 3.30, color: "FBBF24" },   // compétence ≈ 18 → descend
    { endY: 3.90, color: "F87171" },   // compétence ≈ 10 → descend fort
  ];
  muCurves.forEach(c => {
    // Courbe pleine (µ estimé) : part de midY, va vers endY
    s.addShape(pres.shapes.LINE, { x: 0.6, y: midY, w: 3.9, h: c.endY - midY, line: { color: c.color, width: 2.0 } });
    // Ligne pointillée horizontale (vraie compétence cachée) à la hauteur endY
    s.addShape(pres.shapes.LINE, { x: 0.6, y: c.endY, w: 3.9, h: 0, line: { color: c.color, width: 1.0, dashType: "sysDot" } });
  });

  // Point de départ (petit cercle visuel)
  s.addShape(pres.shapes.OVAL, { x: 0.56, y: midY - 0.06, w: 0.12, h: 0.12, fill: { color: C.white }, line: { color: C.white } });
  s.addText("départ\ncommun", { x: 0.62, y: midY - 0.28, w: 0.8, h: 0.28, fontSize: 7, color: C.white, fontFace: "Calibri", align: "left", margin: 0 });

  s.addText([
    { text: "—  µ estimé (part tous de 25)", options: { color: C.purple } },
    { text: "\n·····  Vraie compétence cachée", options: { color: C.gray } },
  ], { x: 0.5, y: 4.46, w: 4.2, h: 0.5, fontSize: 10, fontFace: "Calibri", align: "left", margin: 0 });

  // ─── GRAPHE σ (droite) ──────────────────────────
  addCard(s, 5.1, 1.05, 4.55, 4.2, C.bg2);
  s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y: 1.05, w: 4.55, h: 0.04, fill: { color: C.cyan }, line: { color: C.cyan } });
  s.addText("Graphe σ — Certitude croissante", { x: 5.25, y: 1.12, w: 4.2, h: 0.35, fontSize: 13, bold: true, color: C.cyan, fontFace: "Calibri", align: "left", margin: 0 });

  s.addShape(pres.shapes.RECTANGLE, { x: 5.25, y: 1.55, w: 4.2, h: 2.6, fill: { color: "0A0F1E" }, line: { color: C.border, width: 1 } });
  s.addText("Matchs joués →", { x: 5.25, y: 4.2, w: 4.2, h: 0.22, fontSize: 9, color: C.gray2, fontFace: "Calibri", align: "center", margin: 0 });

  // σ=8.33 au départ (haut du graphe)
  s.addShape(pres.shapes.LINE, { x: 5.3, y: 1.63, w: 4.1, h: 0, line: { color: C.border, width: 0.75, dashType: "dash" } });
  s.addText("σ = 8.33 (départ)", { x: 5.27, y: 1.53, w: 1.3, h: 0.2, fontSize: 7, color: C.gray2, fontFace: "Calibri", align: "left", margin: 0 });

  // Toutes les courbes σ partent du haut et descendent (légèrement décalées)
  const sigmaCurves = ["4ADE80", "60A5FA", "F87171", "FBBF24", "A78BFA"];
  sigmaCurves.forEach((color, i) => {
    const offset = i * 0.04;
    const startSigmaY = 1.64 + offset;
    const endSigmaY = 3.82 + offset * 0.3;
    s.addShape(pres.shapes.LINE, { x: 5.3, y: startSigmaY, w: 4.0, h: endSigmaY - startSigmaY, line: { color, width: 1.8 } });
  });

  // Ligne σ→0 en bas
  s.addShape(pres.shapes.LINE, { x: 5.3, y: 4.02, w: 4.0, h: 0, line: { color: C.cyan, width: 1, dashType: "dash" } });
  s.addText("σ → 0  (certitude maximale)", { x: 5.27, y: 3.92, w: 2.5, h: 0.2, fontSize: 7, color: C.cyan, fontFace: "Calibri", align: "left", margin: 0 });

  s.addText([
    { text: "«  µ converge sans jamais voir le vrai niveau\n", options: { color: C.white, bold: true } },
    { text: "  C'est l'inférence Bayésienne  »", options: { color: C.gray, italic: true } },
  ], { x: 5.25, y: 4.46, w: 4.2, h: 0.6, fontSize: 10, fontFace: "Calibri", align: "left", margin: 0 });
}

// ══════════════════════════════════════════════════
// SLIDE 7 — ELO vs TRUESKILL
// ══════════════════════════════════════════════════
{
  const s = pres.addSlide();
  addBg(s);
  addTitle(s, "🏆 Niveau Bon — ELO vs TrueSkill");
  addBadge(s, "Niveau Bon", 0.4, 0.0, C.orange);
  const rows = [
    [
      { text: "Critère",      options: { bold: true, color: C.white, fill: { color: "1E293B" } } },
      { text: "ELO",          options: { bold: true, color: C.red,   fill: { color: "2A1010" } } },
      { text: "TrueSkill",    options: { bold: true, color: C.green, fill: { color: "0A1A0E" } } },
    ],
    [{ text: "Variables",    options: { color: C.gray,  fill: { color: C.bg3 } } }, { text: "1  (elo)",           options: { color: C.white, fill: { color: C.bg3 } } }, { text: "2  (µ, σ)",             options: { color: C.white, fill: { color: C.bg3 } } }],
    [{ text: "Mise à jour",  options: { color: C.gray,  fill: { color: C.bg2 } } }, { text: "K = 32 fixe",        options: { color: C.white, fill: { color: C.bg2 } } }, { text: "Proportionnel à σ²",    options: { color: C.white, fill: { color: C.bg2 } } }],
    [{ text: "Incertitude",  options: { color: C.gray,  fill: { color: C.bg3 } } }, { text: "Non modélisée ❌",    options: { color: C.red,   fill: { color: C.bg3 } } }, { text: "σ explicite ✅",         options: { color: C.green, fill: { color: C.bg3 } } }],
    [{ text: "Équipes",      options: { color: C.gray,  fill: { color: C.bg2 } } }, { text: "Non natif ❌",        options: { color: C.red,   fill: { color: C.bg2 } } }, { text: "Natif ✅",               options: { color: C.green, fill: { color: C.bg2 } } }],
    [{ text: "Match nul",    options: { color: C.gray,  fill: { color: C.bg3 } } }, { text: "Résultat = 0.5",      options: { color: C.white, fill: { color: C.bg3 } } }, { text: "draw_probability",       options: { color: C.white, fill: { color: C.bg3 } } }],
    [{ text: "Convergence",  options: { color: C.gray,  fill: { color: C.bg2 } } }, { text: "Lente",               options: { color: C.red,   fill: { color: C.bg2 } } }, { text: "3-4× plus rapide ✅",    options: { color: C.green, fill: { color: C.bg2 } } }],
  ];
  s.addTable(rows, { x: 0.4, y: 1.05, w: 9.2, h: 3.8, fontSize: 13, fontFace: "Calibri", border: { pt: 1, color: C.border }, colW: [2.3, 3.45, 3.45], rowH: 0.5, align: "center", valign: "middle" });
  addCard(s, 0.4, 5.0, 9.2, 0.5, "0A0F1E");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 5.0, w: 0.06, h: 0.5, fill: { color: C.yellow }, line: { color: C.yellow } });
  s.addText("Les deux convergent vers le même classement, mais TrueSkill dit AVEC QUELLE CERTITUDE", { x: 0.55, y: 5.06, w: 8.9, h: 0.38, fontSize: 12, italic: true, color: C.yellow, fontFace: "Calibri", align: "left", margin: 0, valign: "middle" });
}

// ══════════════════════════════════════════════════
// SLIDE 8 — ÉQUIPES
// ══════════════════════════════════════════════════
{
  const s = pres.addSlide();
  addBg(s);
  addTitle(s, "🏆 Matchs par Équipes — 2v2 avec Match Nul");
  addBadge(s, "Niveau Bon", 0.4, 0.0, C.orange);

  addCard(s, 0.35, 1.05, 4.55, 2.3, C.bg2);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 1.05, w: 4.55, h: 0.04, fill: { color: C.orange }, line: { color: C.orange } });
  s.addText("Force d'une équipe", { x: 0.5, y: 1.12, w: 4.2, h: 0.35, fontSize: 13, bold: true, color: C.orange, fontFace: "Calibri", align: "left", margin: 0 });
  s.addText("µ_équipe  =  µ₁ + µ₂ + ...", { x: 0.5, y: 1.55, w: 4.2, h: 0.35, fontSize: 14, bold: true, color: C.white, fontFace: "Consolas", align: "left", margin: 0 });
  s.addText("Somme des niveaux individuels", { x: 0.5, y: 1.88, w: 4.2, h: 0.25, fontSize: 10, color: C.gray, fontFace: "Calibri", align: "left", margin: 0 });
  s.addText("σ_équipe  =  √(σ₁² + σ₂² + ...)", { x: 0.5, y: 2.18, w: 4.2, h: 0.35, fontSize: 14, bold: true, color: C.white, fontFace: "Consolas", align: "left", margin: 0 });
  s.addText("Racine de la somme des variances", { x: 0.5, y: 2.52, w: 4.2, h: 0.25, fontSize: 10, color: C.gray, fontFace: "Calibri", align: "left", margin: 0 });

  addCard(s, 5.1, 1.05, 4.55, 2.3, C.bg2);
  s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y: 1.05, w: 4.55, h: 0.04, fill: { color: C.cyan }, line: { color: C.cyan } });
  s.addText("Probabilité de match nul", { x: 5.25, y: 1.12, w: 4.2, h: 0.35, fontSize: 13, bold: true, color: C.cyan, fontFace: "Calibri", align: "left", margin: 0 });
  s.addText("proximite  =  1 - |prob_e1 - 0.5| × 2", { x: 5.25, y: 1.55, w: 4.2, h: 0.32, fontSize: 12, bold: true, color: C.white, fontFace: "Consolas", align: "left", margin: 0 });
  s.addText("prob_nul  =  draw_prob × proximite", { x: 5.25, y: 1.9, w: 4.2, h: 0.32, fontSize: 12, bold: true, color: C.white, fontFace: "Consolas", align: "left", margin: 0 });
  s.addText("Plus les équipes sont proches → plus le nul est probable", { x: 5.25, y: 2.3, w: 4.2, h: 0.35, fontSize: 11, italic: true, color: C.yellow, fontFace: "Calibri", align: "left", margin: 0 });

  addCard(s, 0.35, 3.5, 9.3, 1.85, "0A0F1E");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 3.5, w: 9.3, h: 0.04, fill: { color: C.purple }, line: { color: C.purple } });
  s.addText("Algorithme Snake Draft — Équipes équilibrées", { x: 0.5, y: 3.57, w: 9.0, h: 0.35, fontSize: 13, bold: true, color: C.purple, fontFace: "Calibri", align: "left", margin: 0 });
  s.addText("Trier [A, B, C, D, E, F] par µ décroissant (A = meilleur)", { x: 0.5, y: 3.95, w: 9.0, h: 0.3, fontSize: 12, color: C.gray, fontFace: "Calibri", align: "left", margin: 0 });
  const players = ["A", "B", "C", "D", "E", "F"];
  const teams = [C.orange, C.cyan, C.orange, C.cyan, C.orange, C.cyan];
  players.forEach((p, i) => {
    const px = 0.55 + i * 1.45;
    s.addShape(pres.shapes.RECTANGLE, { x: px, y: 4.3, w: 1.15, h: 0.45, fill: { color: teams[i] === C.orange ? "2A1500" : "001A2A" }, line: { color: teams[i], width: 1.5 } });
    s.addText(p, { x: px, y: 4.3, w: 1.15, h: 0.45, fontSize: 18, bold: true, color: teams[i], fontFace: "Calibri", align: "center", margin: 0, valign: "middle" });
  });
  s.addText("E1 = [A, C, E]  (orange)        E2 = [B, D, F]  (bleu)", { x: 0.5, y: 4.85, w: 9.0, h: 0.3, fontSize: 11, color: C.gray, fontFace: "Calibri", align: "center", margin: 0 });
}

// ══════════════════════════════════════════════════
// SLIDE 9 — SAISONS
// ══════════════════════════════════════════════════
{
  const s = pres.addSlide();
  addBg(s);
  addTitle(s, "🚀 Niveau Excellent — Dynamique Temporelle");
  addBadge(s, "Niveau Excellent", 0.4, 0.0, C.red);

  const seasons = ["Saison 1\n70 matchs", "DECAY\nσ × 1.15", "Saison 2\n70 matchs", "DECAY\nσ × 1.15", "Saison 3\n70 matchs"];
  const seasonColors = [C.purple, C.yellow, C.purple, C.yellow, C.purple];
  seasons.forEach((label, i) => {
    const sx = 0.35 + i * 1.9;
    const isDecay = label.startsWith("DECAY");
    s.addShape(pres.shapes.RECTANGLE, { x: sx, y: 1.15, w: 1.7, h: 0.8, fill: { color: isDecay ? "1A1200" : "0A0F2A" }, line: { color: seasonColors[i], width: isDecay ? 2 : 1.5 } });
    s.addText(label, { x: sx, y: 1.15, w: 1.7, h: 0.8, fontSize: 12, bold: isDecay, color: seasonColors[i], fontFace: "Calibri", align: "center", margin: 0, valign: "middle" });
    if (i < seasons.length - 1) {
      s.addShape(pres.shapes.LINE, { x: sx + 1.7, y: 1.55, w: 0.2, h: 0, line: { color: C.border, width: 1.5 } });
    }
  });

  addCard(s, 0.35, 2.15, 4.55, 3.2, C.bg2);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 2.15, w: 4.55, h: 0.04, fill: { color: C.yellow }, line: { color: C.yellow } });
  s.addText("Skill Decay — Inter-saison", { x: 0.5, y: 2.22, w: 4.2, h: 0.35, fontSize: 13, bold: true, color: C.yellow, fontFace: "Calibri", align: "left", margin: 0 });
  const decayPoints = ["σ augmente de 15%  →  σ_new = σ × 1.15", "µ reste inchangé  →  toujours le même niveau estimé", "Le système est moins certain → recalibrage", "Les compétences cachées évoluent aussi\n(random.gauss(µ=0, σ=2))"];
  s.addText(decayPoints.map((t, i) => ({ text: t, options: { bullet: true, color: i < 2 ? C.white : C.gray, breakLine: i < decayPoints.length - 1 } })), { x: 0.5, y: 2.65, w: 4.2, h: 2.4, fontSize: 12, fontFace: "Calibri", margin: 0, paraSpaceAfter: 10 });

  addCard(s, 5.1, 2.15, 4.55, 3.2, C.bg2);
  s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y: 2.15, w: 4.55, h: 0.04, fill: { color: C.red }, line: { color: C.red } });
  s.addText("Analogie Jeux Vidéo", { x: 5.25, y: 2.22, w: 4.2, h: 0.35, fontSize: 13, bold: true, color: C.red, fontFace: "Calibri", align: "left", margin: 0 });
  const games = [
    { game: "League of Legends", detail: "Reset de rang entre saisons" },
    { game: "Valorant",          detail: "Réévaluation des rangs" },
    { game: "Xbox Live",         detail: "Système TrueSkill original (2006)" },
  ];
  games.forEach((g, i) => {
    const gy = 2.72 + i * 0.8;
    addCard(s, 5.25, gy, 4.2, 0.65, "1E293B");
    s.addShape(pres.shapes.RECTANGLE, { x: 5.25, y: gy, w: 0.06, h: 0.65, fill: { color: C.red }, line: { color: C.red } });
    s.addText(g.game,   { x: 5.38, y: gy + 0.04, w: 3.9, h: 0.28, fontSize: 12, bold: true, color: C.white, fontFace: "Calibri", align: "left", margin: 0 });
    s.addText(g.detail, { x: 5.38, y: gy + 0.33, w: 3.9, h: 0.25, fontSize: 10, color: C.gray, fontFace: "Calibri", align: "left", margin: 0 });
  });
}

// ══════════════════════════════════════════════════
// SLIDE 10 — TRUESKILL 2 (CORRIGÉE)
// ══════════════════════════════════════════════════
{
  const s = pres.addSlide();
  addBg(s);
  addTitle(s, "🚀 TrueSkill 2 — Score Composite  (Microsoft 2018)");
  addBadge(s, "Niveau Excellent", 0.4, 0.0, C.red);

  const comps = [
    { num: "1", title: "Score TrueSkill 1", formula: "µ − 3σ",                  desc: "Base du classement conservateur",              color: C.purple },
    { num: "2", title: "Consistance",       formula: "nb_victoires / nb_matchs", desc: "Taux de victoire  (0 = perd tout, 1 = gagne tout)", color: C.cyan   },
    { num: "3", title: "Facteur d'activité",formula: "1 - e^(-n / 15)",          desc: "Confiance liée au nombre de matchs joués",      color: C.orange },
  ];
  comps.forEach((c, i) => {
    const cx = 0.35 + i * 3.22;
    addCard(s, cx, 1.05, 3.0, 2.35, C.bg2);
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: 1.05, w: 3.0, h: 0.04, fill: { color: c.color }, line: { color: c.color } });
    s.addShape(pres.shapes.OVAL, { x: cx + 0.1, y: 1.12, w: 0.38, h: 0.38, fill: { color: c.color }, line: { color: c.color } });
    s.addText(c.num, { x: cx + 0.1, y: 1.12, w: 0.38, h: 0.38, fontSize: 14, bold: true, color: C.white, fontFace: "Calibri", align: "center", margin: 0, valign: "middle" });
    s.addText(c.title,   { x: cx + 0.55, y: 1.13, w: 2.35, h: 0.38, fontSize: 12, bold: true, color: c.color, fontFace: "Calibri", align: "left", margin: 0 });
    s.addText(c.formula, { x: cx + 0.1,  y: 1.6,  w: 2.8,  h: 0.38, fontSize: 13, bold: true, color: C.white, fontFace: "Consolas", align: "center", margin: 0 });
    s.addText(c.desc,    { x: cx + 0.1,  y: 2.0,  w: 2.8,  h: 0.35, fontSize: 10, color: C.gray, fontFace: "Calibri", align: "center", margin: 0 });
  });

  addCard(s, 0.35, 3.55, 9.3, 0.85, "0A0F1E");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 3.55, w: 9.3, h: 0.04, fill: { color: C.yellow }, line: { color: C.yellow } });
  s.addText("Score TS2  =  (µ − 3σ)  ×  (0.4 + 0.6 × consistance)  ×  (0.4 + 0.6 × activité)", {
    x: 0.5, y: 3.65, w: 9.0, h: 0.65,
    fontSize: 15, bold: true, color: C.yellow, fontFace: "Consolas", align: "center", margin: 0, valign: "middle",
  });

  // ─── EXEMPLE CORRIGÉ ──────────────────────────────────────────
  // Même nb_matchs = 30 pour les deux → activité identique = 0.865
  // Joueur A : µ=35, σ=2, 80% victoires, 30 matchs
  //   TS1 = 29   TS2 = 29 × 0.88 × 0.919 ≈ 23.5  ✓
  // Joueur B : µ=34, σ=2, 30% victoires, 30 matchs
  //   TS1 = 28   TS2 = 28 × 0.58 × 0.919 ≈ 14.9
  addCard(s, 0.35, 4.5, 9.3, 1.0, C.bg2);
  const tsExamples = [
    { label: "Joueur A", detail: "µ=35, σ=2, 80% victoires, 30 matchs", ts1: "TS1 = 29", ts2: "TS2 ≈ 23.5", c: C.green },
    { label: "Joueur B", detail: "µ=34, σ=2, 30% victoires, 30 matchs", ts1: "TS1 = 28", ts2: "TS2 ≈ 14.9", c: C.red   },
  ];
  tsExamples.forEach((ex, i) => {
    const ex_x = 0.5 + i * 4.5;
    s.addText(ex.label + "  ·  " + ex.detail, { x: ex_x, y: 4.58, w: 4.2, h: 0.28, fontSize: 11, color: C.gray, fontFace: "Calibri", align: "left", margin: 0 });
    s.addText(ex.ts1 + "   →   " + ex.ts2,    { x: ex_x, y: 4.88, w: 4.2, h: 0.35, fontSize: 14, bold: true, color: ex.c, fontFace: "Consolas", align: "left", margin: 0 });
  });
  s.addText("A classé DEVANT B malgré des µ quasi-identiques — la consistance fait la différence !", {
    x: 0.35, y: 5.12, w: 9.3, h: 0.3,
    fontSize: 11, italic: true, bold: true, color: C.yellow, fontFace: "Calibri", align: "center", margin: 0,
  });
}

// ══════════════════════════════════════════════════
// SLIDE 11 — DASHBOARD
// ══════════════════════════════════════════════════
{
  const s = pres.addSlide();
  addBg(s);
  addTitle(s, "Interface Graphique — Dashboard Streamlit");
  const tabs = [
    { icon: "🎯", title: "Niveau Minimum", desc: "Graphes de convergence µ/σ interactifs (Plotly)", color: C.green },
    { icon: "🏆", title: "Niveau Bon",     desc: "Comparaison ELO vs TrueSkill + matchs d'équipes", color: C.orange },
    { icon: "🚀", title: "Niveau Excellent", desc: "Saisons + heatmap des classements inter-saisons", color: C.red },
    { icon: "🎮", title: "Matchmaking Live", desc: "Sélectionner 2 joueurs → proba en temps réel", color: C.cyan },
  ];
  tabs.forEach((tab, i) => {
    const tx = 0.35 + (i % 2) * 4.85;
    const ty = 1.05 + Math.floor(i / 2) * 1.55;
    addCard(s, tx, ty, 4.6, 1.35, C.bg2);
    s.addShape(pres.shapes.RECTANGLE, { x: tx, y: ty, w: 4.6, h: 0.04, fill: { color: tab.color }, line: { color: tab.color } });
    s.addText(tab.icon + "  " + tab.title, { x: tx + 0.15, y: ty + 0.1, w: 4.3, h: 0.38, fontSize: 14, bold: true, color: tab.color, fontFace: "Calibri", align: "left", margin: 0 });
    s.addText(tab.desc, { x: tx + 0.15, y: ty + 0.52, w: 4.3, h: 0.65, fontSize: 12, color: C.gray, fontFace: "Calibri", align: "left", margin: 0 });
  });
  addCard(s, 0.35, 4.25, 9.3, 1.2, "0A0F1E");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 4.25, w: 9.3, h: 0.04, fill: { color: C.purple }, line: { color: C.purple } });
  s.addText("Sidebar interactive", { x: 0.5, y: 4.32, w: 3.0, h: 0.3, fontSize: 12, bold: true, color: C.purple, fontFace: "Calibri", align: "left", margin: 0 });
  const features = ["Sliders : nb_joueurs, nb_matchs", "Graine aléatoire (reproductible)", "@st.cache_data : évite de relancer si params inchangés", "Plotly : graphes zoomables avec survol"];
  s.addText(features.map((f, i) => ({ text: f, options: { bullet: true, color: C.gray, breakLine: i < features.length - 1 } })), { x: 0.5, y: 4.68, w: 9.0, h: 0.65, fontSize: 11, fontFace: "Calibri", margin: 0 });
}

// ══════════════════════════════════════════════════
// SLIDE 12 — COMPLEXITÉ & TECHNOLOGIES
// ══════════════════════════════════════════════════
{
  const s = pres.addSlide();
  addBg(s);
  addTitle(s, "Technologies & Complexité Algorithmique");
  const compRows = [
    [{ text: "Opération",              options: { bold: true, color: C.white, fill: { color: "1E293B" } } }, { text: "Complexité", options: { bold: true, color: C.white, fill: { color: "1E293B" } } }, { text: "Explication", options: { bold: true, color: C.white, fill: { color: "1E293B" } } }],
    [{ text: "Simulation N matchs",    options: { color: C.white, fill: { color: C.bg3 } } }, { text: "O(N)",       options: { bold: true, color: C.green,  fill: { color: C.bg3 } } }, { text: "N appels à simuler_match()",             options: { color: C.gray, fill: { color: C.bg3 } } }],
    [{ text: "Classement final",       options: { color: C.white, fill: { color: C.bg2 } } }, { text: "O(n log n)", options: { bold: true, color: C.yellow, fill: { color: C.bg2 } } }, { text: "sorted() = tri rapide",                  options: { color: C.gray, fill: { color: C.bg2 } } }],
    [{ text: "Matchmaking intelligent",options: { color: C.white, fill: { color: C.bg3 } } }, { text: "O(N×n)",     options: { bold: true, color: C.orange, fill: { color: C.bg3 } } }, { text: "Recherche adversaire à chaque match",     options: { color: C.gray, fill: { color: C.bg3 } } }],
    [{ text: "Saisons complètes",      options: { color: C.white, fill: { color: C.bg2 } } }, { text: "O(S×N)",     options: { bold: true, color: C.cyan,   fill: { color: C.bg2 } } }, { text: "S saisons × N matchs/saison",            options: { color: C.gray, fill: { color: C.bg2 } } }],
    [{ text: "Programme complet",      options: { color: C.white, fill: { color: "0A1A0E" } } }, { text: "O(N)",    options: { bold: true, color: C.green,  fill: { color: "0A1A0E" } } }, { text: "< 1 seconde pour 200 matchs / 10 joueurs", options: { color: C.green, fill: { color: "0A1A0E" } } }],
  ];
  s.addTable(compRows, { x: 0.35, y: 1.05, w: 9.3, h: 3.0, fontSize: 12, fontFace: "Calibri", border: { pt: 1, color: C.border }, colW: [2.8, 1.6, 4.9], rowH: 0.44, align: "center", valign: "middle" });
  addCard(s, 0.35, 4.2, 9.3, 1.25, C.bg2);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 4.2, w: 9.3, h: 0.04, fill: { color: C.purple }, line: { color: C.purple } });
  s.addText("Stack technique", { x: 0.5, y: 4.27, w: 9.0, h: 0.3, fontSize: 12, bold: true, color: C.purple, fontFace: "Calibri", align: "left", margin: 0 });
  const techs = [
    { name: "Python", color: C.yellow }, { name: "trueskill", color: C.purple }, { name: "matplotlib", color: C.orange },
    { name: "pandas", color: C.cyan }, { name: "numpy", color: C.green }, { name: "streamlit", color: C.red }, { name: "plotly", color: "A78BFA" },
  ];
  techs.forEach((t, i) => {
    const tx = 0.5 + i * 1.32;
    s.addShape(pres.shapes.RECTANGLE, { x: tx, y: 4.67, w: 1.18, h: 0.52, fill: { color: "1E293B" }, line: { color: t.color, width: 1.5 } });
    s.addText(t.name, { x: tx, y: 4.67, w: 1.18, h: 0.52, fontSize: 11, bold: true, color: t.color, fontFace: "Calibri", align: "center", margin: 0, valign: "middle" });
  });
}

// ══════════════════════════════════════════════════
// SLIDE 13 — CONCLUSION
// ══════════════════════════════════════════════════
{
  const s = pres.addSlide();
  addBg(s, C.bg);
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.12, fill: { color: C.purple }, line: { color: C.purple } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.505, w: 10, h: 0.12, fill: { color: C.purple2 }, line: { color: C.purple2 } });
  addTitle(s, "Bilan & Perspectives");

  addCard(s, 0.35, 1.05, 4.55, 3.5, C.bg2);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 1.05, w: 4.55, h: 0.04, fill: { color: C.green }, line: { color: C.green } });
  s.addText("Ce qu'on a démontré", { x: 0.5, y: 1.12, w: 4.2, h: 0.35, fontSize: 13, bold: true, color: C.green, fontFace: "Calibri", align: "left", margin: 0 });
  const bilan = ["Convergence Bayésienne sans voir le vrai niveau", "TrueSkill surpasse ELO en richesse d'information", "Équipes et matchs nuls gérés nativement", "Modélisation temporelle (decay inter-saison)", "TrueSkill 2 : classification plus discriminante"];
  s.addText(bilan.map((t, i) => ({ text: "✓  " + t, options: { color: i < 2 ? C.white : C.gray, breakLine: i < bilan.length - 1 } })), { x: 0.5, y: 1.6, w: 4.2, h: 2.7, fontSize: 12, fontFace: "Calibri", margin: 0, paraSpaceAfter: 10, bullet: false });

  addCard(s, 5.1, 1.05, 4.55, 3.5, C.bg2);
  s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y: 1.05, w: 4.55, h: 0.04, fill: { color: C.cyan }, line: { color: C.cyan } });
  s.addText("Perspectives", { x: 5.25, y: 1.12, w: 4.2, h: 0.35, fontSize: 13, bold: true, color: C.cyan, fontFace: "Calibri", align: "left", margin: 0 });
  const perspectives = ["Données réelles — Chess.com, League of Legends", "Implémenter l'algorithme EP manuellement", "Mesurer la vitesse de convergence TS1 vs TS2", "Modéliser les comportements (AFK, smurfs)"];
  s.addText(perspectives.map((t, i) => ({ text: "→  " + t, options: { color: i === 0 ? C.white : C.gray, breakLine: i < perspectives.length - 1 } })), { x: 5.25, y: 1.6, w: 4.2, h: 2.7, fontSize: 12, fontFace: "Calibri", margin: 0, paraSpaceAfter: 10, bullet: false });

  addCard(s, 0.35, 4.7, 9.3, 0.75, "0A0F2A");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 4.7, w: 0.08, h: 0.75, fill: { color: C.purple }, line: { color: C.purple } });
  s.addText("«  TrueSkill : là où ELO donne un chiffre, Bayes donne une certitude  »", { x: 0.55, y: 4.76, w: 9.0, h: 0.55, fontSize: 15, bold: true, italic: true, color: C.purple, fontFace: "Calibri", align: "center", margin: 0, valign: "middle" });
}

// ══════════════════════════════════════════════════
// SAUVEGARDE
// ══════════════════════════════════════════════════
const outputPath = "/Users/jeanhaj/Desktop/ING4/Intelligence Artificiel/groupe-02-trueskill-matchmaking/TrueSkill_Presentation_ECE_ING4.pptx";
pres.writeFile({ fileName: outputPath })
  .then(() => console.log("✅ Présentation corrigée : " + outputPath))
  .catch(err => console.error("❌ Erreur :", err));
