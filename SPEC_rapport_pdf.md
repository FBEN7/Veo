# SPEC — Rapport post-match PDF (projet match-report)

Document de spécifications à destination d'un assistant de code (GitHub Copilot).
Objectif : générer un module `src/pdf_report.py` produisant un rapport PDF
de qualité commerciale, utilisable en démo de vente auprès de clubs amateurs belges.

---

## 1. Contexte et contraintes

- **Input** : les données produites par le pipeline existant, exclusivement :
  `output/tracks_teams.parquet` (tracks projetés terrain) et les DataFrames/dicts
  retournés par `src/stats.py` (`physical_stats`, `field_tilt`, `possession_proxy`,
  `ball_detection_rate`).
- **Output** : `output/report_{date}_{club}.pdf`, format A4 portrait, 5-6 pages.
- **Stack imposé** : Python. Recommandé : **Jinja2 + HTML/CSS + WeasyPrint**
  (réutilise l'approche template existante ; CSS print bien supporté).
  Alternative acceptable : ReportLab si WeasyPrint pose problème d'installation.
- Graphiques générés avec **matplotlib + mplsoccer**, exportés en **SVG**
  (vectoriel, net à l'impression), intégrés dans le HTML.
- **Règle d'or** : le rapport ne contient QUE des métriques réellement calculées
  par le pipeline. Aucun placeholder de métrique non implémentée (pas de xG,
  pas de passes, pas de tirs). Un rapport de vente qui promet ce que le produit
  ne fait pas est un piège commercial.

## 2. Design system

- **Couleurs configurables par club** via `config/branding.yaml` :
  `primary` (défaut `#1B4332`), `accent` (défaut `#D4A017`), `dark` (`#0D1B2A`),
  `light_bg` (`#F7F7F5`). Le vert terrain des pitchs : `#2d5a27`.
- **Typographie** : une seule famille, `Inter` (fallback `Helvetica, Arial`).
  Titres : bold, 20-24pt. Corps : 9.5-10.5pt. Chiffres de KPI : 28-34pt bold.
  Jamais plus de 2 graisses par page.
- **Grille** : marges 18mm, gouttières 6mm, grille 12 colonnes.
- **Éléments récurrents** :
  - En-tête de page (sauf couverture) : logo club (si fourni dans branding.yaml,
    sinon nom du club en texte), libellé du match, numéro de page.
  - Pied de page : "Généré par [nom produit] — données issues d'analyse vidéo
    automatisée" + date de génération.
- **Interdits** : ombres portées lourdes, dégradés criards, plus de 4 couleurs
  par graphique, camemberts (utiliser des barres), 3D.
- Tous les graphiques partagent le même style matplotlib (un module
  `src/chart_style.py` centralise rcParams : fond transparent, spines top/right
  supprimées, grille horizontale légère `#E0E0E0`).

## 3. Structure page par page

### Page 1 — Couverture
- Bandeau plein `primary` sur le tiers supérieur : "RAPPORT POST-MATCH".
- Libellé du match (équipes), date, compétition — saisis via CLI `--label`,
  `--date`, `--competition`.
- Mini-pitch décoratif (mplsoccer, lignes seules) en filigrane bas de page.
- Pas de score : le pipeline ne le connaît pas. Champ optionnel `--score`
  affiché uniquement s'il est fourni.

### Page 2 — Synthèse exécutive
- **4 à 6 cartes KPI** (2 rangées max) : distance totale par équipe (km),
  sprints par équipe, field tilt, possession* (uniquement si
  `ball_detection_rate >= 0.3`, sinon la carte est remplacée par une carte
  "Occupation du terrain").
- Chaque carte : chiffre géant, libellé, et une ligne de contexte
  (ex. "team_A a couvert 4,2 km de plus").
- **Bloc "3 enseignements"** : 3 phrases générées par règles simples à partir
  des données (ex. si field tilt > 5m d'écart → "Le jeu s'est majoritairement
  déroulé dans la moitié de terrain de X"). Logique dans
  `src/insights.py`, purement déterministe, max 5 règles.
- Astérisque obligatoire sur la possession : "proxy : joueur le plus proche
  du ballon".

### Page 3 — Occupation du terrain
- Deux heatmaps côte à côte (team_A / team_B), pitch complet, cmap unique
  (`hot` ou custom dérivé de `primary`), même échelle.
- En dessous : **timeline du field tilt** (ligne, x = minutes, y = position
  moyenne en X des deux équipes, fenêtre glissante 5 min). Zones ombrées
  quand une équipe domine territorialement.

### Page 4 — Données physiques
- Tableau par track : équipe, minutes suivies, distance (m), vitesse max,
  sprints. Trié par distance décroissante, zébrage léger, max 20 lignes
  (le reste agrégé en "autres tracks").
- **Encadré d'honnêteté obligatoire** (fond `light_bg`, bord gauche `accent`) :
  "Un joueur peut apparaître sous plusieurs tracks (interruptions de suivi).
  Les distances sont des minimums par segment." Ce texte est non négociable :
  en démo de vente, l'honnêteté méthodologique est un argument, pas une faiblesse.
- Histogramme des vitesses (toutes détections confondues, bins de 2 km/h),
  seuil sprint (20 km/h) marqué d'une ligne verticale `accent`.

### Page 5 — Physionomie par période
- Découpage en 2 mi-temps (paramètre `--halftime-min`, défaut 45+temps réel
  détecté ; à défaut, moitié de la durée vidéo).
- Heatmaps par mi-temps (grille 2×2 : équipe × période).
- Barres comparées : distance et sprints par mi-temps et par équipe
  (détecter une baisse physique en 2e mi-temps est un argument de vente fort
  pour les coachs).

### Page 6 — Méthodologie & qualité des données
- Taux de détection du ballon, nombre de tracks, durée analysée,
  résolution vidéo, modèle utilisé.
- 4-5 phrases expliquant le pipeline en langage non technique.
- Mention des limites (caméra fixe requise, pas d'événements de jeu).

## 4. Contrat de données (découplage)

Le module PDF ne lit PAS les parquets directement. `main.py` assemble un
`report_payload: dict` (schéma ci-dessous) et le passe à
`pdf_report.build(payload, branding)`. Cela permet de tester le PDF avec des
données factices (`tests/fixtures/sample_payload.json`).

```json
{
  "meta": {"label": str, "date": str, "competition": str, "score": str|null,
            "video_duration_min": float, "model": str},
  "quality": {"ball_detection_rate": float, "n_tracks": int},
  "kpis": {"distance_km": {"team_A": float, "team_B": float},
            "sprints": {"team_A": int, "team_B": int},
            "field_tilt": {"team_A": float, "team_B": float},
            "possession": {"team_A": float|null, "team_B": float|null}},
  "physical": [{"track_id": int, "team": str, "minutes_tracked": float,
                 "distance_m": float, "top_speed_kmh": float, "n_sprints": int}],
  "tilt_timeline": [{"minute": float, "team_A_x": float, "team_B_x": float}],
  "halves": {"first": {...mêmes agrégats...}, "second": {...}},
  "heatmap_points": {"team_A": [[x, y]], "team_B": [[x, y]]}
}
```

## 5. Critères d'acceptation

1. `python main.py data/match.mp4 --pdf` produit le PDF sans erreur ;
   `pytest tests/test_pdf.py` génère un PDF depuis le payload factice.
2. PDF ≤ 8 Mo, graphiques vectoriels (zoomer à 400% : aucun texte pixelisé).
3. Aucune métrique affichée qui ne soit pas dans le payload ; possession
   masquée automatiquement si `ball_detection_rate < 0.3`.
4. Changer `branding.yaml` recolore l'ensemble du rapport sans toucher au code.
5. Les noms d'équipes réels (`--team-a "RFC X" --team-b "US Y"`) remplacent
   team_A/team_B partout.
6. L'encadré méthodologique de la page 4 est présent (test automatisé sur le
   texte extrait du PDF).
7. Impression N&B lisible : les deux équipes restent distinguables
   (utiliser aussi des motifs/markers différents, pas seulement la couleur).

## 6. Hors périmètre (ne pas générer)

- Événements de jeu (passes, tirs, duels, xG, corners).
- Stats individuelles nominatives (les tracks ne sont pas des joueurs identifiés).
- Comparaisons avec des matchs précédents (pas de base de données pour l'instant).
- Toute métrique "inventée" pour remplir l'espace.
