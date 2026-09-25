# Plan UI — Suppression monitor + refonte dashboard + graphiques

**Date** : 2026-09-25 · **Repo** : `/Volumes/Twoa Files/Git/llama.cpp` · **Branche** : `main`
**Méthode** : TDD strict (rouge → vert) · **Palette** : Catppuccin Mocha
**Statut** : ✅ exécuté (voir « Résultat » en fin)

## Contexte

Suite de `PLAN-FONCTIONNEL.md` (environnement réparé, 2 bugs TDD corrigés, commit a87c7c6).
Trois parties exécutées en commits séparés (bisect facile) :

| Commit | Contenu | Message |
|---|---|---|
| 1 | fixes bugs TDD + docs plan (existant validé) | `fix(tui): thinking au switch backend + rollback sidecar corrompu (TDD)` |
| 2 | suppression llama-monitor | `refactor(tui): remove llama-monitor integration` |
| 3 | refonte UI dashboard | `feat(tui): dashboard redesign (catppuccin-mocha, single action bar)` |
| 4 | graphiques stats | `feat(tui): stats visuals (digits, labeled sparklines, ctx bar)` |

## Partie 1 — Suppression llama-monitor (commit 2)

**Code retiré de src/tui.py (9 zones)** :
- Constantes `LLAMA_MONITOR_BIN` / `_PORT` / `_PRESETS` (l.30-32)
- Aide `("m", "Démarre llama-monitor")` (l.112) · `Binding("m", ...)` (l.329)
- Sidebar `▪ llama-monitor` + `lbl-monitor-status` (l.405-407)
- `Button btn-monitor` (l.434) · appel `_refresh_monitor()` (l.501) · méthode (l.574-579)
- `_build_llama_monitor_cmd()` + `action_start_llama_monitor()` (l.790-820)
- Entrée `"btn-monitor"` dans `on_button_pressed` (l.855)

**Tests** : suppression des 4 tests monitor (`test_tui_has_llama_monitor_action`,
`test_tui_has_llama_monitor_binding`, `test_llama_monitor_cmd_includes_models_dir`,
`test_tui_has_monitor_status_label`) + ajout d'un test d'absence global
(`test_tui_monitor_fully_removed`) : aucun hit `monitor` dans tui.py, touche `m` libre.

**Périphérie** : `config/llama-monitor-presets.json` supprimé, README.md (3 lignes env) et
CLAUDE.md (3 lignes env + binding m) purgés. Critère : `grep -ri monitor` → 0 hit.

**Décisions validées** : preset supprimé, touche `m` laissée libre, 2 commits séparés.

## Partie 2 — Refonte UI dashboard (commit 3)

Palette **Catppuccin Mocha** : fond `#1e1e2e`, surface `#313244`, vert `#a6e3a1`,
rouge `#f38ba8`, orange `#fab387`, bleu `#89b4fa`, texte `#cdd6f4`, muted `#9399b2`.

**Structure** : header 2 blocs (serveur + proxy), fusion des 2 barres de boutons en une
action-bar unique (LLM à gauche, proxy/logs à droite), sidebar en cartes visuelles
(fond surface, bordures subtiles, titres de section colorés), badges `●/○` unifiés.

**Cycles TDD** :
- 🔴 Cycle 4 structure : présence `#action-bar`, absence des anciens ids `#controls`/
  `#proxy-controls`, ids existants (`lbl-tps`, `sparkline-*`, `tab-slots`) conservés
- 🟢 implémentation layout + CSS
- 🔴 Cycle 5 palette : `test_tui_palette_is_catppuccin` (aucun `#272822` Monokai)
- 🟢 application palette → smoke `run_test()` Textual

**Garde-fou** : aucune logique métier — uniquement `compose()`, CSS, bindings, boutons.

## Partie 3 — Graphiques stats (commit 4)

| Problème | Fichier | Fix |
|---|---|---|
| cache_history polluée en idle | stats_collector.py:63 | append uniquement si processing ou transition génér→idle |
| 2 sparklines sans titre | tui.py:414-415 | labels `⚡ t/s` et `Cache %` au-dessus |
| Sparklines height:2 | tui.py:308-313 | hauteurs 4/3 |
| t/s en Label texte | tui.py:413 | widget `Digits` géant (disponible textual 8.2.8) |
| ctx_used_pct invisible | — | `ProgressBar` bindée sur ctx |

**Cycles TDD** :
- 🔴 Cycle A : `test_cache_history_not_polluted_when_idle` → 🟢 fix `poll()`
- 🔴 Cycle B : tests structure (`lbl-spark-tps`, `lbl-spark-cache`, `digits-tps`,
  `progress-ctx`) → 🟢 implémentation compose + refresh
- 🔴 Cycle C : test CSS (hauteurs 4/3, couleurs catppuccin, plus de Monokai) → 🟢
- 🟢 Cycle D : suite complète + smoke génération live (Digits/sparklines alimentées)

**Garde-fou** : `TokenStats` inchangé ; tests tps/cache existants restent verts sans modif.

## Livrable final

4 commits, ~200 tests verts (pytest + bun), TUI sans monitor, UI catppuccin dashboard,
graphiques labellisés + Digits + ProgressBar ctx, backup tar + bundle dans `~/backups/`.

## Résultat (2026-09-25)

| Commit | Hash | Contenu |
|---|---|---|
| 1 | `a87c7c6` | fixes bugs TDD (thinking + rollback sidecar) + PLAN-FONCTIONNEL.md |
| 2 | `2d63d25` | docs plan UI redesign |
| 3 | `b14a512` | suppression llama-monitor (9 zones tui.py, 4 tests, preset json, README/CLAUDE.md) |
| 4 | `d3caed1` | dashboard catppuccin-mocha, action-bar unique (2 barres fusionnées) |
| 5 | `68c90c6` | stats visuals : Digits tps géant, sparklines labellisées h4/h3, ProgressBar ctx |

**Tests finaux** : 203 pytest + 15 bun au vert. Nouveaux tests TDD : `test_tui_monitor_fully_removed`,
`test_tui_single_action_bar`, `test_tui_palette_is_catppuccin`,
`test_cache_history_not_polluted_when_idle`, `test_cache_history_records_final_value_on_generation_end`,
tests structure (labels sparklines, Digits, ProgressBar, style).

**Smoke réel** : `run_test()` 120×40 → structure OK, Digits « 23.7 », ProgressBar 0.4 %, sparklines
alimentées [23.7, 24.1] / [87.0], screenshot SVG validé (palette catppuccin présente) → `Docs/tui-screenshot.svg`.
Backups : `~/backups/llama-tui-20260925-1158.tar.gz` + `~/backups/llama-tui-history.bundle`.