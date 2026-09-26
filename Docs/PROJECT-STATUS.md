# Status projet — llama-tui

**Date** : 2026-09-26 · **Repo** : `~/Llamacpp-server` · **Branche** : `main`
**Head** : `7ce6e4b` · **Working tree** : propre (untracked : `.DS_Store` uniquement)

## Qu'est-ce que ce projet

Malgré son nom, ce repo n'est **pas un fork de llama.cpp**. C'est une stack d'inférence
LLM locale avec TUI (Python ~1 770 lignes + TypeScript 218 lignes) qui orchestre :

1. **llama-server** (port 8082) — serveur OpenAI-compatible de llama.cpp, modèles GGUF via Metal
2. **proxy bun** (port 8001) — `src/fast_proxy.ts` : pont Anthropic API → OpenAI, pour piloter
   Claude Code (ou Cline) avec les modèles locaux
3. **TUI** (`llama-tui`) — gestion des deux processus : start/stop/restart, changement de
   modèle à chaud, stats temps réel (t/s, cache, ctx), profils de tâche, téléchargement HF

## Environnement (réparé le 2026-09-26, migration machine a2com → andyaugustine)

| Composant | Version | Note |
|---|---|---|
| llama.cpp (brew) | **0.5.0 (build 11146)** via Homebrew | installé ; **bottle SANS WebUI** (voir ci-dessous) |
| llama.cpp (custom) | **0.5.0 (b11146, WebUI embarquée)** | `~/builds/llama.cpp/build/bin/llama-server` — binaire actif du TUI (via `server_bin`) |
| Python | 3.11.16 (Homebrew) | `.venv` recréée (l'ancienne pointait vers un Python supprimé) |
| bun | 1.4.2 | backend proxy par défaut |
| cmake / node | 4.4.3 / 26.10 | requis pour le rebuild custom (build npm UI natif, plus de shim node→bun) |
| litellm | non installé | décision : backend bun uniquement |
| alias | `llama-tui` dans `~/.zshrc` (créé) | lance la venv directement (contourne le python3 système 3.9) |
| `config/server.json` → `server_bin` | `~/builds/llama.cpp/build/bin/llama-server` | persisté dans server.json (gitignore, machine-spécifique) |

### Binaire llama-server custom (WebUI)

Le bottle Homebrew est compilé sans `LLAMA_BUILD_UI` → `GET /` = 404 (diagnostic 2026-09-25).
Solution en place : build source v0.5.0 avec WebUI embarquée (test garde-fou
`tests/test_webui_assets.py`) :

```bash
# Rebuild si besoin (l'UI est construite via npm automatiquement par scripts/ui-assets.cmake)
cd ~/builds/llama.cpp
git pull && cmake -S . -B build -DLLAMA_BUILD_UI=ON -DLLAMA_USE_PREBUILT_UI=ON \
      -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF
cmake --build build --target llama-server -j8
# Prérequis : brew install cmake node

# ⚠ MAINTENANCE : ne jamais remplacer server_bin par le binaire brew
# (bottle sans WebUI). `brew upgrade llama.cpp` est sans effet sur le binaire du TUI.
```

Note : le bucket HF `ggml-org/llama-ui` (prébuilt) est privé (401) — l'UI se construit
localement via npm (`scripts/ui-assets.cmake` la gère nativement : build + gzip + embed,
70 assets). `GET /` requiert `--compressed` côté curl (gzip) ; les navigateurs gèrent
nativement. Version affichée : `0.5.0-dev (build 1)` (shallow clone sans build-info, même
commit `7fe450e19` que le bottle — cosmétique).

## Réparation 2026-09-26 (migration machine)

Session de remise en service après migration (`/Users/a2com/...` → `/Users/andyaugustine/...`) :
venv recréée, `brew install python@3.11 llama.cpp cmake node`, alias `~/.zshrc` créé,
`server_bin` corrigé, build custom WebUI refait (`GET /` → 200). **206 pytest + 15 bun
tests, 100 % verts** (garde-fou WebUI inclus). Chaîne complète validée : llama-server
(health 12-16 s) + proxy bun (`/v1/messages` Anthropic → « OK ») + widgets TUI.

## Ce qui a été fait (chronologie)

### 1. Audit initial
Analyse complète : 12 modules Python + 1 TS, graphe de dépendances acyclique, tests existants
par module (~196), docs à jour. Problèmes identifiés : god-file `tui.py` (855 l.), bug thinking,
pas de rollback sidecar, dead code, logs non purgés (840 Mo), sécurité proxy (hors périmètre).

### 2. PLAN-FONCTIONNEL.md — rendre le TUI opérationnel ✅
- **Env** : `brew install python@3.11 llama.cpp bun` ; maj vérifiée (`brew outdated` vide) ;
  `.venv` recréée ; logs tronqués ; 196 pytest + 15 bun verts
- **Bug 1 (TDD)** : flag `thinking` perdu au switch backend (`action_switch_proxy_backend`
  reconstruisait ProxyManager sans `thinking`) — 🔴 test rouge → 🟢 fix
- **Bug 2 (TDD)** : sidecar JSON corrompu au chargement de modèle = état moitié-muté — fix double :
  `model_manager.load()` → `ConfigError` + clés protégées (`model/host/port/...`) non écrasables
  par un sidecar ; `_load_and_restart` → try/except, aucun état muté, serveur non relancé
- **Validation réelle** : modèle MTP 33 Go chargé (~62 s), inférence 23.7 t/s, proxy bun OK
  (bridge Anthropic streaming), reload à chaud Qwen3.8-27B (11.9 t/s), `/slots` OK

### 3. PLAN-UI-REDESIGN.md — 3 parties en TDD ✅

**Partie 1 — Suppression llama-monitor** (commit `b14a512`)
- 9 zones de `tui.py` (constantes, binding `m`, bouton, label sidebar, refresh, actions)
- 4 tests monitor supprimés + test d'absence global (`test_tui_monitor_fully_removed`)
- `config/llama-monitor-presets.json` supprimé, README/CLAUDE.md purgés
- Critère atteint : `grep -ri monitor` → 0 hit (hors test d'absence)

**Partie 2 — Refonte dashboard** (commit `d3caed1`)
- Palette **Catppuccin Mocha** (fond `#1e1e2e`, surface `#313244`, vert `#a6e3a1`,
  rouge `#f38ba8`, orange `#fab387`, bleu `#89b4fa`, texte `#cdd6f4`) — Monokai éradiqué
- 2 barres de boutons fusionnées en une `#action-bar` unique
- Bordures `round`, sections bleues, badges `●/○` conservés
- Test palette (`test_tui_palette_is_catppuccin`) + screenshot SVG validé

**Partie 3 — Graphiques** (commit `68c90c6`)
- `stats_collector.poll()` : `cache_history` n'est plus polluée de 0 % en idle
  (archivage uniquement pendant la génération + 1 point final à la transition)
- 2 sparklines labellisées (`⚡ tokens/s`, `Cache %`), hauteurs 4/3
- Widget **Digits** géant pour le t/s instantané (remplace le Label texte)
- **ProgressBar** pour l'occupation du contexte (`ctx_used_pct` enfin visualisé)

### Historique des commits de la session

| Hash | Message |
|---|---|
| `a87c7c6` | fix(tui): thinking au switch backend + rollback sidecar corrompu (TDD) |
| `2d63d25` | docs: plan UI redesign |
| `b14a512` | refactor(tui): remove llama-monitor integration |
| `d3caed1` | feat(tui): dashboard redesign (catppuccin-mocha, single action bar) |
| `68c90c6` | feat(tui): stats visuals (digits, labeled sparklines, ctx bar) |
| `1ae536b` | docs: plan UI redesign exécuté (résultats + hashes) |
| `e479959` | docs: status du projet |
| `7ce6e4b` | fix(config): server_bin persiste dans server.json (indépendant de l'env TUI) |
| (2026-09-26) | docs: migration machine + rebuild WebUI (build npm natif, plus de shim bun) |

## État des tests

**206 pytest + 15 bun tests, 100 % verts**. Garde-fou `test_webui_assets.py` : échoue
si le binaire llama-server actif ne sert pas la WebUI (404).
Nouveaux tests TDD créés cette session : test d'absence monitor, action-bar unique,
palette catppuccin, cache_history idle, cache fin de génération, structure graphiques
(labels/Digits/ProgressBar/style), thinking au switch, rollback sidecar.

## Backups

- `~/backups/llama-tui-20260925-1158.tar.gz` (source, hors models/.venv/.git)
- `~/backups/llama-tui-history.bundle` (historique git complet, restaurable via `git clone`)

## Documentation livrée

| Fichier | Contenu |
|---|---|
| `Docs/PLAN-FONCTIONNEL.md` | Plan de remise en service + routine de maintenance llama.cpp (`brew outdated` → `upgrade` → retest pytest) |
| `Docs/PLAN-UI-REDESIGN.md` | Plan UI (3 parties) + résultats d'exécution avec hashes |
| `Docs/tui-screenshot.svg` | Capture du rendu dashboard (générée via `run_test()`) |
| `Docs/PROJECT-STATUS.md` | Ce document |

## Connu / hors périmètre (reporté)

Identifié à l'audit, volontairement non traité :
- **Sécurité proxy** : fast_proxy.ts sans auth sur toutes les interfaces ; mapping `tools`/
  `tool_choice` Anthropic absent (drop silencieux) ; `usage` SSE approximatif
- **Path traversal** dans `ModelManager.download()` (`local_filename` non normalisé)
- **tui.py reste un monolithe** (~750 lignes post-nettoyage) : découpage screens/services à prévoir
- Duplication `_resolve()` (3 modules), racine projet incohérente dans `server_manager.py`
- Sleeps bloquants UI (`time.sleep` dans actions restart/monitor→supprimé/download)
- `bench.py` non câblé à l'UI ; `check_version()` dead code
- `/` (filtre modèles) documenté dans CLAUDE.md mais inexistant dans les BINDINGS

## Comment lancer

```bash
llama-tui          # alias configuré dans ~/.zshrc
# touches : a = tout démarrer · l = charger modèle · t = profil · b = backend · ? = aide
```

Maintenance llama.cpp :
```bash
brew update && brew outdated llama.cpp   # si non vide :
brew upgrade llama.cpp && pytest         # puis relancer le TUI
```