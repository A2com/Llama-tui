# Plan — Rendre llama-tui pleinement fonctionnel

**Date** : 2026-09-25 · **Repo** : `/Volumes/Twoa Files/Git/llama.cpp` · **Branche** : `main`
**Statut** : ✅ exécuté (voir « Résultat d'exécution » en fin de document)

## Contexte et diagnostic

Stack d'inférence LLM locale : TUI Textual qui orchestre `llama-server` (port 8082) et un proxy
Anthropic→OpenAI (port 8001) pour piloter Claude Code avec des modèles GGUF locaux (Metal).

**Bloquants détectés à l'audit :**

- `.venv` cassée : créée depuis l'ancien chemin `~/llama.ccp` avec un Python 3.11 (Python.org) supprimé depuis
- `llama-server`, `bun`, `litellm` absents du PATH
- `/usr/bin/python3` = 3.9.6 (trop vieux pour textual 8) ; Homebrew 7.0.6 présent
- Logs non purgés : `config/logs/server.log` ≈ 790 Mo, `logs/proxy.log` ≈ 50 Mo
- **Bug code 1** : `action_switch_proxy_backend` (src/tui.py:759-761) reconstruit `ProxyManager`
  sans le flag `thinking` → réinitialisation silencieuse
- **Bug code 2** : `_load_and_restart` (src/tui.py:634-668) sans try/except autour de `model_mgr.load()`
  → sidecar JSON corrompu = état moitié-muté, pas de rollback
- Modèles OK : 3 GGUF (Qwen3.6-35B MTP 33 Go, Uncensored 21 Go, Qwen3.8-27B 16 Go) + 2 mmproj + sidecars

**Décisions validées** : binaires via Homebrew ; backend proxy = bun uniquement (pas litellm) ;
correctifs = bugs bloquants uniquement ; développement des fixes en TDD puis alias simple.

## Phase 0 — Réparer l'environnement

| #   | Action                              | Commande / critère                                                                       |
|-----|-------------------------------------|------------------------------------------------------------------------------------------|
| 0.1 | Installer python 3.11, llama.cpp, bun | `brew install python@3.11 llama.cpp bun` (llama.cpp 0.5.0, bun 1.4.2, python 3.11.16) |
| 0.2 | **Vérifier maj llama.cpp**          | `brew update` → `brew outdated llama.cpp` → si non vide : `brew upgrade llama.cpp`      |
| 0.2b| Sanity-check post-install           | `llama-server --version` ≥ b10310, flag MTP `--spec-type draft-mtp` accepté             |
| 0.3 | Recréer la venv                     | `rm -rf .venv && /opt/homebrew/bin/python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt` |
| 0.4 | Purger les logs                     | tronquer `config/logs/server.log` et `logs/proxy.log` (~840 Mo libérés)                 |
| 0.5 | Suite existante au vert             | `pytest` + `bun test tests/fast_proxy.test.ts` (~200 tests)                              |

### Routine de maintenance llama.cpp

À chaque reprise du projet :

```bash
brew update && brew outdated llama.cpp   # si sortie non vide :
brew upgrade llama.cpp                   # puis re-test pytest avant de relancer le TUI
```

## Phase 1 — Correctifs en TDD (rouge → vert)

**Bug 1 — flag `thinking` perdu au switch de backend**

- 🔴 `tests/test_tui.py` : test `action_switch_proxy_backend` avec `thinking=True` → le nouveau
  ProxyManager doit recevoir `thinking=True` (échoue actuellement)
- 🟢 Fix src/tui.py:759-761 : passer `thinking=self._thinking`

**Bug 2 — rollback si sidecar corrompu**

- 🔴 Test : sidecar JSON invalide → `_load_and_restart` logue l'erreur **sans** muter
  `_manager`/`_proxy`/`_stats` ni démarrer
- 🟢 Fix : try/except autour de `model_mgr.load(path)` dans `_load_and_restart`, restauration de
  l'ancien proxy ; blinder `json.loads(sidecar)` dans `model_manager.load()` avec `ConfigError`

Contrainte TDD : test écrit d'abord (rouge constaté), fix minimal, suite complète au vert avant de
passer au bug suivant. Aucun refactoring annexe.

## Phase 2 — Alias `llama-tui`

Dans `~/.zshrc` (chemins avec espaces → quoting obligatoire) :

```zsh
alias llama-tui='"/Volumes/Twoa Files/Git/llama.cpp/.venv/bin/python" "/Volumes/Twoa Files/Git/llama.cpp/llama-tui"'
```

Justification : contourne le shebang `#!/usr/bin/env python3` qui attraperait le Python système 3.9
incompatible. Vérif : `type llama-tui`.

## Phase 3 — Validation fonctionnelle

1. `llama-tui` → `a` : llama-server charge le modèle par défaut `Qwen3.6-35B-A3B-MTP`
   (config/server.json) ; proxy bun sur 8001
2. Stats t/s via `/slots`, `/health` OK
3. `l` sur chaque modèle : Qwen3.8-27B (*qualite*), Uncensored (*uncensored*) — chargement à chaud
4. Bug 1 en réel : switch backend `b` avec thinking actif → flag préservé
5. `t` : les 3 profils de tâche switchables

## Hors périmètre (identifié à l'audit, reporté)

Path traversal dans `download()`, sleeps bloquants UI, découpage de `tui.py`, factorisation
`_resolve()`/managers, mapping `tools` de fast_proxy.ts, auth proxy, harmonisation racine projet.

## Résultat d'exécution (2026-09-25)

| Phase | Résultat |
|---|---|
| 0.1–0.2 | `brew install python@3.11 llama.cpp bun` → llama.cpp **0.5.0 (build 11146)** déjà à jour (`brew outdated` vide), flag `draft-mtp` supporté ✓ |
| 0.3 | `.venv` recréée (Python 3.11.16 Homebrew), deps OK |
| 0.4 | Logs tronqués (~840 Mo libérés) |
| 0.5 | **196 pytest + 15 bun tests au vert** |
| 1 — Bug 1 | 🔴 test rouge (`test_switch_proxy_backend_preserves_thinking`) → 🟢 fix tui.py : `thinking=self._thinking` au switch backend |
| 1 — Bug 2 | Fix double : `model_manager.load()` → sidecar corrompu = `ConfigError`, clés protégées (`model/host/port/...`) non écrasables ; `tui._load_and_restart` → try/except, log d'erreur, aucune mutation, serveur non relancé |
| 2 | Alias `llama-tui` créé dans `~/.zshrc` (vérifié `type llama-tui`) |
| 3 | Validation réelle : modèle MTP 33 Go chargé en ~62 s, `/health` ok, inférence OpenAI OK (23.7 t/s), proxy bun 8001 OK (bridge Anthropic `/v1/messages` streaming OK), reload à chaud vers Qwen3.8-27B OK (11.9 t/s), `/slots` pollé par StatsCollector OK |

**Total final** : 198 pytest + 15 bun tests au vert. Config `server.json` restaurée sur le modèle MTP par défaut.