# Reproduktion (lokal)

Die folgenden Schritte starten den offiziellen RAGFlow-Stand `v0.27.2`, ingestieren beide Deval-PDFs und prüfen Retrieval, Antworten und Provenienz.

## 1. Umgebung vorbereiten

Voraussetzungen: Docker Compose, Python 3.9–3.12, mindestens 16 GB RAM und auf macOS/ARM ein Docker-Setup mit x86-Emulation.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
cp .env.example .env
```

In `.env` eintragen:

```dotenv
RAGFLOW_BASE_URL=http://localhost:9380
RAGFLOW_API_KEY=<RAGFlow-API-Key>
RAGFLOW_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5@Builtin
RAGFLOW_LLM_MODEL=gpt-5.6-luna@codex@OpenAI
```

`.env` bleibt lokal und ist ignoriert. Die Basis-URL darf **nicht** `/api/v1` enthalten.

## 2. RAGFlow starten

```bash
./scripts/ragflow-compose.sh verify
./scripts/ragflow-compose.sh up
./scripts/ragflow-compose.sh wait
python -m deval_ragflow doctor
```

`doctor` muss `health.ok: true`, eine erfolgreiche authentifizierte Dataset-Abfrage und `sqlite_writable: true` melden.

## 3. Externes Chat-/GraphRAG-Modell konfigurieren

Ollama wird für dieses Setup nicht benötigt. In RAGFlow unter **Settings → Model providers → OpenAI** eine Instanz `codex` oder den Namen des eigenen Gateways anlegen:

- Base URL: `https://api.openai.com/v1` oder die URL des eigenen OpenAI-kompatiblen Gateways
- API-Key: Key des externen LLM-Providers
- Typ: Chat
- Modell: die tatsächlich verfügbare externe Modell-ID, z. B. `gpt-5.6-luna`

Für die drei im Webchat hinterlegten Modell-IDs `gpt-5.6-luna`, `gpt-5.6-sol` und `gpt-5.6-terra` muss RAGFlow jeweils ein Chatmodell mit exakt dieser ID kennen. Der passende zusammengesetzte Wert in `.env` ist z. B. `gpt-5.6-luna@codex@OpenAI`. Bei einem anderen Instanznamen oder einer anderen Factory den exakten Wert aus RAGFlows Chatmodell-Katalog verwenden.

Der externe Provider-Key wird ausschließlich in RAGFlow gespeichert. `RAGFLOW_API_KEY` ist dagegen der separate RAGFlow-Key, den DEval für die API-Aufrufe benötigt.

## 4. Beide PDFs ingestieren

Beide Dateien sind `mixed`; deshalb ist `--allow-mixed` erforderlich:

```bash
python -m deval_ragflow ingest \
  data/input/2024_DEval_Minderungsstudie_Web__Zusammenfassung.pdf \
  --allow-mixed --timeout 1800

python -m deval_ragflow ingest \
  data/input/2025_DEval_Dezentralisierung_Afrika_Zusammenfassung.pdf \
  --allow-mixed --timeout 1800
```

Ein erneuter Aufruf mit denselben Bytes ist idempotent und verwendet die lokale Version sowie das deterministische Remote-Mapping weiter.

## 5. Fakten abrufen und fragen

`query` liefert die RAGFlow-Treffer und die lokal aufgelösten Quellen:

```bash
python -m deval_ragflow query "Welche zentralen Ergebnisse nennt die Minderungsstudie?"
```

`ask` erzeugt zusätzlich über RAGFlows Chat-API eine kurze Antwort. Die angezeigten Zitate kommen aus einem separaten Retrieval-Aufruf und werden nie durch ein LLM gematcht:

```bash
python -m deval_ragflow ask \
  "Welche zentralen Ergebnisse nennt die Minderungsstudie?" \
  --timeout 360
```

Für maschinenlesbare Ausgabe:

```bash
python -m deval_ragflow ask \
  "Welche zentralen Ergebnisse nennt die Minderungsstudie?" \
  --timeout 360 --json
```

Das kleine CPU-Modell kann langsam sein. Ein expliziter Timeout- oder Provider-Fehler beendet den Befehl mit einem Fehler; er wird nicht als erfolgreiche Antwort ausgegeben.

## 6. Automatisierte Prüfung

```bash
python -m compileall -q src tests
python -m pytest -q
RUN_LIVE_RAGFLOW_TESTS=true python -m pytest -q -rs tests/test_live_ragflow.py
```

Ohne `RUN_LIVE_RAGFLOW_TESTS=true` wird der Live-Test absichtlich übersprungen. Mit gesetztem Flag sind Netzwerk-, Modell-, Parsing- und Citation-Fehler echte Testfehler.

## 7. Webchat-Vertikalslice

Nach dem Start von RAGFlow und dem Anlegen der Modell-Provider laufen API und UI getrennt:

```bash
# Terminal 1, Repository-Root
.venv/bin/deval-webchat --host 127.0.0.1 --port 8790

# Terminal 2
cd frontend
pnpm install
DEVAL_API_URL=http://127.0.0.1:8790 pnpm dev --host 127.0.0.1
```

Die UI ist anschließend unter `http://localhost:5173` erreichbar. Sammlung anlegen, mehrere PDFs auswählen und den Status bis **GraphRAG bereit** verfolgen. Das Modell wird aus `config/models.json` geladen; deaktivierte Einträge bleiben unsichtbar. Erst nach dem erfolgreichen GraphRAG-Aufbau ist der Chat freigeschaltet. Zitate werden nach der Chatantwort separat über `/retrieval` aufgelöst. Der rechte Quellenbereich ist weiterhin nur ein gekennzeichneter Prototyp-Mock.

Die für den Webchat verwendete Web-API läuft hier auf `http://127.0.0.1:8790` und stellt bereit:

- `GET /api/collections` und `POST /api/collections`
- `POST /api/collections/{id}/upload` (multipart, asynchron)
- `GET /api/jobs/{id}` und `GET /api/collections/{id}/graph`
- `GET /api/models`, `POST /api/chat/sessions`, `POST /api/query`, `POST /api/chat`

## 8. Optional: GraphRAG

Nach erfolgreichem Parsing kann der offizielle GraphRAG-Ablauf getestet werden:

```bash
python -m deval_ragflow graph --timeout 3600
```

Der Befehl startet RAGFlows GraphRAG, wartet auf den dokumentierten Abschlusszustand und gibt den Graphen aus. Eine FalkorDB- oder separate Visualisierungsintegration ist in diesem PoC nicht enthalten; ein leerer Graph, fehlender Provider oder Timeout bleibt ein ehrliches Ergebnis.

## 9. Aufräumen

Die Compose-Hülle entfernt keine persistenten Volumes. Einen ausschließlich lokal erzeugten Dataset-Eintrag kann man nur mit seiner bekannten ID und der exakten Bestätigung löschen:

```bash
python -m deval_ragflow reset-test-data \
  --dataset-id REMOTE_DATASET_ID \
  --confirm 'DELETE DEVAL TEST DATA'
```
