# Reproduktion (lokal)

Die folgenden Schritte starten den offiziellen RAGFlow-Stand `v0.27.1`, ingestieren beide Deval-PDFs und prüfen Retrieval, Antworten und Provenienz.

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
RAGFLOW_LLM_MODEL=qwen2.5:0.5b@local@Ollama
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

## 3. Lokales Chat-Modell konfigurieren

```bash
docker run -d --name deval-ollama \
  --restart unless-stopped \
  -p 127.0.0.1:11434:11434 \
  -v deval_ollama_data:/root/.ollama \
  ollama/ollama@sha256:684d8674b4315fa18f4f0e973a118ec2652ed96f67563277839985175858e0ba
docker exec deval-ollama ollama pull qwen2.5:0.5b
```

In RAGFlow unter **Settings → Model providers → Ollama** die Instanz `local` anlegen:

- Base URL: `http://host.docker.internal:11434`
- Modell: `qwen2.5:0.5b`
- Typ: Chat

Auf Linux kann statt `host.docker.internal` die für Docker erreichbare Host-Adresse nötig sein.

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

## 7. Optional: GraphRAG

Nach erfolgreichem Parsing kann der offizielle GraphRAG-Ablauf getestet werden:

```bash
python -m deval_ragflow graph --timeout 3600
```

Der Befehl startet RAGFlows GraphRAG, wartet auf den dokumentierten Abschlusszustand und gibt den Graphen aus. Eine FalkorDB- oder separate Visualisierungsintegration ist in diesem PoC nicht enthalten; ein leerer Graph, fehlender Provider oder Timeout bleibt ein ehrliches Ergebnis.

## 8. Aufräumen

Die Compose-Hülle entfernt keine persistenten Volumes. Einen ausschließlich lokal erzeugten Dataset-Eintrag kann man nur mit seiner bekannten ID und der exakten Bestätigung löschen:

```bash
python -m deval_ragflow reset-test-data \
  --dataset-id REMOTE_DATASET_ID \
  --confirm 'DELETE DEVAL TEST DATA'
```
