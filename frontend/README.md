# DEval Webchat UI

Erster React/Vite-Oberflächen-Slice auf Basis des Figma-Exports in `Downloads/Build it/src`.

```bash
cd frontend
pnpm install
pnpm dev
```

Die Oberfläche spricht im Dev-Modus über Vite-Proxy mit der lokalen Web-API. Starte dafür in zwei Terminals:

```bash
# Terminal 1, aus dem Repository-Root
.venv/bin/deval-webchat

# Terminal 2
cd frontend
pnpm dev
```

Der Happy Path ist verbunden: Sammlung anlegen, mehrere PDFs hochladen, Verarbeitung/GraphRAG-Status pollen, Modell wählen, chatten und deterministische Zitate anzeigen. Der Quellenbereich bleibt als Prototyp-Mock gekennzeichnet; PDF-Navigation und Hervorhebung folgen später.
