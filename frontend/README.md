# DEval Webchat UI

Erster React/Vite-Oberflächen-Slice auf Basis des Figma-Exports in `Downloads/Build it/src`.

```bash
cd frontend
pnpm install
DEVAL_API_URL=http://127.0.0.1:8790 \
VITE_RAGFLOW_WEB_URL=http://127.0.0.1 \
pnpm dev --host 127.0.0.1
```

Die Oberfläche spricht im Dev-Modus über Vite-Proxy mit der lokalen Web-API. Starte dafür in zwei Terminals:

```bash
# Terminal 1, aus dem Repository-Root
.venv/bin/deval-webchat

# Terminal 2
cd frontend
pnpm dev
```

Die UI ist unter `http://127.0.0.1:5173` erreichbar. Über den Umschalter **RAGFlow-Backend** lässt sich die RAGFlow-Weboberfläche direkt im Frontend einblenden; `VITE_RAGFLOW_WEB_URL` kann dafür auf eine andere RAGFlow-UI-URL zeigen. Der Happy Path ist verbunden: Sammlung anlegen, mehrere PDFs hochladen, Verarbeitung/GraphRAG-Status pollen, ein extern in RAGFlow konfiguriertes Modell wählen, chatten und deterministische Zitate anzeigen. Über `+` im Chatkopf lassen sich beliebig viele Chats je Sammlung anlegen und wechseln; Verlauf und Modellwahl werden je Chat ausschließlich im Browser gespeichert. Standardmäßig nutzt das Backend RAGFlows stateless OpenAI-kompatiblen Endpoint ohne RAGFlow-Session. Das Quelldokument lässt sich über den Dateinamen im Quellenbereich als PDF öffnen; die Seitenvorschau bleibt lokal.
