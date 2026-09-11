# Requirements: DEval-Webchat – allererster MVP

**Status:** bewusst schlanker vertikaler Arbeitsentwurf
**Ziel:** schnell eine prüfbare Oberfläche mit echtem Upload-, GraphRAG- und Chat-Flow sehen
**Notation:** Mit `A-xx` markierte Punkte sind bewusst gewählte, später änderbare Defaults/Annahmen.

## 1. Ziel und Leitentscheidung

Der allererste MVP ist kein vollständiges Produkt. Er soll schnell zeigen, ob der grundlegende DEval-Webchat fachlich und visuell passt. Deshalb wird zunächst ausschließlich der Happy Path umgesetzt. Grenzfälle, Komfortfunktionen und vollständige Verwaltungsprozesse werden bewusst verschoben.

Der Kernfluss lautet:

1. Sammlung anlegen oder auswählen
2. mehrere PDF-Dateien hochladen
3. aus den PDFs genau einen GraphRAG für die Sammlung aufbauen
4. den Zustand „GraphRAG bereit“ sehen
5. einen Chat mit dieser Sammlung und einem vorkonfigurierten Modell starten
6. eine Frage oder einen Zusammenfassungsauftrag stellen
7. eine Chatantwort mit sichtbaren Zitationen sehen

Eine Sammlung ist eine Gruppe von Dokumenten mit einem gemeinsamen inhaltlichen DEval-Kontext. Sie besitzt genau einen GraphRAG. Der Chat fragt diesen GraphRAG ab; das gewählte LLM formuliert aus den gelieferten Ergebnissen die Chatantwort.

## 2. Im allerersten MVP enthalten

### Sammlung und Upload

- Ein geschützter Pilotzugang öffnet die Weboberfläche.
- Nutzer:innen können eine benannte Sammlung anlegen und eine Sammlung auswählen.
- Mehrere PDFs können in eine Sammlung hochgeladen werden.
- Der Upload- und Aufbauvorgang zeigt mindestens „in Verarbeitung“ und „GraphRAG bereit“.
- Nach dem Upload wird für die Sammlung genau ein GraphRAG aufgebaut.

### Chat

- Ein neuer Chat verwendet im ersten MVP genau eine ausgewählte Sammlung.
- Vor dem ersten Beitrag wird ein Modell aus mehreren zentral vorkonfigurierten Modell-/Provider-Einträgen gewählt.
- Der Katalog enthält neben Ollama mindestens die in Pi.dev unter `openai-codex` verfügbaren GPT-5.6-Modelle `gpt-5.6-luna`, `gpt-5.6-sol` und `gpt-5.6-terra`.
- Nutzer:innen stellen eine Faktenfrage oder bitten um eine Zusammenfassung.
- Das ausgewählte Chat-LLM erhält die Ergebnisse des Sammlung-GraphRAGs und formuliert daraus die Antwort.
- Die Antwort wird als KI-generiert gekennzeichnet.

### Zitationen

- Chatantworten zeigen sichtbare Zitationen mit mindestens Dokumentname, Seite und Textausschnitt.
- Die Zitationen stammen aus den gelieferten GraphRAG-Ergebnissen und werden nicht vom LLM erfunden.
- Ein echter Dokumentbrowser, Seitensprung und Highlighting sind für diesen ersten MVP nicht erforderlich.
- Eine visuelle Quellenansicht darf als klar beschrifteter **Prototyp-Mock** gezeigt werden; sie darf nicht als bereits funktionierende Quellenöffnung ausgegeben werden.

## 3. Bewusst nicht enthalten

- mehrere Sammlungen in einem Chat
- `@`-Dokument-Mentions und Autocomplete
- separate Quellensuche neben dem Chat
- gespeicherte Chatverläufe oder Synchronisation
- Modellwechsel innerhalb eines laufenden Chats
- echter PDF-Dokumentbrowser, Seitennavigation und Text-Highlighting
- OCR, reine Scan-PDFs und vollständige Behandlung fehlerhafter Dateien
- Teilbestand, Wiederaufnahme, Fallbacks und komplexe Statusmodelle
- Umbenennen, Löschen, Teilen oder Rollenverwaltung für Sammlungen
- Graphvisualisierung und sichtbare Wissensgraph-Ansichten
- Websuche, allgemeines Wissen und freie Assistentenfunktionen
- formaler Freigabeworkflow für KI-Ausgaben

## 4. Akzeptanzkriterien für den vertikalen Slice

### AK-01: Sammlung und Upload

- Die Weboberfläche ermöglicht das Anlegen einer Sammlung.
- Mehrere gültige PDF-Dateien können dieser Sammlung zugeordnet werden.
- Nach dem Upload ist sichtbar, dass die Verarbeitung läuft.

### AK-02: GraphRAG-Bereitschaft

- Aus den hochgeladenen PDFs wird genau ein GraphRAG für die Sammlung aufgebaut.
- Nach erfolgreichem Aufbau zeigt die Oberfläche den Zustand „GraphRAG bereit“.
- Erst danach kann die Sammlung für den Chat ausgewählt werden.

### AK-03: Chatstart

- Nutzer:innen wählen genau eine bereite Sammlung.
- Nutzer:innen wählen ein Modell aus der zentralen Modell-/Provider-Konfiguration; neben Ollama stehen mindestens die GPT-5.6-Codex-IDs aus Pi.dev zur Auswahl.
- Danach kann eine erste Frage gesendet werden.

### AK-04: Chatantwort mit Zitationen

- Die Frage wird gegen den GraphRAG der gewählten Sammlung verarbeitet.
- Das gewählte LLM erzeugt daraus eine Chatantwort.
- Die Antwort enthält sichtbare Zitationen mit Dokumentname, Seite und Textausschnitt.
- Die Zitationen sind zunächst reine Anzeigeelemente; eine funktionierende Navigation ist kein Abnahmekriterium.

### AK-05: UI-Prüfung

- Der vollständige Happy Path ist in einer zusammenhängenden Oberfläche sichtbar: Sammlung, Upload-/Bereitschaftszustand, Chat, Modellwahl und Antwort.
- Ein optionaler Quellenfenster-Mock ist eindeutig als „Prototyp“ oder „nicht implementiert“ gekennzeichnet.

## 5. Änderbare Annahmen und Defaults

Diese Punkte sind bewusst gewählte Vereinfachungen für den allerersten MVP. Sie sind keine Festlegung für die spätere Produktversion:

- **A-01 Chat-Kontext:** Ein Chat verwendet zunächst genau eine Sammlung. Mehrere Sammlungen werden erst nach der ersten UI- und Flow-Prüfung ergänzt.
- **A-02 Modellbindung:** Das Modell wird beim Chatstart gewählt und bleibt für diesen Chat fest. Modellwechsel innerhalb des Chats kommt später.
- **A-03 Quellenansicht:** Zitationen werden zunächst nur angezeigt. Ein echter rechter PDF-Viewer wird später ergänzt; ein Mock ist nur für die visuelle Prüfung erlaubt und wird klar gekennzeichnet.
- **A-04 Suche:** Eine separate Suchansicht wird zunächst nicht gebaut. Fragen und Zusammenfassungsaufträge laufen direkt über den Chat.
- **A-05 Chat-Historie:** Chats werden im ersten UI-Slice nicht dauerhaft gespeichert. Lokale Sitzungen werden erst ergänzt, wenn der Kernflow passt.
- **A-06 Happy Path:** Für die erste Vorführung werden gültige, textbasierte PDFs vorausgesetzt. OCR, Scan-PDFs, Teilfehler und detaillierte Fehlerbehandlung sind bewusst nicht Teil der Abnahme.
- **A-07 Sammlungsverwaltung:** Zunächst sind nur Anlegen und Auswählen erforderlich. Umbenennen, Löschen, Teilen und Rechteverwaltung folgen später.
- **A-08 Backend-Echtheit:** Upload, GraphRAG-Aufbau, Retrieval und Chatantwort sollen echt funktionieren. Nur die optionale Quellenansicht darf visuell gemockt werden.
- **A-09 Modellkonfiguration:** Modelle und Provider werden zentral über eine Konfigurationsdatei bereitgestellt; neben Ollama werden mindestens die drei bestätigten Pi.dev-Codex-IDs angeboten. Nutzer:innen können keine eigenen Modelle oder Zugangsdaten eintragen. Die tatsächliche Nutzung setzt den passenden Provider in RAGFlow voraus.
- **A-10 Begriffsverständnis:** „Graph-Write“ wird in diesem Dokument als Aufbau des einen GraphRAG einer Sammlung verstanden. Falls damit ein separater Dienst gemeint ist, wird nur diese Bezeichnung angepasst.

## 6. Leitprinzipien

- **Schnell sichtbar werden:** Erst eine benutzbare Oberfläche, dann Vollständigkeit.
- **Echter Kern, bewusst falscher Rand:** Der Upload-/GraphRAG-/Chat-Kern wird nicht simuliert; ein UI-Mock wird nur als solcher bezeichnet.
- **Keine unnötige Vorarbeit:** Grenzfälle werden erst implementiert, wenn der Happy Path fachlich und visuell akzeptiert ist.
- **Nachvollziehbarkeit:** Auch im kleinen MVP bleiben Zitationen Bestandteil jeder Chatantwort.
