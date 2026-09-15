import { useEffect, useRef, useState } from "react"
import {
  type Citation,
  type Collection,
  type GraphRAGStatus,
  type Model,
  createCollection,
  deleteCollectionDocument,
  getCollection,
  listCollections,
  listModels,
  sendChat,
  uploadCollection,
} from "./api"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  citations?: Citation[]
  model?: string
}

interface CitationSelection {
  citation: Citation
  index: number
}

interface StoredChat {
  id: string
  messages: Message[]
  modelId: string | null
}

interface StoredChatState {
  activeChatId: string
  chats: StoredChat[]
}

const CHAT_STORAGE_PREFIX = "deval-webchat:v1:"

function chatStorageKey(collectionId: string) {
  return `${CHAT_STORAGE_PREFIX}${encodeURIComponent(collectionId)}`
}

function newStoredChat(): StoredChat {
  return { id: crypto.randomUUID(), messages: [], modelId: null }
}

function emptyChatState(): StoredChatState {
  const chat = newStoredChat()
  return { activeChatId: chat.id, chats: [chat] }
}

function isStoredCitation(value: unknown): value is Citation {
  if (!value || typeof value !== "object") return false
  const citation = value as Record<string, unknown>
  return (
    typeof citation.document === "string" &&
    (typeof citation.page === "number" || citation.page === null) &&
    typeof citation.excerpt === "string" &&
    (citation.resolved === undefined ||
      typeof citation.resolved === "boolean") &&
    (citation.document_url === undefined ||
      typeof citation.document_url === "string") &&
    (citation.image_id === undefined ||
      typeof citation.image_id === "string") &&
    (citation.image_url === undefined ||
      typeof citation.image_url === "string") &&
    (citation.source === undefined ||
      (typeof citation.source === "object" &&
        citation.source !== null &&
        !Array.isArray(citation.source)))
  )
}

function isStoredMessage(value: unknown): value is Message {
  if (!value || typeof value !== "object") return false
  const message = value as Record<string, unknown>
  return (
    typeof message.id === "string" &&
    (message.role === "user" || message.role === "assistant") &&
    typeof message.content === "string" &&
    (message.citations === undefined || Array.isArray(message.citations)) &&
    (message.model === undefined || typeof message.model === "string")
  )
}

function isStoredChat(value: unknown): value is StoredChat {
  if (!value || typeof value !== "object") return false
  const chat = value as Record<string, unknown>
  return (
    typeof chat.id === "string" &&
    chat.id.length > 0 &&
    Array.isArray(chat.messages) &&
    (chat.modelId === null || typeof chat.modelId === "string")
  )
}

function sanitizeMessages(value: unknown): Message[] {
  return Array.isArray(value)
    ? value.filter(isStoredMessage).map((message) => ({
        ...message,
        citations: Array.isArray(message.citations)
          ? message.citations.filter(isStoredCitation)
          : undefined,
      }))
    : []
}

function readStoredChats(collectionId: string): StoredChatState {
  const empty = emptyChatState()
  if (typeof window === "undefined") return empty
  try {
    const raw = window.localStorage.getItem(chatStorageKey(collectionId))
    if (!raw) return empty
    const value: unknown = JSON.parse(raw)
    if (!value || typeof value !== "object") return empty
    const stored = value as Record<string, unknown>
    if (Array.isArray(stored.chats)) {
      const chats = stored.chats
        .filter(isStoredChat)
        .map((chat) => ({ ...chat, messages: sanitizeMessages(chat.messages) }))
      if (!chats.length) return empty
      const activeChatId =
        typeof stored.activeChatId === "string" &&
        chats.some((chat) => chat.id === stored.activeChatId)
          ? stored.activeChatId
          : chats[0].id
      return { activeChatId, chats }
    }

    // Migrate the original one-chat-per-collection format in place.
    if (Array.isArray(stored.messages) || typeof stored.modelId === "string") {
      const chat = {
        ...newStoredChat(),
        messages: sanitizeMessages(stored.messages),
        modelId: typeof stored.modelId === "string" ? stored.modelId : null,
      }
      return { activeChatId: chat.id, chats: [chat] }
    }
    return empty
  } catch {
    return empty
  }
}

function writeStoredChats(collectionId: string, state: StoredChatState) {
  if (typeof window === "undefined") return
  try {
    window.localStorage.setItem(
      chatStorageKey(collectionId),
      JSON.stringify(state),
    )
  } catch {
    // Private browsing and full storage must not break the chat.
  }
}

// ── Icons ──────────────────────────────────────────────────────────────────

function IconFolder({
  size = 16,
  className = "",
}: {
  size?: number
  className?: string
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      className={className}
    >
      <path
        d="M1.5 3.5A1 1 0 0 1 2.5 2.5H6l1.5 1.5H13.5A1 1 0 0 1 14.5 5v7a1 1 0 0 1-1 1H2.5a1 1 0 0 1-1-1V3.5z"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function IconPlus({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none">
      <path
        d="M8 3v10M3 8h10"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  )
}

function IconUpload({ size = 24 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <path
        d="M12 16V8M12 8l-3 3M12 8l3 3"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  )
}

function IconSend({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none">
      <path
        d="M14 8L2 2l3 6-3 6 12-6z"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinejoin="round"
        fill="currentColor"
      />
    </svg>
  )
}

function IconCheck({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 14 14" fill="none">
      <path
        d="M2.5 7l3 3 6-6"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function IconDocument({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 14 14" fill="none">
      <rect
        x="2.5"
        y="1.5"
        width="9"
        height="11"
        rx="1"
        stroke="currentColor"
        strokeWidth="1.2"
      />
      <path
        d="M4.5 5h5M4.5 7.5h5M4.5 10h3"
        stroke="currentColor"
        strokeWidth="1"
        strokeLinecap="round"
      />
    </svg>
  )
}

function IconSpinner({ size = 16 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      className="animate-spin"
    >
      <circle
        cx="8"
        cy="8"
        r="6"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeOpacity="0.25"
      />
      <path
        d="M8 2a6 6 0 0 1 6 6"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  )
}

// ── Sub-components ─────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: GraphRAGStatus }) {
  const configs = {
    idle: {
      label: "Keine Dokumente",
      color: "text-[var(--muted-foreground)] bg-[var(--muted)]",
    },
    processing: {
      label: "Dokumente werden verarbeitet…",
      color: "text-amber-700 bg-amber-50",
    },
    building: {
      label: "GraphRAG wird aktualisiert…",
      color: "text-blue-700 bg-blue-50",
    },
    ready: {
      label: "GraphRAG aktuell",
      color: "text-emerald-700 bg-emerald-50",
    },
    error: {
      label: "Aktualisierung fehlgeschlagen",
      color: "text-red-700 bg-red-50",
    },
  }
  const { label, color } = configs[status]
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-mono font-medium ${color}`}
    >
      {status === "processing" || status === "building" ? (
        <IconSpinner size={10} />
      ) : status === "ready" ? (
        <IconCheck size={10} />
      ) : null}
      {label}
    </span>
  )
}

function CitationLink({
  citation,
  index,
  onSelect,
}: {
  citation: Citation
  index: number
  onSelect: (selection: CitationSelection) => void
}) {
  return (
    <button
      type="button"
      onClick={() => onSelect({ citation, index })}
      title={`${citation.document} · Seite ${citation.page ?? "?"}`}
      aria-label={`Quelle ${index + 1}: ${citation.document}, Seite ${citation.page ?? "?"}`}
      className="inline-flex items-center rounded px-1.5 py-0.5 font-mono text-[11px] text-[var(--accent)] underline decoration-dotted underline-offset-2 hover:bg-[var(--accent)]/10"
    >
      [{index + 1}]
    </button>
  )
}

function AnswerWithCitations({
  answer,
  citations,
  onCitationSelect,
}: {
  answer: string
  citations: Citation[]
  onCitationSelect: (selection: CitationSelection) => void
}) {
  const parts = answer.split(/(\[ID:\d+\])/g)
  const markerPattern = /^\[ID:(\d+)\]$/
  const citationOrder = new Map<number, number>()
  for (const part of parts) {
    const match = markerPattern.exec(part)
    if (!match) continue
    const citationIndex = Number(match[1])
    if (citations[citationIndex] && !citationOrder.has(citationIndex)) {
      citationOrder.set(citationIndex, citationOrder.size)
    }
  }
  return (
    <p className="leading-relaxed whitespace-pre-wrap">
      {parts.map((part, partIndex) => {
        const match = markerPattern.exec(part)
        if (!match) return <span key={partIndex}>{part}</span>
        const citationIndex = Number(match[1])
        const citation = citations[citationIndex]
        const displayIndex = citationOrder.get(citationIndex)
        if (!citation || displayIndex === undefined) {
          return <span key={partIndex}>{part}</span>
        }
        return (
          <CitationLink
            key={partIndex}
            citation={citation}
            index={displayIndex}
            onSelect={onCitationSelect}
          />
        )
      })}
    </p>
  )
}

// ── Upload Panel ───────────────────────────────────────────────────────────

function UploadPanel({
  collection,
  onUpload,
}: {
  collection: Collection
  onUpload: (files: File[]) => Promise<void>
}) {
  const [files, setFiles] = useState<File[]>([])
  const [status, setStatus] = useState<GraphRAGStatus>(collection.status)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState("")
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    setStatus(collection.status)
    if (collection.status === "ready") setFiles([])
  }, [collection.id, collection.status])

  function handleFiles(fileList: FileList | null) {
    if (!fileList) return
    setError("")
    setFiles(Array.from(fileList))
  }

  async function handleUpload() {
    if (files.length === 0) return
    setError("")
    setStatus("processing")
    try {
      await onUpload(files)
    } catch (uploadError) {
      setStatus("idle")
      setError(
        uploadError instanceof Error
          ? uploadError.message
          : "Upload fehlgeschlagen",
      )
    }
  }

  if (status === "ready") {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-6 p-8">
        <div className="flex flex-col items-center gap-3 text-center max-w-sm">
          <div className="w-12 h-12 rounded-full bg-emerald-50 border border-emerald-200 flex items-center justify-center">
            <IconCheck size={20} />
          </div>
          <div>
            <p className="font-semibold text-[15px]">GraphRAG aktuell</p>
            <p className="text-[var(--muted-foreground)] text-[13px] mt-1">
              {collection.documents.length} Dokument
              {collection.documents.length === 1 ? "" : "e"} indiziert. Wählen
              Sie ein Modell und starten Sie den Chat.
            </p>
          </div>
        </div>
        <div className="w-full max-w-sm divide-y divide-[var(--border)] rounded border border-[var(--border)] bg-[var(--card)]">
          {collection.documents.map((doc) => (
            <div
              key={doc}
              className="flex items-center gap-2 px-3 py-2 text-[13px]"
            >
              <IconDocument size={13} />
              <span className="truncate font-mono text-[12px]">{doc}</span>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (status === "processing" || status === "building") {
    const step = status === "processing" ? 1 : 2
    return (
      <div className="flex flex-col items-center justify-center h-full gap-6 p-8">
        <div className="flex flex-col items-center gap-4 text-center max-w-sm w-full">
          <IconSpinner size={32} />
          <div>
            <p className="font-semibold text-[15px]">
              {status === "processing"
                ? "Dateien werden verarbeitet"
                : "GraphRAG wird aufgebaut"}
            </p>
            <p className="text-[var(--muted-foreground)] text-[13px] mt-1">
              {status === "processing"
                ? "Dokumente werden analysiert und aufbereitet…"
                : "Wissensstruktur wird erstellt. Das kann einen Moment dauern…"}
            </p>
          </div>
          <div className="w-full bg-[var(--muted)] rounded-full h-1.5">
            <div
              className="h-1.5 rounded-full bg-[var(--accent)] transition-all duration-1000"
              style={{ width: step === 1 ? "35%" : "75%" }}
            />
          </div>
          <div className="flex gap-6 text-[12px] font-mono text-[var(--muted-foreground)]">
            <span className={step >= 1 ? "text-[var(--accent)]" : ""}>
              1 — Verarbeitung
            </span>
            <span className={step >= 2 ? "text-[var(--accent)]" : ""}>
              2 — GraphRAG-Aufbau
            </span>
            <span>3 — Bereit</span>
          </div>
          <div className="w-full divide-y divide-[var(--border)] rounded border border-[var(--border)] bg-[var(--card)] mt-2">
            {files.map((f) => (
              <div
                key={f.name}
                className="flex items-center gap-2 px-3 py-2 text-[12px] font-mono"
              >
                <IconDocument size={12} />
                <span className="truncate">{f.name}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full p-8 gap-6">
      <div>
        <h2 className="text-[17px] font-semibold">Dokumente hochladen</h2>
        <p className="text-[13px] text-[var(--muted-foreground)] mt-1">
          Sammlung:{" "}
          <span className="font-medium text-[var(--foreground)]">
            {collection.name}
          </span>
        </p>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          handleFiles(e.dataTransfer.files)
        }}
        onClick={() => inputRef.current?.click()}
        className={`flex-1 min-h-[220px] flex flex-col items-center justify-center gap-4 rounded border-2 border-dashed cursor-pointer transition-colors ${
          dragging
            ? "border-[var(--accent)] bg-[var(--accent)]/5"
            : "border-[var(--border)] hover:border-[var(--accent)]/50 hover:bg-[var(--secondary)]"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <IconUpload size={32} />
        <div className="text-center">
          <p className="font-medium text-[14px]">PDF-Dateien hierher ziehen</p>
          <p className="text-[13px] text-[var(--muted-foreground)] mt-0.5">
            oder klicken zum Auswählen
          </p>
        </div>
        <span className="font-mono text-[11px] text-[var(--muted-foreground)] bg-[var(--muted)] px-2 py-0.5 rounded">
          nur textbasierte PDFs
        </span>
      </div>

      {files.length > 0 && (
        <div className="rounded border border-[var(--border)] bg-[var(--card)] divide-y divide-[var(--border)]">
          {files.map((f) => (
            <div
              key={f.name}
              className="flex items-center gap-2 px-3 py-2 text-[13px]"
            >
              <IconDocument size={13} />
              <span className="truncate font-mono text-[12px]">{f.name}</span>
            </div>
          ))}
        </div>
      )}

      {(error || collection.job?.error) && (
        <p className="text-[12px] text-red-700 bg-red-50 border border-red-200 rounded px-3 py-2">
          {error || collection.job?.error}
        </p>
      )}

      <button
        onClick={handleUpload}
        disabled={files.length === 0}
        className="flex items-center justify-center gap-2 px-4 py-2.5 rounded bg-[var(--primary)] text-[var(--primary-foreground)] text-[14px] font-medium disabled:opacity-40 disabled:cursor-not-allowed hover:bg-[var(--accent)] transition-colors"
      >
        <IconUpload size={16} />
        {files.length === 0
          ? "Dateien auswählen"
          : `${files.length} Datei${files.length === 1 ? "" : "en"} hochladen`}
      </button>
    </div>
  )
}

// ── Chat Panel ─────────────────────────────────────────────────────────────

function ChatPanel({
  collection,
  models,
  onCitationSelect,
}: {
  collection: Collection
  models: Model[]
  onCitationSelect: (selection: CitationSelection | null) => void
}) {
  const [chatState, setChatState] = useState<StoredChatState>(() =>
    readStoredChats(collection.id),
  )
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const conversationId = useRef(crypto.randomUUID())
  const skipChatPersistRef = useRef(false)
  const graphUpdating =
    collection.status === "processing" || collection.status === "building"

  const chat =
    chatState.chats.find((item) => item.id === chatState.activeChatId) ??
    chatState.chats[0]
  const activeChatId = chat?.id ?? chatState.activeChatId
  const messages = chat?.messages ?? []
  const configurableModels = models.filter(
    (model) => model.configured !== false,
  )
  const selectedModel =
    configurableModels.find((model) => model.id === chat?.modelId) ??
    configurableModels[0] ??
    null

  function updateChat(
    chatId: string,
    update: (current: StoredChat) => StoredChat,
  ) {
    setChatState((current) => ({
      ...current,
      chats: current.chats.map((item) =>
        item.id === chatId ? update(item) : item,
      ),
    }))
  }

  function selectChat(chatId: string) {
    if (loading || chatId === activeChatId) return
    setChatState((current) =>
      current.chats.some((item) => item.id === chatId)
        ? { ...current, activeChatId: chatId }
        : current,
    )
  }

  function createChat() {
    if (loading) return
    const next = newStoredChat()
    setChatState((current) => ({
      activeChatId: next.id,
      chats: [next, ...current.chats],
    }))
  }

  useEffect(() => {
    skipChatPersistRef.current = true
    setChatState(readStoredChats(collection.id))
  }, [collection.id])

  useEffect(() => {
    setInput("")
    onCitationSelect(null)
    conversationId.current = crypto.randomUUID()
  }, [collection.id, activeChatId, onCitationSelect])

  useEffect(() => {
    if (skipChatPersistRef.current) {
      skipChatPersistRef.current = false
      return
    }
    writeStoredChats(collection.id, chatState)
  }, [collection.id, chatState])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  async function sendMessage() {
    const question = input.trim()
    if (!question || !selectedModel || loading || graphUpdating || !chat) return
    const chatId = activeChatId
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: question,
    }
    const history = [...messages, userMsg].map(({ role, content }) => ({
      role,
      content,
    }))
    updateChat(chatId, (current) => ({
      ...current,
      messages: [...current.messages, userMsg],
    }))
    setInput("")
    setLoading(true)
    onCitationSelect(null)

    try {
      const response = await sendChat(
        collection.id,
        selectedModel.id,
        question,
        conversationId.current,
        history,
      )
      const assistantMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: response.answer,
        citations: response.citations,
        model: `${response.model.name} · ${response.model.provider}`,
      }
      updateChat(chatId, (current) => ({
        ...current,
        messages: [...current.messages, assistantMsg],
      }))
    } catch (chatError) {
      const message =
        chatError instanceof Error
          ? chatError.message
          : "Chat-Anfrage fehlgeschlagen"
      updateChat(chatId, (current) => ({
        ...current,
        messages: [
          ...current.messages,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: `Fehler: ${message}`,
          },
        ],
      }))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Chat header */}
      <div className="flex items-center gap-3 px-6 py-3 border-b border-[var(--border)] shrink-0">
        <div className="flex-1 min-w-0">
          <p className="text-[13px] font-medium truncate">{collection.name}</p>
          <div className="flex items-center gap-2 mt-1">
            <StatusBadge status={collection.status} />
          </div>
          {graphUpdating && collection.documents.length > 0 && (
            <p className="mt-1 text-[11px] text-[var(--muted-foreground)]">
              Die Sammlung wird aktualisiert. Der Chat ist pausiert, bis der
              GraphRAG wieder aktuell ist.
            </p>
          )}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <label
            htmlFor="chat-select"
            className="font-mono text-[10px] text-[var(--muted-foreground)] uppercase tracking-wider hidden sm:block"
          >
            Chat
          </label>
          <select
            id="chat-select"
            value={activeChatId}
            onChange={(event) => selectChat(event.target.value)}
            disabled={loading}
            aria-label="Chat auswählen"
            className="w-36 rounded border border-[var(--border)] bg-[var(--card)] px-2 py-1.5 text-[12px] text-[var(--foreground)] focus:outline-none focus:border-[var(--accent)] disabled:opacity-60"
          >
            {chatState.chats.map((item, index) => {
              const firstQuestion = item.messages.find(
                (message) => message.role === "user",
              )?.content
              const label = firstQuestion?.trim()
                ? firstQuestion.trim().slice(0, 28)
                : `Chat ${index + 1}`
              return (
                <option key={item.id} value={item.id}>
                  {label}
                </option>
              )
            })}
          </select>
          <button
            type="button"
            onClick={createChat}
            disabled={loading}
            aria-label="Neuen Chat anlegen"
            title="Neuen Chat anlegen"
            className="inline-flex h-7 w-7 items-center justify-center rounded border border-[var(--border)] text-[var(--muted-foreground)] hover:bg-[var(--secondary)] hover:text-[var(--foreground)] disabled:cursor-not-allowed disabled:opacity-40"
          >
            <IconPlus size={14} />
          </button>
        </div>
        {selectedModel && (
          <div className="flex items-center gap-1.5 font-mono text-[11px] text-[var(--muted-foreground)] bg-[var(--muted)] px-2 py-1 rounded shrink-0">
            <span>{selectedModel.name}</span>
            <span className="opacity-50">·</span>
            <span>{selectedModel.provider}</span>
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4 flex flex-col gap-4">
        {messages.length === 0 && (
          <div className="flex-1 flex flex-col items-center justify-center gap-2 text-center py-12">
            <p className="text-[14px] font-medium">
              {selectedModel ? "Chat bereit" : "Modell wählen und loslegen"}
            </p>
            <p className="text-[13px] text-[var(--muted-foreground)] max-w-xs">
              {selectedModel
                ? "Stellen Sie eine Frage oder geben Sie einen Zusammenfassungsauftrag ein."
                : "Wählen Sie unten ein Modell aus, dann können Sie eine Frage stellen."}
            </p>
          </div>
        )}
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex flex-col gap-1 ${
              msg.role === "user" ? "items-end" : "items-start"
            }`}
          >
            <div
              className={`max-w-[85%] rounded px-4 py-3 text-[14px] leading-relaxed ${
                msg.role === "user"
                  ? "bg-[var(--primary)] text-[var(--primary-foreground)]"
                  : "bg-[var(--card)] border border-[var(--border)]"
              }`}
            >
              {msg.role === "assistant" ? (
                <AnswerWithCitations
                  answer={msg.content}
                  citations={msg.citations ?? []}
                  onCitationSelect={onCitationSelect}
                />
              ) : (
                msg.content
              )}
            </div>
            {msg.role === "assistant" && msg.model && (
              <div className="w-full max-w-[85%]">
                <span className="font-mono text-[10px] text-[var(--muted-foreground)] px-1">
                  KI-generiert · {msg.model}
                </span>
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="flex items-center gap-2 text-[13px] text-[var(--muted-foreground)]">
            <IconSpinner size={14} />
            <span>Antwort wird generiert…</span>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input + model selector */}
      <div className="px-6 py-4 border-t border-[var(--border)] shrink-0 flex flex-col gap-3">
        {/* Textarea + send */}
        <div className="flex gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault()
                sendMessage()
              }
            }}
            placeholder={
              selectedModel
                ? graphUpdating
                  ? "GraphRAG wird aktualisiert…"
                  : "Frage stellen oder Zusammenfassung anfordern…"
                : "Erst Modell auswählen…"
            }
            disabled={!selectedModel || graphUpdating}
            rows={2}
            className="flex-1 resize-none rounded border border-[var(--border)] bg-[var(--card)] px-3 py-2 text-[14px] placeholder:text-[var(--muted-foreground)] focus:outline-none focus:border-[var(--accent)] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <button
            onClick={sendMessage}
            disabled={
              !input.trim() || !selectedModel || loading || graphUpdating
            }
            className="px-3 py-2 rounded bg-[var(--primary)] text-[var(--primary-foreground)] disabled:opacity-40 disabled:cursor-not-allowed hover:bg-[var(--accent)] transition-colors self-end"
          >
            <IconSend size={15} />
          </button>
        </div>

        {/* Model select */}
        <div className="flex items-center gap-3">
          <label
            htmlFor="model-select"
            className="font-mono text-[10px] text-[var(--muted-foreground)] uppercase tracking-wider shrink-0"
          >
            Modell
          </label>
          <select
            id="model-select"
            value={selectedModel?.id ?? ""}
            onChange={(e) => {
              const found =
                configurableModels.find(
                  (model) => model.id === e.target.value,
                ) ?? null
              updateChat(activeChatId, (current) => ({
                ...current,
                modelId: found?.id ?? null,
              }))
            }}
            disabled={messages.length > 0 || loading || graphUpdating}
            className="w-52 shrink-0 rounded border border-[var(--border)] bg-[var(--card)] px-2 py-1.5 text-[13px] text-[var(--foreground)] focus:outline-none focus:border-[var(--accent)] transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
          >
            <option value="">— Modell wählen —</option>
            {models.map((model) => {
              const isConfigured = model.configured !== false
              return (
                <option
                  key={model.id}
                  value={model.id}
                  disabled={!isConfigured}
                >
                  {model.name} · {model.provider}
                  {isConfigured ? "" : " (noch nicht in RAGFlow aktiviert)"}
                </option>
              )
            })}
          </select>
          <p className="font-mono text-[10px] text-[var(--muted-foreground)] hidden sm:block shrink-0">
            Nur Dokumenteninhalte · Chat lokal je Sammlung
          </p>
        </div>
      </div>
    </div>
  )
}

// ── Sources Mock Panel ─────────────────────────────────────────────────────

function SourceImageModal({
  citation,
  onClose,
}: {
  citation: Citation
  onClose: () => void
}) {
  const closeButtonRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    closeButtonRef.current?.focus()
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") onClose()
    }
    document.addEventListener("keydown", closeOnEscape)
    return () => document.removeEventListener("keydown", closeOnEscape)
  }, [onClose])

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="source-image-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <div className="relative max-h-[calc(100vh-2rem)] max-w-[calc(100vw-2rem)] overflow-auto rounded bg-[var(--card)] shadow-2xl">
        <button
          ref={closeButtonRef}
          type="button"
          onClick={onClose}
          aria-label="Screenshot schließen"
          className="absolute right-3 top-3 z-10 inline-flex h-8 w-8 items-center justify-center rounded-full bg-black/70 text-lg leading-none text-white hover:bg-black"
        >
          <span aria-hidden="true">×</span>
        </button>
        <div className="p-4">
          <img
            src={citation.image_url}
            alt={`RAGFlow-Quellenausschnitt aus ${citation.document}`}
            className="block h-auto w-auto max-w-none"
          />
          <div
            id="source-image-title"
            className="mt-4 border-t border-[var(--border)] pt-3"
          >
            <p className="text-[13px] font-medium break-words">
              {citation.document}
            </p>
            <p className="mt-1 font-mono text-[11px] text-[var(--muted-foreground)]">
              Seite {citation.page ?? "?"}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

function SourcesPanel({ selection }: { selection: CitationSelection | null }) {
  const [imageOpen, setImageOpen] = useState(false)

  return (
    <div className="flex flex-col h-full border-l border-[var(--border)]">
      <div className="px-4 py-3 border-b border-[var(--border)]">
        <p className="text-[13px] font-semibold">Quelle</p>
        <p className="text-[10px] font-mono text-[var(--muted-foreground)] mt-1">
          Quellenausschnitt zum ausgewählten Inline-Zitat
        </p>
      </div>

      {selection ? (
        <div className="flex-1 overflow-y-auto p-4">
          <div className="rounded border border-[var(--border)] bg-[var(--card)] overflow-hidden">
            <div className="bg-[var(--secondary)] px-3 py-2 flex items-start gap-2 border-b border-[var(--border)]">
              <span className="font-mono text-[10px] text-[var(--muted-foreground)] bg-[var(--muted)] px-1.5 py-0.5 rounded shrink-0">
                [{selection.index + 1}]
              </span>
              <span className="mt-0.5 shrink-0">
                <IconDocument size={12} />
              </span>
              <div className="min-w-0">
                {selection.citation.document_url ? (
                  <a
                    href={`${selection.citation.document_url}${
                      selection.citation.page === null
                        ? ""
                        : `#page=${selection.citation.page}`
                    }`}
                    target="_blank"
                    rel="noopener noreferrer"
                    title="Gesamtes Dokument öffnen"
                    className="text-[11px] font-medium break-words underline decoration-dotted underline-offset-2 hover:text-[var(--accent)]"
                  >
                    {selection.citation.document}
                  </a>
                ) : (
                  <p className="text-[11px] font-medium break-words">
                    {selection.citation.document}
                  </p>
                )}
                <p className="text-[10px] font-mono text-[var(--muted-foreground)] mt-0.5">
                  Seite {selection.citation.page ?? "?"}
                </p>
              </div>
            </div>
            <div className="p-3">
              <p className="text-[12px] text-[var(--muted-foreground)] leading-relaxed whitespace-pre-wrap">
                &ldquo;
                {selection.citation.excerpt || "Kein Textausschnitt verfügbar."}
                &rdquo;
              </p>
            </div>
            {selection.citation.image_url && (
              <div className="border-t border-[var(--border)] bg-[var(--secondary)] p-3">
                <p className="mb-2 font-mono text-[10px] uppercase tracking-wider text-[var(--muted-foreground)]">
                  RAGFlow-Screenshot
                </p>
                <button
                  type="button"
                  onClick={() => setImageOpen(true)}
                  aria-label="RAGFlow-Screenshot vergrößern"
                  title="Screenshot vergrößern"
                  className="block w-full cursor-zoom-in text-left"
                >
                  <img
                    src={selection.citation.image_url}
                    alt={`RAGFlow-Quellenausschnitt aus ${selection.citation.document}`}
                    className="h-auto max-h-[28rem] w-full rounded border border-[var(--border)] bg-white object-contain"
                  />
                </button>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center p-4 text-center">
          <p className="text-[12px] text-[var(--muted-foreground)]">
            Klicken Sie auf ein Inline-Zitat, um den Quellenausschnitt
            anzuzeigen.
          </p>
        </div>
      )}
      {imageOpen && selection?.citation.image_url && (
        <SourceImageModal
          citation={selection.citation}
          onClose={() => setImageOpen(false)}
        />
      )}
    </div>
  )
}

// ── Main App ───────────────────────────────────────────────────────────────

export default function App() {
  const [collections, setCollections] = useState<Collection[]>([])
  const [models, setModels] = useState<Model[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [showNewForm, setShowNewForm] = useState(false)
  const [newName, setNewName] = useState("")
  const [selectedCitation, setSelectedCitation] =
    useState<CitationSelection | null>(null)
  const [expandedCollections, setExpandedCollections] = useState<Set<string>>(
    new Set(),
  )
  const [uploadTargetId, setUploadTargetId] = useState<string | null>(null)
  const [uploadingCollectionId, setUploadingCollectionId] =
    useState<string | null>(null)
  const [deletingDocumentKey, setDeletingDocumentKey] = useState<string | null>(
    null,
  )
  const [sourceWidth, setSourceWidth] = useState(256)
  const [resizing, setResizing] = useState(false)
  const layoutRef = useRef<HTMLDivElement>(null)
  const sidebarUploadRef = useRef<HTMLInputElement>(null)
  const [error, setError] = useState("")

  function maxSourceWidth() {
    const layoutWidth = layoutRef.current?.clientWidth ?? 1200
    return Math.max(220, Math.min(640, layoutWidth - 224 - 8 - 320))
  }

  function resizeSource(event: React.PointerEvent<HTMLDivElement>) {
    if (!resizing || !layoutRef.current) return
    const bounds = layoutRef.current.getBoundingClientRect()
    const nextWidth = bounds.right - event.clientX
    setSourceWidth(Math.min(maxSourceWidth(), Math.max(220, nextWidth)))
  }

  function stopResizing(event?: React.PointerEvent<HTMLDivElement>) {
    if (event?.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
    setResizing(false)
  }

  function resizeWithKeyboard(event: React.KeyboardEvent<HTMLDivElement>) {
    const maximum = maxSourceWidth()
    if (event.key === "ArrowLeft") {
      event.preventDefault()
      setSourceWidth((width) => Math.min(maximum, width + 16))
    } else if (event.key === "ArrowRight") {
      event.preventDefault()
      setSourceWidth((width) => Math.max(220, width - 16))
    } else if (event.key === "Home") {
      event.preventDefault()
      setSourceWidth(220)
    } else if (event.key === "End") {
      event.preventDefault()
      setSourceWidth(maximum)
    }
  }

  useEffect(() => {
    let active = true
    Promise.all([listCollections(), listModels()])
      .then(([loadedCollections, loadedModels]) => {
        if (!active) return
        setCollections(loadedCollections)
        setModels(loadedModels)
        setSelectedId((current) => current ?? loadedCollections[0]?.id ?? null)
        if (loadedCollections[0]) {
          setExpandedCollections((current) =>
            current.size > 0 ? current : new Set([loadedCollections[0].id]),
          )
        }
      })
      .catch((loadError) => {
        if (active) {
          setError(
            loadError instanceof Error
              ? loadError.message
              : "Backend nicht erreichbar",
          )
        }
      })
    return () => {
      active = false
    }
  }, [])

  const selectedCollection =
    collections.find((c) => c.id === selectedId) ?? null
  const selectedStatus = selectedCollection?.status

  useEffect(() => {
    if (
      !selectedId ||
      (selectedStatus !== "processing" && selectedStatus !== "building")
    ) {
      return
    }
    let active = true
    const timer = window.setInterval(() => {
      getCollection(selectedId)
        .then((updated) => {
          if (!active) return
          setCollections((current) =>
            current.map((collection) =>
              collection.id === updated.id ? updated : collection,
            ),
          )
        })
        .catch(() => undefined)
    }, 1500)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [selectedId, selectedStatus])

  async function handleCreateCollection() {
    const trimmed = newName.trim()
    if (!trimmed) return
    setError("")
    try {
      const collection = await createCollection(trimmed)
      setCollections((current) => [...current, collection])
      setSelectedId(collection.id)
      setExpandedCollections((current) => new Set(current).add(collection.id))
      setNewName("")
      setShowNewForm(false)
    } catch (createError) {
      setError(
        createError instanceof Error
          ? createError.message
          : "Sammlung konnte nicht angelegt werden",
      )
    }
  }

  async function handleUpload(files: File[]) {
    if (!selectedId) return
    setError("")
    const updated = await uploadCollection(selectedId, files)
    setCollections((current) =>
      current.map((collection) =>
        collection.id === updated.id ? updated : collection,
      ),
    )
  }

  function selectCollection(id: string) {
    setSelectedId(id)
    setExpandedCollections((current) => new Set(current).add(id))
  }

  function toggleCollection(id: string) {
    setSelectedId(id)
    setExpandedCollections((current) => {
      const next = new Set(current)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  function openCollectionUpload(id: string) {
    setUploadTargetId(id)
    selectCollection(id)
    sidebarUploadRef.current?.click()
  }

  async function handleSidebarUpload(fileList: FileList | null) {
    const collectionId = uploadTargetId
    if (!collectionId || !fileList || fileList.length === 0) return
    setError("")
    setUploadingCollectionId(collectionId)
    try {
      const updated = await uploadCollection(collectionId, Array.from(fileList))
      setCollections((current) =>
        current.map((collection) =>
          collection.id === updated.id ? updated : collection,
        ),
      )
      setSelectedId(collectionId)
      setExpandedCollections((current) => new Set(current).add(collectionId))
    } catch (uploadError) {
      setError(
        uploadError instanceof Error
          ? uploadError.message
          : "Upload fehlgeschlagen",
      )
    } finally {
      setUploadingCollectionId(null)
      setUploadTargetId(null)
      if (sidebarUploadRef.current) sidebarUploadRef.current.value = ""
    }
  }

  async function handleDeleteDocument(
    collectionId: string,
    documentId: string,
    documentName: string,
  ) {
    if (
      !window.confirm(`Dokument „${documentName}“ aus der Sammlung entfernen?`)
    ) {
      return
    }
    const key = `${collectionId}:${documentId}`
    setError("")
    setDeletingDocumentKey(key)
    setSelectedId(collectionId)
    setSelectedCitation(null)
    try {
      const updated = await deleteCollectionDocument(collectionId, documentId)
      setCollections((current) =>
        current.map((collection) =>
          collection.id === updated.id ? updated : collection,
        ),
      )
    } catch (deleteError) {
      setError(
        deleteError instanceof Error
          ? deleteError.message
          : "Dokument konnte nicht entfernt werden",
      )
    } finally {
      setDeletingDocumentKey(null)
    }
  }

  const graphUpdating =
    selectedCollection?.status === "processing" ||
    selectedCollection?.status === "building"
  const showChat =
    selectedCollection?.status === "ready" ||
    (Boolean(selectedCollection?.documents.length) && graphUpdating)

  return (
    <div className="h-full flex flex-col">
      {/* Top bar */}
      <header className="flex items-center gap-4 px-6 h-12 border-b border-[var(--border)] bg-[var(--card)] shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-[15px] font-bold tracking-tight text-[var(--primary)]">
            DEval
          </span>
          <span className="text-[var(--border)]">/</span>
          <span className="text-[14px] font-medium">Webchat</span>
        </div>
        {error && (
          <span className="ml-auto max-w-[50%] truncate font-mono text-[10px] text-red-700 bg-red-50 border border-red-200 px-2 py-0.5 rounded">
            {error}
          </span>
        )}
      </header>

      <div
        ref={layoutRef}
        className={`flex-1 flex overflow-hidden ${
          resizing ? "select-none" : ""
        }`}
      >
        {/* Sidebar */}
        <aside className="w-56 shrink-0 border-r border-[var(--border)] flex flex-col bg-[var(--card)]">
          <div className="px-4 py-3 border-b border-[var(--border)] flex items-center justify-between">
            <span className="text-[11px] font-mono font-medium text-[var(--muted-foreground)] uppercase tracking-wider">
              Sammlungen
            </span>
            <button
              onClick={() => setShowNewForm(true)}
              className="text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors"
              title="Neue Sammlung"
            >
              <IconPlus size={15} />
            </button>
          </div>

          {showNewForm && (
            <div className="px-3 py-2 border-b border-[var(--border)] flex flex-col gap-1.5">
              <input
                autoFocus
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleCreateCollection()
                  if (e.key === "Escape") {
                    setShowNewForm(false)
                    setNewName("")
                  }
                }}
                placeholder="Name der Sammlung"
                className="w-full text-[13px] px-2 py-1.5 rounded border border-[var(--border)] bg-[var(--background)] focus:outline-none focus:border-[var(--accent)]"
              />
              <div className="flex gap-1">
                <button
                  onClick={handleCreateCollection}
                  className="flex-1 text-[11px] py-1 rounded bg-[var(--primary)] text-[var(--primary-foreground)] hover:bg-[var(--accent)] transition-colors"
                >
                  Anlegen
                </button>
                <button
                  onClick={() => {
                    setShowNewForm(false)
                    setNewName("")
                  }}
                  className="flex-1 text-[11px] py-1 rounded border border-[var(--border)] hover:bg-[var(--secondary)] transition-colors"
                >
                  Abbrechen
                </button>
              </div>
            </div>
          )}

          <input
            ref={sidebarUploadRef}
            type="file"
            multiple
            accept=".pdf"
            className="hidden"
            onChange={(event) => handleSidebarUpload(event.target.files)}
          />
          <nav className="flex-1 overflow-y-auto py-1">
            {collections.map((col) => {
              const expanded = expandedCollections.has(col.id)
              const busy =
                col.status === "processing" || col.status === "building"
              const uploading = uploadingCollectionId === col.id
              const selected = selectedId === col.id
              return (
                <div
                  key={col.id}
                  className={`text-left transition-colors ${
                    selected
                      ? "bg-[var(--primary)] text-[var(--primary-foreground)]"
                      : ""
                  }`}
                >
                  <div className="flex items-start">
                    <button
                      type="button"
                      onClick={() => toggleCollection(col.id)}
                      aria-expanded={expanded}
                      aria-label={`${
                        expanded ? "Schließen" : "Öffnen"
                      } ${col.name}`}
                      className="mt-2.5 ml-2 w-4 shrink-0 text-center text-[14px] leading-none opacity-70 hover:opacity-100"
                    >
                      <span
                        className={`inline-block transition-transform ${
                          expanded ? "rotate-90" : ""
                        }`}
                      >
                        ›
                      </span>
                    </button>
                    <button
                      type="button"
                      onClick={() => selectCollection(col.id)}
                      className="flex min-w-0 flex-1 items-start gap-2.5 px-2 py-2.5 text-left transition-colors hover:bg-black/5"
                    >
                      <IconFolder size={14} className="mt-0.5 shrink-0" />
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-[13px] font-medium">
                          {col.name}
                        </p>
                        <div
                          className={`mt-0.5 text-[10px] font-mono ${
                            selected
                              ? "text-[var(--primary-foreground)]/70"
                              : "text-[var(--muted-foreground)]"
                          }`}
                        >
                          {col.status === "ready"
                            ? `${col.documents.length} Dok.`
                            : col.status === "idle"
                              ? "leer"
                              : col.status === "error"
                                ? "Fehler"
                                : "lädt…"}
                        </div>
                      </div>
                    </button>
                    <button
                      type="button"
                      onClick={() => openCollectionUpload(col.id)}
                      disabled={busy || uploading}
                      aria-label={`Dokumente zu ${col.name} hinzufügen`}
                      title="Dokumente hinzufügen"
                      className="mr-2 mt-2 inline-flex h-6 w-6 shrink-0 items-center justify-center rounded text-[var(--muted-foreground)] hover:bg-black/10 hover:text-[var(--foreground)] disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      {uploading ? (
                        <IconSpinner size={13} />
                      ) : (
                        <IconPlus size={14} />
                      )}
                    </button>
                  </div>
                  {expanded && (
                    <div
                      className={`max-h-48 overflow-y-auto pb-1 pl-8 pr-2 ${
                        selected
                          ? "text-[var(--primary-foreground)]/80"
                          : "text-[var(--muted-foreground)]"
                      }`}
                    >
                      {col.document_records.length > 0 ? (
                        col.document_records.map((document) => {
                          const deleting =
                            deletingDocumentKey === `${col.id}:${document.id}`
                          return (
                            <div
                              key={document.id}
                              className="flex items-center gap-2 py-1.5 text-[11px]"
                            >
                              <IconDocument size={12} />
                              <span
                                className="min-w-0 flex-1 truncate font-mono"
                                title={document.name}
                              >
                                {document.name}
                              </span>
                              <button
                                type="button"
                                onClick={() =>
                                  handleDeleteDocument(
                                    col.id,
                                    document.id,
                                    document.name,
                                  )
                                }
                                disabled={busy || deleting}
                                aria-label={`${document.name} aus der Sammlung entfernen`}
                                title="Dokument entfernen"
                                className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded text-[13px] leading-none hover:bg-red-500/15 hover:text-red-700 disabled:cursor-not-allowed disabled:opacity-40"
                              >
                                {deleting ? (
                                  <IconSpinner size={11} />
                                ) : (
                                  <span aria-hidden="true">×</span>
                                )}
                              </button>
                            </div>
                          )
                        })
                      ) : (
                        <p className="py-2 text-[11px] italic opacity-70">
                          Keine Dokumente
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </nav>
        </aside>

        {/* Center */}
        <main className="flex-1 flex flex-col overflow-hidden bg-[var(--background)]">
          {selectedCollection ? (
            showChat ? (
              <ChatPanel
                collection={selectedCollection}
                models={models}
                onCitationSelect={setSelectedCitation}
              />
            ) : (
              <UploadPanel
                collection={selectedCollection}
                onUpload={handleUpload}
              />
            )
          ) : (
            <div className="flex-1 flex items-center justify-center text-[var(--muted-foreground)] text-[14px]">
              Sammlung auswählen oder neu anlegen
            </div>
          )}
        </main>

        {/* Resizable chat/source divider */}
        <div
          role="separator"
          aria-label="Breite zwischen Chat und Quelle ändern"
          aria-orientation="vertical"
          aria-valuemin={220}
          aria-valuemax={640}
          aria-valuenow={Math.round(sourceWidth)}
          tabIndex={0}
          onPointerDown={(event) => {
            event.preventDefault()
            event.currentTarget.setPointerCapture(event.pointerId)
            setResizing(true)
          }}
          onPointerMove={resizeSource}
          onPointerUp={stopResizing}
          onPointerCancel={stopResizing}
          onLostPointerCapture={() => setResizing(false)}
          onKeyDown={resizeWithKeyboard}
          className={`w-1 shrink-0 cursor-col-resize border-l border-[var(--border)] bg-transparent transition-colors hover:bg-[var(--accent)]/20 ${
            resizing ? "bg-[var(--accent)]/30" : ""
          }`}
          style={{ touchAction: "none" }}
        >
          <span className="mx-auto block h-8 w-px rounded-full bg-[var(--muted-foreground)]/40" />
        </div>

        {/* Sources panel */}
        <div
          className="shrink-0 overflow-hidden"
          style={{ width: `${sourceWidth}px` }}
        >
          <SourcesPanel selection={selectedCitation} />
        </div>
      </div>
    </div>
  )
}
