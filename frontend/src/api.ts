export type GraphRAGStatus = "idle" | "processing" | "building" | "outdated" | "ready" | "error"
export type GraphFreshness = "empty" | "current" | "updating" | "outdated" | "error"
export const CROSS_LANGUAGE_OPTIONS = ["German", "English"] as const
export type CrossLanguage = typeof CROSS_LANGUAGE_OPTIONS[number]

export interface GraphInfo {
  state: GraphFreshness
  remote_state: string
  progress: number | null
  message: string
  document_count: number
}

export interface CollectionDocument {
  id: string
  name: string
  state: string
}

export interface Collection {
  id: string
  name: string
  status: GraphRAGStatus
  documents: string[]
  document_records: CollectionDocument[]
  graph: GraphInfo
  job?: {
    id: string
    files: string[]
    operation?: "upload" | "delete"
    completed: number
    total: number
    error: string
  } | null
}

export interface Citation {
  document: string
  page: number | null
  excerpt: string
  resolved?: boolean
  document_url?: string
  image_id?: string
  image_url?: string
  source?: {
    image_id?: string
    positions?: unknown
    [key: string]: unknown
  }
}

export interface Model {
  id: string
  name: string
  provider: string
  description: string
  configured?: boolean
}

export interface ChatMessage {
  role: "user" | "assistant"
  content: string
}

export interface ChatResponse {
  answer: string
  model: Model
  citations: Citation[]
  cross_languages?: CrossLanguage[]
}

const API_BASE = "/api"

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  let payload: unknown = null
  try {
    payload = await response.json()
  } catch {
    // Keep the HTTP status as the useful error for non-JSON failures.
  }
  if (!response.ok) {
    const message =
      typeof payload === "object" && payload !== null && "error" in payload
        ? String(payload.error)
        : `API request failed (${response.status})`
    throw new Error(message)
  }
  return payload as T
}

export async function listCollections(): Promise<Collection[]> {
  const payload = await request<{ collections: Collection[] }>("/collections")
  return payload.collections
}

export async function getCollection(id: string): Promise<Collection> {
  return request<Collection>(`/collections/${encodeURIComponent(id)}`)
}

export async function createCollection(name: string): Promise<Collection> {
  return request<Collection>("/collections", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  })
}

export async function uploadCollection(
  id: string,
  files: File[],
): Promise<Collection> {
  const form = new FormData()
  files.forEach((file) => form.append("files", file, file.name))
  return request<Collection>(`/collections/${encodeURIComponent(id)}/upload`, {
    method: "POST",
    body: form,
  })
}

export async function deleteCollectionDocument(
  collectionId: string,
  documentId: string,
): Promise<Collection> {
  return request<Collection>(
    `/collections/${encodeURIComponent(collectionId)}/documents/${encodeURIComponent(documentId)}`,
    { method: "DELETE" },
  )
}

export async function listModels(): Promise<Model[]> {
  const payload = await request<{ models: Model[] }>("/models")
  return payload.models
}

export async function sendChat(
  collectionId: string,
  modelId: string,
  question: string,
  conversationId: string,
  history: ChatMessage[] = [],
  crossLanguages: CrossLanguage[] = [],
): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      collection_id: collectionId,
      model_id: modelId,
      question,
      conversation_id: conversationId,
      messages: history,
      cross_languages: crossLanguages,
    }),
  })
}
