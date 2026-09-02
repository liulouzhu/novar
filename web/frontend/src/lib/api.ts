/** API 客户端 */

import { useAuthStore } from '@/stores/authStore'

const BASE = ''  // Vite proxy handles /api → localhost:8000

function authHeaders(): HeadersInit {
  const token = useAuthStore.getState().token
  return token ? { Authorization: `Bearer ${token}` } : {}
}

function handleAuthError(res: Response) {
  if (res.status === 401) {
    useAuthStore.getState().logout()
  }
}

export interface SessionMeta {
  session_id: string
  title: string
  message_count: number
  updated_at: string
}

export interface SessionDetail {
  session_id: string
  messages: Array<{
    role: string
    content: string
    tool_calls?: Array<{
      id: string
      type: string
      function: { name: string; arguments: string }
    }>
    tool_call_id?: string
  }>
  title: string
}

export interface HealthStatus {
  status: string
  model: string
  redis: {
    enabled: boolean
    available: boolean
    status: string
  }
  database: {
    status: string
  }
  sandbox: {
    available: boolean
  }
}

export async function fetchHealth(): Promise<HealthStatus> {
  const res = await fetch(`${BASE}/api/health`)
  if (!res.ok) throw new Error('Failed to fetch health')
  return res.json()
}

export interface Paper {
  id: string
  title: string
  authors: string[]
  abstract: string | null
  year: number | null
  source: string | null
  url: string | null
  pdf_path: string | null
  citation_count: number
  is_parsed: boolean
  created_at: string | null
}

export interface PaperFullTextSection {
  section: string
  text: string
  chunk_count: number
}

export interface PaperFullText {
  paper_id: string
  title: string
  sections: PaperFullTextSection[]
  content: string
}

// ── Auth ──

export async function register(username: string, email: string, password: string) {
  const res = await fetch(`${BASE}/api/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, email, password }),
  })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Registration failed')
  }
  return res.json()
}

export async function login(username: string, password: string): Promise<{ access_token: string }> {
  const res = await fetch(`${BASE}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  if (!res.ok) throw new Error('Invalid username or password')
  return res.json()
}

export async function fetchMe(token?: string): Promise<{ id: string; username: string; email: string }> {
  const hdrs: HeadersInit = token ? { Authorization: `Bearer ${token}` } : authHeaders()
  const res = await fetch(`${BASE}/api/auth/me`, { headers: hdrs })
  if (!res.ok) throw new Error('Not authenticated')
  return res.json()
}

// ── Skills ──

export interface SkillMeta {
  name: string
  description: string
}

export async function fetchSkills(): Promise<SkillMeta[]> {
  const res = await fetch(`${BASE}/api/skills`, { headers: authHeaders() })
  if (!res.ok) return []
  return res.json()
}

// ── Sessions ──

export async function fetchSessions(): Promise<SessionMeta[]> {
  const res = await fetch(`${BASE}/api/sessions`, { headers: authHeaders() })
  if (!res.ok) { handleAuthError(res); throw new Error('Failed to fetch sessions') }
  return res.json()
}

export async function createSession(): Promise<SessionMeta> {
  const res = await fetch(`${BASE}/api/sessions`, { method: 'POST', headers: authHeaders() })
  if (!res.ok) { handleAuthError(res); throw new Error('Failed to create session') }
  return res.json()
}

export async function fetchSession(sessionId: string): Promise<SessionDetail> {
  const res = await fetch(`${BASE}/api/sessions/${sessionId}`, { headers: authHeaders() })
  if (!res.ok) { handleAuthError(res); throw new Error('Failed to fetch session') }
  return res.json()
}

export async function deleteSession(sessionId: string): Promise<void> {
  const res = await fetch(`${BASE}/api/sessions/${sessionId}`, { method: 'DELETE', headers: authHeaders() })
  if (!res.ok) { handleAuthError(res); throw new Error('Failed to delete session') }
}

export async function renameSession(sessionId: string, title: string): Promise<void> {
  const res = await fetch(`${BASE}/api/sessions/${sessionId}`, {
    method: 'PATCH',
    headers: { ...authHeaders(), 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  })
  if (!res.ok) { handleAuthError(res); throw new Error('Failed to rename session') }
}

export async function flushMemoryExtraction(sessionId: string): Promise<{ status: string }> {
  const res = await fetch(`${BASE}/api/sessions/${sessionId}/memory/flush`, {
    method: 'POST',
    headers: authHeaders(),
  })
  if (!res.ok) { handleAuthError(res); throw new Error('Failed to flush memory extraction') }
  return res.json()
}

// ── Papers ──

export async function fetchPapers(params?: { q?: string; is_parsed?: boolean }): Promise<Paper[]> {
  const searchParams = new URLSearchParams()
  if (params?.q) searchParams.set('q', params.q)
  if (params?.is_parsed !== undefined) searchParams.set('is_parsed', String(params.is_parsed))
  const qs = searchParams.toString()
  const res = await fetch(`${BASE}/api/papers${qs ? `?${qs}` : ''}`, { headers: authHeaders() })
  if (!res.ok) { handleAuthError(res); throw new Error('Failed to fetch papers') }
  return res.json()
}

export async function fetchPaper(paperId: string): Promise<Paper> {
  const res = await fetch(`${BASE}/api/papers/${encodeURIComponent(paperId)}`, { headers: authHeaders() })
  if (!res.ok) { handleAuthError(res); throw new Error('Failed to fetch paper') }
  return res.json()
}

export async function fetchPaperFullText(paperId: string): Promise<PaperFullText> {
  const res = await fetch(`${BASE}/api/papers/${encodeURIComponent(paperId)}/fulltext`, { headers: authHeaders() })
  if (!res.ok) { handleAuthError(res); throw new Error('Failed to fetch paper full text') }
  return res.json()
}

export async function deletePaper(paperId: string): Promise<void> {
  const res = await fetch(`${BASE}/api/papers/${encodeURIComponent(paperId)}`, { method: 'DELETE', headers: authHeaders() })
  if (!res.ok) { handleAuthError(res); throw new Error('Failed to delete paper') }
}

// ── Upload ──

export interface UploadResult {
  upload_id: string
  filename: string
  already_uploaded: boolean
  file_path?: string | null
  message: string
}

export async function uploadFile(file: File): Promise<UploadResult> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/api/upload`, { method: 'POST', body: form, headers: authHeaders() })
  if (!res.ok) { handleAuthError(res); throw new Error('Failed to upload file') }
  return res.json()
}

// ── Knowledge Graph ──

export interface GraphNode {
  id: string
  type: string
  label: string
  name: string
  title: string
  year: number | null
  citation_count: number
  description: string
  canonical_name?: string
  aliases?: string[]
  source_mentions?: string[]
}

export interface GraphLink {
  source: string
  target: string
  type: string
  alternate_relations?: string[]
  confidence?: number | null
  inference?: string
  shared_tasks?: string[]
  shared_datasets?: string[]
  shared_methods?: string[]
  shared_metrics?: string[]
  evidence_note?: string
}

export interface GraphData {
  nodes: GraphNode[]
  links: GraphLink[]
}

export interface GraphStats {
  total_nodes: number
  total_edges: number
  node_types: Record<string, number>
  edge_types: Record<string, number>
}

export async function fetchGraph(exclude: string = 'Author'): Promise<GraphData> {
  const params = new URLSearchParams()
  params.set('exclude', exclude)
  const qs = params.toString()
  const res = await fetch(`${BASE}/api/graph${qs ? '?' + qs : ''}`, { headers: authHeaders() })
  if (!res.ok) { handleAuthError(res); throw new Error('Failed to fetch graph') }
  return res.json()
}

export async function fetchGraphStats(exclude: string = 'Author'): Promise<GraphStats> {
  const params = new URLSearchParams()
  params.set('exclude', exclude)
  const qs = params.toString()
  const res = await fetch(`${BASE}/api/graph/stats${qs ? '?' + qs : ''}`, { headers: authHeaders() })
  if (!res.ok) { handleAuthError(res); throw new Error('Failed to fetch graph stats') }
  return res.json()
}

// ── Memories ──

export interface MemoryItem {
  id: number
  category: string
  key: string
  value: string
  confidence: number
  pinned: boolean
  tags: string[]
  source: string
  created_at: string | null
  updated_at: string | null
}

export async function fetchMemories(): Promise<MemoryItem[]> {
  const res = await fetch(`${BASE}/api/memories`, { headers: authHeaders() })
  if (!res.ok) { handleAuthError(res); throw new Error('Failed to fetch memories') }
  return res.json()
}

export async function updateMemory(id: number, data: { value?: string; tags?: string[]; confidence?: number }): Promise<MemoryItem> {
  const res = await fetch(`${BASE}/api/memories/${id}`, {
    method: 'PATCH',
    headers: { ...authHeaders(), 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) { handleAuthError(res); throw new Error('Failed to update memory') }
  return res.json()
}

export async function deleteMemory(id: number): Promise<void> {
  const res = await fetch(`${BASE}/api/memories/${id}`, { method: 'DELETE', headers: authHeaders() })
  if (!res.ok) { handleAuthError(res); throw new Error('Failed to delete memory') }
}

export async function clearMemories(): Promise<{ deleted: number }> {
  const res = await fetch(`${BASE}/api/memories`, { method: 'DELETE', headers: authHeaders() })
  if (!res.ok) { handleAuthError(res); throw new Error('Failed to clear memories') }
  return res.json()
}

export async function togglePin(id: number): Promise<MemoryItem> {
  const res = await fetch(`${BASE}/api/memories/${id}/pin`, {
    method: 'PATCH',
    headers: authHeaders(),
  })
  if (!res.ok) { handleAuthError(res); throw new Error('Failed to toggle pin') }
  return res.json()
}

// ── 任务取消 / 状态查询 ─────────────────────────────────────────────────────

export async function cancelTask(sessionId: string): Promise<{ ok: boolean; reason?: string }> {
  const res = await fetch(`${BASE}/api/chat/${sessionId}/cancel`, {
    method: 'POST',
    headers: authHeaders(),
  })
  if (!res.ok) { handleAuthError(res); throw new Error('Failed to cancel task') }
  return res.json()
}

export async function getTaskStatus(sessionId: string): Promise<{ status: string; [key: string]: unknown }> {
  const res = await fetch(`${BASE}/api/chat/${sessionId}/task`, { headers: authHeaders() })
  if (!res.ok) { handleAuthError(res); throw new Error('Failed to get task status') }
  return res.json()
}
