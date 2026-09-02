/** 会话列表状态管理 */

import { create } from 'zustand'
import { type SessionMeta, fetchSessions, createSession as apiCreate, deleteSession as apiDelete, renameSession as apiRename, flushMemoryExtraction } from '@/lib/api'

/** Best-effort flush 前一个会话的记忆提取。失败不阻止后续操作。 */
function flushPreviousSession(oldId: string | null, newId: string | null) {
  if (oldId && newId && oldId !== newId) {
    flushMemoryExtraction(oldId).catch(() => {})
  }
}

interface SessionStore {
  sessions: SessionMeta[]
  currentId: string | null
  loading: boolean

  loadSessions: () => Promise<void>
  createSession: () => Promise<string>
  switchSession: (id: string) => void
  updateSessionTitle: (id: string, title: string) => void
  renameSession: (id: string, title: string) => Promise<void>
  deleteSession: (id: string) => Promise<void>
}

export const useSessionStore = create<SessionStore>((set, get) => ({
  sessions: [],
  currentId: null,
  loading: false,

  loadSessions: async () => {
    set({ loading: true })
    try {
      const sessions = await fetchSessions()
      set({ sessions, loading: false })
    } catch {
      set({ loading: false })
    }
  },

  createSession: async () => {
    const oldId = get().currentId
    const session = await apiCreate()
    flushPreviousSession(oldId, session.session_id)
    set((s) => ({ sessions: [session, ...s.sessions], currentId: session.session_id }))
    return session.session_id
  },

  switchSession: (id: string) => {
    const oldId = get().currentId
    flushPreviousSession(oldId, id)
    set({ currentId: id })
  },

  updateSessionTitle: (id: string, title: string) => {
    const normalized = title.trim().replace(/\s+/g, ' ')
    if (!normalized) return

    const nextTitle = normalized.length > 60 ? `${normalized.slice(0, 60)}...` : normalized
    set((s) => ({
      sessions: s.sessions.map((session) =>
        session.session_id === id
          ? {
              ...session,
              title: session.title === '新会话' || session.title === 'New Chat' || !session.title
                ? nextTitle
                : session.title,
              message_count: Math.max(session.message_count, 1),
            }
          : session,
      ),
    }))
  },

  renameSession: async (id: string, title: string) => {
    const normalized = title.trim().replace(/\s+/g, ' ')
    if (!normalized) return
    const nextTitle = normalized.length > 60 ? `${normalized.slice(0, 60)}...` : normalized
    await apiRename(id, nextTitle)
    set((s) => ({
      sessions: s.sessions.map((session) =>
        session.session_id === id ? { ...session, title: nextTitle } : session,
      ),
    }))
  },

  deleteSession: async (id: string) => {
    await apiDelete(id)
    set((s) => {
      const sessions = s.sessions.filter((x) => x.session_id !== id)
      const currentId = s.currentId === id ? (sessions[0]?.session_id ?? null) : s.currentId
      return { sessions, currentId }
    })
    // 删除会话不 flush — session 和 messages 已删除，flush 必然 404
    // 后端通过 forget_session 清理 scheduler 状态
  },
}))
