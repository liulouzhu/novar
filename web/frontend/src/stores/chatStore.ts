/** 聊天消息状态管理 */

import { create } from 'zustand'
import { type Message, type ToolCallState, type TaskState, type MessagePart, type VerificationReport } from '@/lib/ws'
import { generateId } from '@/lib/utils'

interface ChatStore {
  // 按 session_id 存储消息
  messagesBySession: Record<string, Message[]>
  isStreaming: boolean
  streamingMessageId: string | null
  streamingTaskState: TaskState | null

  getMessages: (sessionId: string) => Message[]
  addUserMessage: (sessionId: string, content: string) => string
  startAssistantMessage: (sessionId: string) => string
  appendText: (messageId: string, content: string) => void
  addToolCall: (messageId: string, tool: ToolCallState) => void
  updateToolCall: (messageId: string, toolName: string, update: Partial<ToolCallState>) => void
  finishMessage: (messageId: string) => void
  setStreaming: (streaming: boolean) => void
  setMessages: (sessionId: string, messages: Message[]) => void
  updateTaskState: (messageId: string, state: TaskState) => void
  updateVerification: (messageId: string, report: VerificationReport) => void
  clearSession: (sessionId: string) => void
}

export const useChatStore = create<ChatStore>((set, get) => ({
  messagesBySession: {},
  isStreaming: false,
  streamingMessageId: null,
  streamingTaskState: null,

  getMessages: (sessionId: string) => {
    return get().messagesBySession[sessionId] || []
  },

  addUserMessage: (sessionId: string, content: string) => {
    const id = generateId()
    const msg: Message = { id, role: 'user', content, timestamp: Date.now() }
    set((s) => ({
      messagesBySession: {
        ...s.messagesBySession,
        [sessionId]: [...(s.messagesBySession[sessionId] || []), msg],
      },
    }))
    return id
  },

  startAssistantMessage: (sessionId: string) => {
    const id = generateId()
    const msg: Message = { id, role: 'assistant', content: '', orderedParts: [], toolCalls: [], isStreaming: true, timestamp: Date.now() }
    set((s) => ({
      messagesBySession: {
        ...s.messagesBySession,
        [sessionId]: [...(s.messagesBySession[sessionId] || []), msg],
      },
      streamingMessageId: id,
    }))
    return id
  },

  appendText: (messageId: string, content: string) => {
    set((s) => {
      const newBySession = { ...s.messagesBySession }
      for (const sid of Object.keys(newBySession)) {
        const msgs = newBySession[sid]
        const idx = msgs.findIndex((m) => m.id === messageId)
        if (idx !== -1) {
          newBySession[sid] = [...msgs]
          const parts = [...(msgs[idx].orderedParts || [])]
          const last = parts[parts.length - 1]
          if (last && last.type === 'text') {
            parts[parts.length - 1] = { type: 'text', content: last.content + content }
          } else {
            parts.push({ type: 'text', content })
          }
          newBySession[sid][idx] = {
            ...msgs[idx],
            content: msgs[idx].content + content,
            orderedParts: parts,
          }
          break
        }
      }
      return { messagesBySession: newBySession }
    })
  },

  addToolCall: (messageId: string, tool: ToolCallState) => {
    set((s) => {
      const newBySession = { ...s.messagesBySession }
      for (const sid of Object.keys(newBySession)) {
        const msgs = newBySession[sid]
        const idx = msgs.findIndex((m) => m.id === messageId)
        if (idx !== -1) {
          newBySession[sid] = [...msgs]
          const parts = [...(msgs[idx].orderedParts || []), { type: 'tool' as const, toolCallId: tool.id }]
          newBySession[sid][idx] = {
            ...msgs[idx],
            toolCalls: [...(msgs[idx].toolCalls || []), tool],
            orderedParts: parts,
          }
          break
        }
      }
      return { messagesBySession: newBySession }
    })
  },

  updateToolCall: (messageId: string, toolName: string, update: Partial<ToolCallState>) => {
    set((s) => {
      const newBySession = { ...s.messagesBySession }
      for (const sid of Object.keys(newBySession)) {
        const msgs = newBySession[sid]
        const idx = msgs.findIndex((m) => m.id === messageId)
        if (idx !== -1) {
          newBySession[sid] = [...msgs]
          const toolCalls = (msgs[idx].toolCalls || []).map((tc) =>
            tc.name === toolName && tc.status === 'running' ? { ...tc, ...update } : tc,
          )
          newBySession[sid][idx] = { ...msgs[idx], toolCalls }
          break
        }
      }
      return { messagesBySession: newBySession }
    })
  },

  finishMessage: (messageId: string) => {
    set((s) => {
      const newBySession = { ...s.messagesBySession }
      for (const sid of Object.keys(newBySession)) {
        const msgs = newBySession[sid]
        const idx = msgs.findIndex((m) => m.id === messageId)
        if (idx !== -1) {
          newBySession[sid] = [...msgs]
          newBySession[sid][idx] = { ...msgs[idx], isStreaming: false }
          break
        }
      }
      return { messagesBySession: newBySession, streamingMessageId: null, streamingTaskState: null }
    })
  },

  setStreaming: (streaming: boolean) => set({
    isStreaming: streaming,
    ...(streaming ? {} : { streamingTaskState: null }),
  }),

  setMessages: (sessionId: string, messages: Message[]) => {
    set((s) => ({
      messagesBySession: { ...s.messagesBySession, [sessionId]: messages },
    }))
  },

  updateTaskState: (messageId: string, state: TaskState) => {
    set((s) => {
      const newBySession = { ...s.messagesBySession }
      for (const sid of Object.keys(newBySession)) {
        const msgs = newBySession[sid]
        const idx = msgs.findIndex((m) => m.id === messageId)
        if (idx !== -1) {
          newBySession[sid] = [...msgs]
          newBySession[sid][idx] = { ...msgs[idx], taskState: state }
          break
        }
      }
      return { messagesBySession: newBySession, streamingTaskState: state }
    })
  },

  updateVerification: (messageId: string, report: VerificationReport) => {
    set((s) => {
      const newBySession = { ...s.messagesBySession }
      for (const sid of Object.keys(newBySession)) {
        const msgs = newBySession[sid]
        const idx = msgs.findIndex((m) => m.id === messageId)
        if (idx !== -1) {
          newBySession[sid] = [...msgs]
          newBySession[sid][idx] = { ...msgs[idx], verification: report }
          break
        }
      }
      return { messagesBySession: newBySession }
    })
  },

  clearSession: (sessionId: string) => {
    set((s) => {
      const newBySession = { ...s.messagesBySession }
      delete newBySession[sessionId]
      return { messagesBySession: newBySession }
    })
  },
}))
