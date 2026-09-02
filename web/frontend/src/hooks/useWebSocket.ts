/** WebSocket 连接管理 Hook */

import { useCallback, useEffect, useRef, useState } from 'react'
import { type ServerEvent } from '@/lib/ws'
import { useAuthStore } from '@/stores/authStore'

interface UseWebSocketOptions {
  sessionId: string
  onEvent: (event: ServerEvent) => void
}

export function useWebSocket({ sessionId, onEvent }: UseWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null)
  const [connected, setConnected] = useState(false)
  const onEventRef = useRef(onEvent)
  onEventRef.current = onEvent

  // 建立连接
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const token = useAuthStore.getState().token
    const url = `${protocol}//${host}/ws/chat/${sessionId}?token=${encodeURIComponent(token || '')}`

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
    ws.onerror = () => setConnected(false)

    ws.onmessage = (e) => {
      try {
        const event = JSON.parse(e.data) as ServerEvent
        onEventRef.current(event)
      } catch {
        console.error('Failed to parse WebSocket message:', e.data)
      }
    }

    return () => {
      ws.close()
    }
  }, [sessionId])

  const sendMessage = useCallback((data: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  return { connected, sendMessage }
}
