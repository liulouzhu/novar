/** 对话主区域 */

import { useEffect, useRef } from 'react'
import { useChat } from '@/hooks/useChat'
import { useChatStore } from '@/stores/chatStore'
import { MessageBubble } from './MessageBubble'
import { InputBox } from './InputBox'
import { TaskStatePanel } from './TaskStatePanel'
import { Wifi, WifiOff, Loader2, PanelRightClose, PanelRightOpen } from 'lucide-react'

interface Props {
  sessionId: string
  panelOpen: boolean
  onTogglePanel: () => void
}

export function ChatArea({ sessionId, panelOpen, onTogglePanel }: Props) {
  const { messages, isStreaming, connected, send, stop } = useChat(sessionId)
  const streamingTaskState = useChatStore((s) => s.streamingTaskState)
  const scrollRef = useRef<HTMLDivElement>(null)
  const autoScrollRef = useRef(true)

  // 自动滚动到底部
  useEffect(() => {
    if (autoScrollRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, streamingTaskState])

  // 检测用户是否手动滚动
  const handleScroll = () => {
    if (!scrollRef.current) return
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current
    autoScrollRef.current = scrollHeight - scrollTop - clientHeight < 100
  }

  return (
    <div className="flex flex-col h-full">
      {/* 顶栏 */}
      <div
        className="flex items-center justify-between px-4 py-2 border-b shrink-0"
        style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-primary)' }}
      >
        <div className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
          对话
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs" style={{ color: connected ? 'var(--success)' : 'var(--error)' }}>
            {connected ? <Wifi size={12} /> : <WifiOff size={12} />}
            <span>{connected ? '已连接' : '未连接'}</span>
          </div>
          <button
            onClick={onTogglePanel}
            className="p-1.5 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            title={panelOpen ? '隐藏论文库' : '显示论文库'}
          >
            {panelOpen ? (
              <PanelRightClose size={16} style={{ color: 'var(--text-secondary)' }} />
            ) : (
              <PanelRightOpen size={16} style={{ color: 'var(--text-secondary)' }} />
            )}
          </button>
        </div>
      </div>

      {/* 消息列表 */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-4 py-6"
        style={{ backgroundColor: 'var(--bg-primary)' }}
      >
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}

          {isStreaming && messages.length > 0 && messages[messages.length - 1]?.role === 'assistant' && (
            <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-tertiary)' }}>
              <Loader2 size={12} className="animate-spin" />
              <span>Agent 正在工作中...</span>
            </div>
          )}
        </div>
      </div>

      {/* 任务状态面板（可收缩，仅 streaming 时显示） */}
      {isStreaming && streamingTaskState && (
        <div className="shrink-0 max-w-3xl mx-auto w-full px-4 pb-1">
          <TaskStatePanel taskState={streamingTaskState} />
        </div>
      )}

      {/* 输入框 */}
      <InputBox onSend={send} onStop={stop} disabled={isStreaming} />
    </div>
  )
}
