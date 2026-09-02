/** 左侧导航栏 */

import { useEffect, useState } from 'react'
import { useSessionStore } from '@/stores/sessionStore'
import { useThemeStore } from '@/stores/themeStore'
import { cn } from '@/lib/utils'
import { Plus, Trash2, MessageSquare, Loader2, Sun, Moon, X, FileText, Network, ChevronDown, ChevronRight, LogOut, Pencil, Brain } from 'lucide-react'
import { useAuthStore } from '@/stores/authStore'

export type PageKey = 'chat' | 'papers' | 'graph' | 'memory'

interface Props {
  activePage: PageKey
  onNavigate: (page: PageKey) => void
}

export function SessionSidebar({ activePage, onNavigate }: Props) {
  const { sessions, currentId, loading, loadSessions, createSession, switchSession, deleteSession, renameSession } = useSessionStore()
  const { resolved, setTheme } = useThemeStore()
  const { user, logout } = useAuthStore()
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null)
  const [sessionsExpanded, setSessionsExpanded] = useState(true)
  const [renamingId, setRenamingId] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')

  useEffect(() => {
    loadSessions()
  }, [loadSessions])

  const handleNew = async () => {
    const id = await createSession()
    onNavigate('chat')
    switchSession(id)
  }

  const handleDeleteClick = (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    setDeleteTarget(id)
  }

  const handleConfirmDelete = async () => {
    if (deleteTarget) {
      await deleteSession(deleteTarget)
      setDeleteTarget(null)
    }
  }

  const handleStartRename = (e: React.MouseEvent, id: string, currentTitle: string) => {
    e.stopPropagation()
    setRenamingId(id)
    setRenameValue(currentTitle)
  }

  const handleConfirmRename = async () => {
    if (renamingId && renameValue.trim()) {
      await renameSession(renamingId, renameValue)
    }
    setRenamingId(null)
    setRenameValue('')
  }

  const handleRenameKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleConfirmRename()
    } else if (e.key === 'Escape') {
      setRenamingId(null)
      setRenameValue('')
    }
  }

  const toggleTheme = () => {
    setTheme(resolved === 'dark' ? 'light' : 'dark')
  }

  const nonEmptySessions = sessions.filter((s) => s.message_count > 0)

  return (
    <div
      className="w-60 flex flex-col border-r shrink-0"
      style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
    >
      {/* 头部 */}
      <div className="p-3 border-b flex items-center justify-between" style={{ borderColor: 'var(--border-color)' }}>
        <h1 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          🔬 Novare
        </h1>
        <button
          onClick={toggleTheme}
          className="p-1.5 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
          title={resolved === 'dark' ? '切换浅色主题' : '切换深色主题'}
        >
          {resolved === 'dark' ? (
            <Sun size={15} style={{ color: 'var(--text-secondary)' }} />
          ) : (
            <Moon size={15} style={{ color: 'var(--text-secondary)' }} />
          )}
        </button>
      </div>

      {/* 导航按钮 */}
      <div className="p-2 space-y-0.5 border-b" style={{ borderColor: 'var(--border-color)' }}>
        {([
          { key: 'chat' as PageKey, icon: <MessageSquare size={15} />, label: '对话' },
          { key: 'papers' as PageKey, icon: <FileText size={15} />, label: '论文库' },
          { key: 'graph' as PageKey, icon: <Network size={15} />, label: '知识图谱' },
          { key: 'memory' as PageKey, icon: <Brain size={15} />, label: '记忆管理' },
        ]).map((item) => (
          <button
            key={item.key}
            onClick={() => onNavigate(item.key)}
            className={cn(
              'w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors text-left',
            )}
            style={{
              backgroundColor: activePage === item.key ? 'var(--accent-light)' : 'transparent',
              color: activePage === item.key ? 'var(--accent)' : 'var(--text-primary)',
              fontWeight: activePage === item.key ? 500 : 400,
            }}
          >
            {item.icon}
            {item.label}
          </button>
        ))}
      </div>

      {/* 会话列表（可折叠） */}
      <div className="flex-1 overflow-y-auto">
        {/* 折叠头 */}
        <button
          onClick={() => setSessionsExpanded(!sessionsExpanded)}
          className="w-full flex items-center justify-between px-3 py-2.5 text-xs font-medium hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          style={{ color: 'var(--text-tertiary)' }}
        >
          <span>最近对话 ({nonEmptySessions.length})</span>
          <div className="flex items-center gap-1.5">
            <span
              onClick={(e) => { e.stopPropagation(); handleNew() }}
              className="p-0.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
              title="新建会话"
            >
              <Plus size={13} />
            </span>
            {sessionsExpanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
          </div>
        </button>

        {/* 会话列表内容 */}
        {sessionsExpanded && (
          <div className="px-2 pb-2 space-y-0.5">
            {loading && (
              <div className="flex items-center justify-center py-4">
                <Loader2 size={14} className="animate-spin" style={{ color: 'var(--text-tertiary)' }} />
              </div>
            )}
            {nonEmptySessions.map((s) => (
              <div
                key={s.session_id}
                onClick={() => { if (renamingId !== s.session_id) { switchSession(s.session_id); onNavigate('chat') } }}
                className={cn(
                  'group flex items-center gap-2 px-3 py-1.5 rounded-lg cursor-pointer transition-colors text-sm',
                  currentId === s.session_id && activePage === 'chat'
                    ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                    : 'hover:bg-gray-100 dark:hover:bg-gray-800',
                )}
                style={!(currentId === s.session_id && activePage === 'chat') ? { color: 'var(--text-primary)' } : undefined}
              >
                {renamingId === s.session_id ? (
                  <input
                    autoFocus
                    value={renameValue}
                    onChange={(e) => setRenameValue(e.target.value)}
                    onBlur={handleConfirmRename}
                    onKeyDown={handleRenameKeyDown}
                    onClick={(e) => e.stopPropagation()}
                    className="flex-1 min-w-0 text-xs bg-transparent outline-none border rounded px-1.5 py-0.5"
                    style={{ borderColor: 'var(--accent)', color: 'var(--text-primary)' }}
                    maxLength={60}
                  />
                ) : (
                  <span className="truncate flex-1 text-xs">{s.title || s.session_id}</span>
                )}
                <button
                  onClick={(e) => handleStartRename(e, s.session_id, s.title)}
                  className="opacity-0 group-hover:opacity-100 p-0.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-opacity"
                  title="重命名"
                >
                  <Pencil size={11} style={{ color: 'var(--text-secondary)' }} />
                </button>
                <button
                  onClick={(e) => handleDeleteClick(e, s.session_id)}
                  className="opacity-0 group-hover:opacity-100 p-0.5 rounded hover:bg-red-100 dark:hover:bg-red-900/30 transition-opacity"
                >
                  <Trash2 size={11} style={{ color: 'var(--error)' }} />
                </button>
              </div>
            ))}

            {!loading && nonEmptySessions.length === 0 && (
              <div className="text-center py-6 text-xs" style={{ color: 'var(--text-tertiary)' }}>
                暂无对话
              </div>
            )}
          </div>
        )}
      </div>

      {/* 用户信息与登出 */}
      <div className="p-3 border-t flex items-center gap-2" style={{ borderColor: 'var(--border-color)' }}>
        <div
          className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-medium shrink-0"
          style={{ backgroundColor: 'var(--accent-light)', color: 'var(--accent)' }}
        >
          {user?.username?.[0]?.toUpperCase() || '?'}
        </div>
        <span className="flex-1 truncate text-xs" style={{ color: 'var(--text-primary)' }}>
          {user?.username || '未登录'}
        </span>
        <button
          onClick={logout}
          className="p-1.5 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
          title="退出登录"
        >
          <LogOut size={14} style={{ color: 'var(--text-secondary)' }} />
        </button>
      </div>

      {/* 删除确认弹窗 */}
      {deleteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ backgroundColor: 'rgba(0,0,0,0.4)' }} onClick={() => setDeleteTarget(null)}>
          <div
            className="w-72 rounded-xl border shadow-xl p-5"
            style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-primary)' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>删除会话</div>
              <button onClick={() => setDeleteTarget(null)} className="p-1 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
                <X size={14} style={{ color: 'var(--text-tertiary)' }} />
              </button>
            </div>
            <p className="text-sm mb-5" style={{ color: 'var(--text-secondary)' }}>确定要删除此会话吗？删除后不可恢复。</p>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setDeleteTarget(null)}
                className="px-3 py-1.5 text-sm rounded-lg border transition-colors hover:bg-gray-50 dark:hover:bg-gray-800"
                style={{ borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}
              >取消</button>
              <button
                onClick={handleConfirmDelete}
                className="px-3 py-1.5 text-sm rounded-lg text-white transition-colors hover:opacity-90"
                style={{ backgroundColor: 'var(--error)' }}
              >删除</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
