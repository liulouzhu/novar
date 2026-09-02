/** 记忆管理页面 */

import { useState, useEffect, useCallback } from 'react'
import {
  type MemoryItem,
  fetchMemories,
  updateMemory,
  deleteMemory,
  clearMemories,
  togglePin,
} from '@/lib/api'
import {
  Brain,
  Pin,
  PinOff,
  Trash2,
  Pencil,
  X,
  Check,
  Loader2,
  AlertTriangle,
  RefreshCw,
} from 'lucide-react'

const CATEGORY_LABELS: Record<string, string> = {
  research_preference: '研究偏好',
  interaction_preference: '交互偏好',
}

const CONFIDENCE_STYLE = (c: number) => {
  if (c >= 0.8) return { color: '#22c55e', label: '高' }
  if (c >= 0.5) return { color: '#eab308', label: '中' }
  return { color: '#ef4444', label: '低' }
}

export function MemoryPage() {
  const [memories, setMemories] = useState<MemoryItem[]>([])
  const [loading, setLoading] = useState(false)
  const [editingId, setEditingId] = useState<number | null>(null)
  const [editValue, setEditValue] = useState('')
  const [editConfidence, setEditConfidence] = useState(1.0)
  const [clearConfirm, setClearConfirm] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<number | null>(null)
  const [actionLoading, setActionLoading] = useState<number | null>(null)

  const loadMemories = useCallback(async () => {
    setLoading(true)
    try {
      const data = await fetchMemories()
      setMemories(data)
    } catch (e) {
      console.error('Failed to load memories:', e)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadMemories()
  }, [loadMemories])

  // ── 编辑 ──

  const startEdit = (m: MemoryItem) => {
    setEditingId(m.id)
    setEditValue(m.value)
    setEditConfidence(m.confidence)
  }

  const cancelEdit = () => {
    setEditingId(null)
    setEditValue('')
  }

  const saveEdit = async () => {
    if (editingId === null) return
    setActionLoading(editingId)
    try {
      const updated = await updateMemory(editingId, {
        value: editValue,
        confidence: editConfidence,
      })
      setMemories((prev) => prev.map((m) => (m.id === editingId ? updated : m)))
      cancelEdit()
    } catch (e) {
      console.error('Failed to update memory:', e)
    } finally {
      setActionLoading(null)
    }
  }

  // ── Pin ──

  const handleTogglePin = async (id: number) => {
    setActionLoading(id)
    try {
      const updated = await togglePin(id)
      setMemories((prev) => prev.map((m) => (m.id === id ? updated : m)))
    } catch (e) {
      console.error('Failed to toggle pin:', e)
    } finally {
      setActionLoading(null)
    }
  }

  // ── 删除 ──

  const confirmDelete = async () => {
    if (deleteTarget === null) return
    setActionLoading(deleteTarget)
    try {
      await deleteMemory(deleteTarget)
      setMemories((prev) => prev.filter((m) => m.id !== deleteTarget))
      setDeleteTarget(null)
    } catch (e) {
      console.error('Failed to delete memory:', e)
    } finally {
      setActionLoading(null)
    }
  }

  // ── 清空 ──

  const handleClear = async () => {
    setLoading(true)
    try {
      await clearMemories()
      setMemories([])
      setClearConfirm(false)
    } catch (e) {
      console.error('Failed to clear memories:', e)
    } finally {
      setLoading(false)
    }
  }

  // ── 按 category 分组 ──

  const grouped = memories.reduce<Record<string, MemoryItem[]>>((acc, m) => {
    ;(acc[m.category] ??= []).push(m)
    return acc
  }, {})

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* 顶栏 */}
      <div
        className="flex items-center justify-between px-6 py-4 border-b shrink-0"
        style={{ borderColor: 'var(--border-color)' }}
      >
        <div className="flex items-center gap-2">
          <Brain size={20} style={{ color: 'var(--accent)' }} />
          <h1 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
            长期记忆
          </h1>
          <span className="text-xs ml-2" style={{ color: 'var(--text-tertiary)' }}>
            {memories.length} 条
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={loadMemories}
            disabled={loading}
            className="p-1.5 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
            title="刷新"
          >
            <RefreshCw size={15} style={{ color: 'var(--text-secondary)' }} />
          </button>
          {memories.length > 0 && (
            <button
              onClick={() => setClearConfirm(true)}
              className="px-3 py-1.5 text-xs rounded-md transition-colors"
              style={{
                color: '#ef4444',
                border: '1px solid rgba(239,68,68,0.3)',
                backgroundColor: 'transparent',
              }}
            >
              清空全部
            </button>
          )}
        </div>
      </div>

      {/* 清空确认 */}
      {clearConfirm && (
        <div
          className="mx-6 mt-4 p-3 rounded-lg border flex items-center gap-3"
          style={{ borderColor: 'rgba(239,68,68,0.4)', backgroundColor: 'rgba(239,68,68,0.05)' }}
        >
          <AlertTriangle size={16} style={{ color: '#ef4444', flexShrink: 0 }} />
          <span className="flex-1 text-sm" style={{ color: 'var(--text-primary)' }}>
            确定清空全部记忆？此操作不可撤销。
          </span>
          <button
            onClick={handleClear}
            className="px-3 py-1 text-xs rounded-md text-white"
            style={{ backgroundColor: '#ef4444' }}
          >
            确认清空
          </button>
          <button
            onClick={() => setClearConfirm(false)}
            className="px-3 py-1 text-xs rounded-md"
            style={{ color: 'var(--text-secondary)' }}
          >
            取消
          </button>
        </div>
      )}

      {/* 主内容 */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6">
        {loading && memories.length === 0 && (
          <div className="flex items-center justify-center py-20">
            <Loader2 size={24} className="animate-spin" style={{ color: 'var(--text-tertiary)' }} />
          </div>
        )}

        {!loading && memories.length === 0 && (
          <div className="text-center py-20" style={{ color: 'var(--text-tertiary)' }}>
            <Brain size={40} className="mx-auto mb-3 opacity-30" />
            <p className="text-sm">暂无长期记忆</p>
            <p className="text-xs mt-1">对话过程中助手会自动提取记忆</p>
          </div>
        )}

        {Object.entries(grouped).map(([category, items]) => (
          <div key={category}>
            <h2
              className="text-xs font-medium uppercase tracking-wider mb-3"
              style={{ color: 'var(--text-tertiary)' }}
            >
              {CATEGORY_LABELS[category] || category}
            </h2>
            <div className="space-y-2">
              {items.map((m) => (
                <div
                  key={m.id}
                  className="rounded-lg border p-3 transition-colors"
                  style={{
                    borderColor: 'var(--border-color)',
                    backgroundColor: 'var(--bg-secondary)',
                  }}
                >
                  {editingId === m.id ? (
                    /* ── 编辑模式 ── */
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <span
                          className="text-xs font-medium px-2 py-0.5 rounded"
                          style={{
                            backgroundColor: 'var(--accent-light)',
                            color: 'var(--accent)',
                          }}
                        >
                          {m.key}
                        </span>
                      </div>
                      <textarea
                        value={editValue}
                        onChange={(e) => setEditValue(e.target.value)}
                        rows={2}
                        className="w-full rounded-md border px-3 py-2 text-sm resize-none focus:outline-none focus:ring-1"
                        style={{
                          borderColor: 'var(--border-color)',
                          backgroundColor: 'var(--bg-primary)',
                          color: 'var(--text-primary)',
                        }}
                      />
                      <div className="flex items-center gap-3">
                        <label className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
                          置信度
                          <input
                            type="range"
                            min={0}
                            max={1}
                            step={0.1}
                            value={editConfidence}
                            onChange={(e) => setEditConfidence(parseFloat(e.target.value))}
                            className="w-24"
                          />
                          <span style={{ color: CONFIDENCE_STYLE(editConfidence).color, fontWeight: 500 }}>
                            {editConfidence.toFixed(1)}
                          </span>
                        </label>
                        <div className="flex-1" />
                        <button
                          onClick={saveEdit}
                          disabled={actionLoading === m.id}
                          className="flex items-center gap-1 px-2.5 py-1 text-xs rounded-md text-white"
                          style={{ backgroundColor: 'var(--accent)' }}
                        >
                          <Check size={12} /> 保存
                        </button>
                        <button
                          onClick={cancelEdit}
                          className="flex items-center gap-1 px-2.5 py-1 text-xs rounded-md"
                          style={{ color: 'var(--text-secondary)' }}
                        >
                          <X size={12} /> 取消
                        </button>
                      </div>
                    </div>
                  ) : (
                    /* ── 展示模式 ── */
                    <div className="flex items-start gap-3">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span
                            className="text-xs font-medium px-2 py-0.5 rounded"
                            style={{
                              backgroundColor: 'var(--accent-light)',
                              color: 'var(--accent)',
                            }}
                          >
                            {m.key}
                          </span>
                          {m.pinned && (
                            <span
                              className="flex items-center gap-0.5 text-xs px-1.5 py-0.5 rounded"
                              style={{ backgroundColor: 'rgba(59,130,246,0.1)', color: '#3b82f6' }}
                            >
                              <Pin size={10} /> 已固定
                            </span>
                          )}
                          {m.source === 'auto' && (
                            <span
                              className="text-xs px-1.5 py-0.5 rounded"
                              style={{ backgroundColor: 'var(--bg-primary)', color: 'var(--text-tertiary)' }}
                            >
                              自动提取
                            </span>
                          )}
                        </div>
                        <p className="text-sm leading-relaxed" style={{ color: 'var(--text-primary)' }}>
                          {m.value}
                        </p>
                        <div className="flex items-center gap-3 mt-1.5">
                          <span
                            className="text-xs"
                            style={{ color: CONFIDENCE_STYLE(m.confidence).color }}
                          >
                            置信度 {m.confidence.toFixed(1)} · {CONFIDENCE_STYLE(m.confidence).label}
                          </span>
                        </div>
                      </div>
                      {/* 操作按钮 */}
                      <div className="flex items-center gap-1 shrink-0">
                        <button
                          onClick={() => startEdit(m)}
                          className="p-1.5 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
                          title="编辑"
                        >
                          <Pencil size={13} style={{ color: 'var(--text-secondary)' }} />
                        </button>
                        <button
                          onClick={() => handleTogglePin(m.id)}
                          disabled={actionLoading === m.id}
                          className="p-1.5 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
                          title={m.pinned ? '取消固定' : '固定'}
                        >
                          {m.pinned ? (
                            <PinOff size={13} style={{ color: '#3b82f6' }} />
                          ) : (
                            <Pin size={13} style={{ color: 'var(--text-secondary)' }} />
                          )}
                        </button>
                        <button
                          onClick={() => setDeleteTarget(m.id)}
                          className="p-1.5 rounded-md hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
                          title="删除"
                        >
                          <Trash2 size={13} style={{ color: '#ef4444' }} />
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* 删除确认弹窗 */}
      {deleteTarget !== null && (
        <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ backgroundColor: 'rgba(0,0,0,0.4)' }}>
          <div
            className="rounded-xl border p-5 w-80 space-y-3"
            style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-primary)' }}
          >
            <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
              删除这条记忆？
            </p>
            <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
              {memories.find((m) => m.id === deleteTarget)?.value}
            </p>
            <div className="flex justify-end gap-2 pt-2">
              <button
                onClick={() => setDeleteTarget(null)}
                className="px-3 py-1.5 text-xs rounded-md"
                style={{ color: 'var(--text-secondary)' }}
              >
                取消
              </button>
              <button
                onClick={confirmDelete}
                disabled={actionLoading === deleteTarget}
                className="px-3 py-1.5 text-xs rounded-md text-white"
                style={{ backgroundColor: '#ef4444' }}
              >
                删除
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
