/** 工具调用折叠面板 */

import { useState } from 'react'
import { type ToolCallState } from '@/lib/ws'
import { formatDuration } from '@/lib/utils'
import { ChevronDown, ChevronRight, CheckCircle, XCircle, Loader2, Wrench } from 'lucide-react'

interface Props {
  toolCall: ToolCallState
}

const TOOL_LABELS: Record<string, string> = {
  paper_search: '📄 论文搜索',
  paper_parse: '📖 论文解析',
  rag_query: '🔍 RAG 查询',
  knowledge_graph: '🕸️ 知识图谱',
  code_execute: '💻 代码执行',
  read_file: '📂 读取文件',
  write_file: '📝 写入文件',
  edit_file: '✏️ 编辑文件',
  glob_search: '🔎 文件搜索',
  grep_search: '🔎 内容搜索',
}

export function ToolCallCard({ toolCall }: Props) {
  const [expanded, setExpanded] = useState(false)

  const label = TOOL_LABELS[toolCall.name] || `🔧 ${toolCall.name}`

  // 摘要行：优先使用结构化 summary，降级到旧的 result 截断
  const headline = toolCall.summary
    || (toolCall.error ? toolCall.error.slice(0, 100) : '')
    || (toolCall.result ? toolCall.result.slice(0, 100) : '')

  // 状态判定：优先使用 ok 字段，降级到 status
  const isOk = toolCall.ok !== undefined ? toolCall.ok : toolCall.status === 'success'

  const statusIcon = {
    running: <Loader2 size={14} className="animate-spin" style={{ color: 'var(--accent)' }} />,
    success: <CheckCircle size={14} style={{ color: isOk ? 'var(--success)' : 'var(--error)' }} />,
    error: <XCircle size={14} style={{ color: 'var(--error)' }} />,
  }[toolCall.status]

  const borderColor = {
    running: 'var(--accent)',
    success: isOk ? 'var(--success)' : 'var(--error)',
    error: 'var(--error)',
  }[toolCall.status]

  // 展开面板内容：优先使用 dataPreview（结构化），降级到 result/error
  const detailContent = toolCall.dataPreview
    ? JSON.stringify(toolCall.dataPreview, null, 2)
    : toolCall.error || toolCall.result || ''

  return (
    <div
      className="rounded-lg border transition-all tool-panel-enter"
      style={{ borderColor, backgroundColor: 'var(--bg-tool-panel)' }}
    >
      {/* 折叠头部 */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-gray-50 dark:hover:bg-gray-800/50 rounded-lg transition-colors"
      >
        {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        {statusIcon}
        <span className="font-medium" style={{ color: 'var(--text-primary)' }}>{label}</span>
        {headline && (
          <span className="text-xs truncate ml-1" style={{ color: 'var(--text-tertiary)' }}>
            {headline}
          </span>
        )}
        <span className="text-xs ml-auto shrink-0" style={{ color: 'var(--text-tertiary)' }}>
          {toolCall.status === 'running' && '执行中...'}
          {toolCall.status === 'success' && toolCall.duration !== undefined && formatDuration(toolCall.duration)}
          {toolCall.status === 'error' && '失败'}
        </span>
      </button>

      {/* 展开内容 */}
      {expanded && (
        <div className="px-3 pb-3 space-y-3 border-t" style={{ borderColor: 'var(--border-color)' }}>
          {/* warnings */}
          {toolCall.warnings && toolCall.warnings.length > 0 && (
            <div className="mt-3">
              <div className="text-xs font-medium mb-1.5" style={{ color: 'var(--text-tertiary)' }}>⚠️ 警告</div>
              <ul className="text-xs space-y-0.5">
                {toolCall.warnings.map((w, i) => (
                  <li key={i} style={{ color: 'var(--error)' }}>• {w}</li>
                ))}
              </ul>
            </div>
          )}

          {/* 参数 */}
          <div className="mt-3">
            <div className="text-xs font-medium mb-1.5 flex items-center gap-1" style={{ color: 'var(--text-tertiary)' }}>
              📥 参数
            </div>
            <pre
              className="text-xs p-2.5 rounded-md overflow-x-auto"
              style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
            >
              {JSON.stringify(toolCall.params, null, 2)}
            </pre>
          </div>

          {/* 返回结果 / 结构化数据 */}
          {detailContent && (
            <div>
              <div className="text-xs font-medium mb-1.5 flex items-center gap-1" style={{ color: 'var(--text-tertiary)' }}>
                {toolCall.status === 'error' || !isOk ? '❌ 错误' : '📤 返回结果'}
              </div>
              <pre
                className="text-xs p-2.5 rounded-md overflow-x-auto max-h-60 overflow-y-auto whitespace-pre-wrap"
                style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
              >
                {detailContent}
              </pre>
            </div>
          )}

          {/* sources */}
          {toolCall.sources && toolCall.sources.length > 0 && (
            <div>
              <div className="text-xs font-medium mb-1.5" style={{ color: 'var(--text-tertiary)' }}>📚 引用来源</div>
              <ul className="text-xs space-y-0.5">
                {toolCall.sources.map((s, i) => (
                  <li key={i} style={{ color: 'var(--text-secondary)' }}>
                    • {String(s.title || s.id || JSON.stringify(s))}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* 状态行 */}
          <div className="flex items-center gap-3 text-xs" style={{ color: 'var(--text-tertiary)' }}>
            <span>
              状态：
              {toolCall.status === 'success' && isOk && '✅ 成功'}
              {toolCall.status === 'success' && !isOk && '❌ 失败'}
              {toolCall.status === 'error' && '❌ 失败'}
              {toolCall.status === 'running' && '⏳ 执行中'}
            </span>
            {toolCall.duration !== undefined && (
              <span>⏱ {formatDuration(toolCall.duration)}</span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
