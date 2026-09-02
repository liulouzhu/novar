/** 任务状态折叠面板 — 显示当前 turn 的 TaskState */

import { useState } from 'react'
import { type TaskState } from '@/lib/ws'
import { ChevronDown, ChevronRight, Target, CheckCircle2, Clock, Wrench, Lightbulb, HelpCircle } from 'lucide-react'

interface Props {
  taskState: TaskState
}

export function TaskStatePanel({ taskState }: Props) {
  const [expanded, setExpanded] = useState(false)

  const completedCount = taskState.completed.length
  const pendingCount = taskState.pending.length

  return (
    <div
      className="rounded-lg border mb-3 transition-all"
      style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-tool-panel)' }}
    >
      {/* 折叠头部 — 默认折叠，只显示摘要 */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-gray-50 dark:hover:bg-gray-800/50 rounded-lg transition-colors"
      >
        {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <Target size={14} style={{ color: 'var(--accent)' }} />
        <span className="font-medium truncate" style={{ color: 'var(--text-primary)' }}>
          {taskState.goal}
        </span>
        <span className="text-xs ml-auto shrink-0" style={{ color: 'var(--text-tertiary)' }}>
          ✅ {completedCount} · ⏳ {pendingCount}
        </span>
      </button>

      {/* 展开内容 */}
      {expanded && (
        <div className="px-3 pb-3 space-y-3 border-t" style={{ borderColor: 'var(--border-color)' }}>
          {/* 目标 */}
          <SectionRow icon={<Target size={13} />} label="目标">
            <span className="text-sm" style={{ color: 'var(--text-primary)' }}>{taskState.goal}</span>
          </SectionRow>

          {/* 已完成 */}
          {taskState.completed.length > 0 && (
            <SectionRow icon={<CheckCircle2 size={13} />} label="已完成">
              <ul className="space-y-1">
                {taskState.completed.map((step, i) => (
                  <li key={i} className="text-sm flex items-start gap-1.5" style={{ color: 'var(--text-primary)' }}>
                    <span className="shrink-0 mt-0.5" style={{ color: 'var(--success)' }}>✓</span>
                    <span>{step}</span>
                  </li>
                ))}
              </ul>
            </SectionRow>
          )}

          {/* 待办 */}
          {taskState.pending.length > 0 && (
            <SectionRow icon={<Clock size={13} />} label="待办">
              <ul className="space-y-1">
                {taskState.pending.map((step, i) => (
                  <li key={i} className="text-sm flex items-start gap-1.5" style={{ color: 'var(--text-secondary)' }}>
                    <span className="shrink-0 mt-0.5">○</span>
                    <span>{step}</span>
                  </li>
                ))}
              </ul>
            </SectionRow>
          )}

          {/* 已用工具 */}
          {taskState.tools_used.length > 0 && (
            <SectionRow icon={<Wrench size={13} />} label="工具">
              <div className="flex flex-wrap gap-1.5">
                {taskState.tools_used.map((tool, i) => (
                  <span
                    key={i}
                    className="text-xs px-2 py-0.5 rounded-full"
                    style={{
                      backgroundColor: 'var(--bg-tertiary)',
                      color: 'var(--text-secondary)',
                      border: '1px solid var(--border-color)',
                    }}
                  >
                    {tool}
                  </span>
                ))}
              </div>
            </SectionRow>
          )}

          {/* 关键发现 */}
          {taskState.key_findings.length > 0 && (
            <SectionRow icon={<Lightbulb size={13} />} label="发现">
              <ul className="space-y-1">
                {taskState.key_findings.map((finding, i) => (
                  <li key={i} className="text-sm" style={{ color: 'var(--text-primary)' }}>
                    • {finding}
                  </li>
                ))}
              </ul>
            </SectionRow>
          )}

          {/* 缺失信息 */}
          {taskState.missing_info.length > 0 && (
            <SectionRow icon={<HelpCircle size={13} />} label="缺失">
              <ul className="space-y-1">
                {taskState.missing_info.map((info, i) => (
                  <li key={i} className="text-sm" style={{ color: 'var(--error)' }}>
                    ⚠ {info}
                  </li>
                ))}
              </ul>
            </SectionRow>
          )}
        </div>
      )}
    </div>
  )
}

/** 通用行布局：icon + label 在左，children 在右/下方 */
function SectionRow({ icon, label, children }: { icon: React.ReactNode; label: string; children: React.ReactNode }) {
  return (
    <div className="mt-3">
      <div className="text-xs font-medium mb-1.5 flex items-center gap-1.5" style={{ color: 'var(--text-tertiary)' }}>
        {icon}
        {label}
      </div>
      {children}
    </div>
  )
}
