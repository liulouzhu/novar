/** 单条消息气泡 */

import { type Message, type VerificationReport } from '@/lib/ws'
import { ToolCallCard } from './ToolCallCard'
import { MarkdownRenderer } from '../shared/MarkdownRenderer'
import { cn } from '@/lib/utils'
import { User, Bot, ShieldCheck, ShieldAlert } from 'lucide-react'

interface Props {
  message: Message
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user'
  const isSystem = message.role === 'system'

  if (isSystem) {
    return (
      <div className="text-center text-xs py-2" style={{ color: 'var(--text-tertiary)' }}>
        {message.content}
      </div>
    )
  }

  // 构建工具调用查找表
  const toolCallMap = new Map(
    (message.toolCalls || []).map((tc) => [tc.id, tc]),
  )

  /** 按时间顺序渲染内容（orderedParts 优先，兼容旧数据） */
  const renderContent = () => {
    const parts = message.orderedParts

    if (parts && parts.length > 0) {
      return parts.map((part, i) => {
        if (part.type === 'text') {
          if (!part.content) return null
          return (
            <div
              key={`text-${i}`}
              className={cn(
                'rounded-2xl px-4 py-3 text-sm leading-relaxed',
                isUser
                  ? 'bg-primary-500 text-white rounded-tr-md'
                  : 'rounded-tl-md',
              )}
              style={
                !isUser
                  ? { backgroundColor: 'var(--bg-bubble-agent)', border: '1px solid var(--border-color)' }
                  : undefined
              }
            >
              {isUser ? (
                <div className="whitespace-pre-wrap">{part.content}</div>
              ) : (
                <MarkdownRenderer content={part.content} />
              )}
            </div>
          )
        }
        // tool part
        const tc = toolCallMap.get(part.toolCallId)
        if (!tc) return null
        return <ToolCallCard key={tc.id} toolCall={tc} />
      })
    }

    // 兼容旧数据：工具调用在上，文本在下
    return (
      <>
        {message.toolCalls && message.toolCalls.length > 0 && (
          <div className="space-y-2 mb-3">
            {message.toolCalls.map((tc) => (
              <ToolCallCard key={tc.id} toolCall={tc} />
            ))}
          </div>
        )}
        {message.content && (
          <div
            className={cn(
              'rounded-2xl px-4 py-3 text-sm leading-relaxed',
              isUser
                ? 'bg-primary-500 text-white rounded-tr-md'
                : 'rounded-tl-md',
            )}
            style={
              !isUser
                ? { backgroundColor: 'var(--bg-bubble-agent)', border: '1px solid var(--border-color)' }
                : undefined
            }
          >
            {isUser ? (
              <div className="whitespace-pre-wrap">{message.content}</div>
            ) : (
              <MarkdownRenderer content={message.content} />
            )}
          </div>
        )}
      </>
    )
  }

  return (
    <div className={cn('flex gap-3', isUser ? 'flex-row-reverse' : 'flex-row')}>
      {/* 头像 */}
      <div
        className={cn(
          'w-8 h-8 rounded-full flex items-center justify-center shrink-0',
          isUser ? 'bg-primary-100 dark:bg-primary-900/40' : 'bg-gray-100 dark:bg-gray-800',
        )}
      >
        {isUser ? (
          <User size={16} className="text-primary-600 dark:text-primary-400" />
        ) : (
          <Bot size={16} style={{ color: 'var(--text-secondary)' }} />
        )}
      </div>

      {/* 消息内容 */}
      <div className={cn('min-w-0 max-w-[85%] space-y-2', isUser ? 'text-right' : '')}>
        {renderContent()}
        {!isUser && message.verification && (
          <VerificationPanel report={message.verification} />
        )}
      </div>
    </div>
  )
}

function VerificationPanel({ report }: { report: VerificationReport }) {
  const risky = report.assessments.filter((item) => item.verdict !== 'SUPPORTED')
  const isHigh = report.risk_level === 'high'
  const evidenceMap = new Map(report.evidence.map((item) => [item.evidence_id, item]))

  return (
    <details
      className="rounded-xl px-3 py-2 text-left text-xs"
      style={{ backgroundColor: 'var(--bg-bubble-agent)', border: '1px solid var(--border-color)' }}
    >
      <summary className="flex cursor-pointer list-none items-center gap-2 font-medium">
        {isHigh ? <ShieldAlert size={15} className="text-red-500" /> : <ShieldCheck size={15} className="text-emerald-500" />}
        <span>证据核验：{report.risk_level.toUpperCase()}</span>
        <span style={{ color: 'var(--text-tertiary)' }}>
          {report.did_revise ? '已修正回答' : '未修改回答'}
        </span>
      </summary>

      <div className="mt-3 space-y-3" style={{ color: 'var(--text-secondary)' }}>
        <div>
          共核验 {report.claims.length} 条事实，反向检索 {report.rag_queries} 次；
          整体风险 {(report.risk_score * 100).toFixed(0)}%。
        </div>
        {report.warnings.map((warning, index) => (
          <div key={index} className="text-amber-600 dark:text-amber-400">{warning}</div>
        ))}
        {risky.map((assessment) => {
          const claim = report.claims.find((item) => item.claim_id === assessment.claim_id)
          return (
            <div key={assessment.claim_id} className="space-y-1 border-t pt-2" style={{ borderColor: 'var(--border-color)' }}>
              <div className="font-medium" style={{ color: 'var(--text-primary)' }}>
                [{assessment.verdict}] {claim?.text || assessment.claim_id}
              </div>
              {assessment.reasoning && <div>{assessment.reasoning}</div>}
              {assessment.evidence_ids.map((id) => {
                const evidence = evidenceMap.get(id)
                if (!evidence) return null
                return (
                  <div key={id} className="rounded p-2" style={{ backgroundColor: 'var(--bg-secondary)' }}>
                    <div className="font-medium">{evidence.title || evidence.paper_id} · {evidence.section || evidence.chunk_id}</div>
                    <div className="mt-1 line-clamp-4">{evidence.text}</div>
                  </div>
                )
              })}
            </div>
          )
        })}
      </div>
    </details>
  )
}
