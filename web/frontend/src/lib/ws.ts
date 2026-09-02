/** WebSocket 事件类型定义 */

// 任务状态结构
export interface TaskState {
  goal: string
  completed: string[]
  pending: string[]
  tools_used: string[]
  key_findings: string[]
  missing_info: string[]
}

export interface VerificationClaim {
  claim_id: string
  text: string
  importance: 'high' | 'medium' | 'low'
  claim_type: string
}

export interface VerificationEvidence {
  evidence_id: string
  claim_id: string
  paper_id: string
  chunk_id: string
  title: string
  section: string
  text: string
  score: number | null
}

export interface VerificationAssessment {
  claim_id: string
  verdict: 'SUPPORTED' | 'CONTRADICTED' | 'NOT_ENOUGH_EVIDENCE'
  evidence_ids: string[]
  reasoning: string
  risk: number
}

export interface VerificationReport {
  status: string
  risk_score: number
  max_claim_risk: number
  risk_level: 'low' | 'medium' | 'high'
  did_revise: boolean
  rag_queries: number
  llm_calls: number
  warnings: string[]
  claims: VerificationClaim[]
  evidence: VerificationEvidence[]
  assessments: VerificationAssessment[]
}

// 服务端 → 客户端
export type ServerEvent =
  | { type: 'text_delta'; content: string }
  | { type: 'reasoning_delta'; content: string }
  | { type: 'tool_start'; tool: string; params: Record<string, unknown> }
  | { type: 'tool_end'; tool: string; ok: boolean; summary: string; result: string; data_preview: unknown; warnings: string[]; sources: Array<Record<string, unknown>>; duration: number }
  | { type: 'tool_error'; tool: string; ok: boolean; summary: string; error: string; data_preview: unknown; warnings: string[] }
  | { type: 'task_state'; goal: string; completed: string[]; pending: string[]; tools_used: string[]; key_findings: string[]; missing_info: string[] }
  | ({ type: 'verification' } & VerificationReport)
  | { type: 'cancelled'; message: string }
  | { type: 'done'; usage?: Record<string, number> }
  | { type: 'error'; message: string }

// 客户端 → 服务端
export type ClientMessage =
  | { type: 'send'; content: string }
  | { type: 'send_with_refs'; content: string; references: Array<{ type: string; id: string; title?: string }> }
  | { type: 'stop' }

// 工具调用状态
export interface ToolCallState {
  id: string
  name: string
  params: Record<string, unknown>
  status: 'running' | 'success' | 'error'
  ok?: boolean
  summary?: string
  result?: string
  dataPreview?: unknown
  warnings?: string[]
  sources?: Array<Record<string, unknown>>
  duration?: number
  error?: string
}

// 消息内容片段（按时间顺序交织文本和工具调用）
export type MessagePart =
  | { type: 'text'; content: string }
  | { type: 'tool'; toolCallId: string }

// 消息
export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  orderedParts?: MessagePart[]
  toolCalls?: ToolCallState[]
  taskState?: TaskState
  verification?: VerificationReport
  isStreaming?: boolean
  timestamp: number
}
