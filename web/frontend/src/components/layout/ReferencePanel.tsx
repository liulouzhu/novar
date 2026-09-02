/** 右侧引用面板 */

import { useState, useEffect } from 'react'
import { type Paper, fetchPapers } from '@/lib/api'
import { FileText, ExternalLink, Search, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'

interface Props {
  sessionId: string
}

export function ReferencePanel({ sessionId }: Props) {
  const [papers, setPapers] = useState<Paper[]>([])
  const [loading, setLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [expandedPaper, setExpandedPaper] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    fetchPapers()
      .then(setPapers)
      .catch((e) => console.error('Failed to load papers:', e))
      .finally(() => setLoading(false))
  }, [sessionId])

  const filteredPapers = searchQuery
    ? papers.filter((p) => p.title.toLowerCase().includes(searchQuery.toLowerCase()))
    : papers

  const parsedPapers = filteredPapers.filter((p) => p.is_parsed)
  const metaPapers = filteredPapers.filter((p) => !p.is_parsed)

  return (
    <div
      className="w-80 flex flex-col border-l shrink-0 overflow-hidden"
      style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
    >
      {/* 头部 */}
      <div className="p-3 border-b" style={{ borderColor: 'var(--border-color)' }}>
        <h2 className="text-sm font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
          📚 论文库
        </h2>
        <div className="relative">
          <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-tertiary)' }} />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="搜索论文..."
            className="w-full pl-8 pr-3 py-1.5 text-sm rounded-md border outline-none focus:ring-1"
            style={{
              borderColor: 'var(--border-color)',
              backgroundColor: 'var(--bg-primary)',
              color: 'var(--text-primary)',
              '--tw-ring-color': 'var(--border-focus)',
            } as React.CSSProperties}
          />
        </div>
      </div>

      {/* 论文列表 */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {loading && (
          <div className="flex items-center justify-center py-8">
            <Loader2 size={16} className="animate-spin" style={{ color: 'var(--text-tertiary)' }} />
          </div>
        )}

        {/* 已解析论文 */}
        {parsedPapers.length > 0 && (
          <div>
            <div className="text-xs font-medium mb-2 flex items-center gap-1" style={{ color: 'var(--text-tertiary)' }}>
              ✅ 已解析 ({parsedPapers.length})
            </div>
            <div className="space-y-1.5">
              {parsedPapers.map((p) => (
                <PaperCard
                  key={p.id}
                  paper={p}
                  expanded={expandedPaper === p.id}
                  onToggle={() => setExpandedPaper(expandedPaper === p.id ? null : p.id)}
                />
              ))}
            </div>
          </div>
        )}

        {/* 仅元数据论文 */}
        {metaPapers.length > 0 && (
          <div>
            <div className="text-xs font-medium mb-2 flex items-center gap-1" style={{ color: 'var(--text-tertiary)' }}>
              ⏳ 仅元数据 ({metaPapers.length})
            </div>
            <div className="space-y-1.5">
              {metaPapers.map((p) => (
                <PaperCard
                  key={p.id}
                  paper={p}
                  expanded={expandedPaper === p.id}
                  onToggle={() => setExpandedPaper(expandedPaper === p.id ? null : p.id)}
                />
              ))}
            </div>
          </div>
        )}

        {!loading && filteredPapers.length === 0 && (
          <div className="text-center py-8 text-sm" style={{ color: 'var(--text-tertiary)' }}>
            暂无论文
          </div>
        )}
      </div>
    </div>
  )
}

function PaperCard({ paper, expanded, onToggle }: { paper: Paper; expanded: boolean; onToggle: () => void }) {
  return (
    <div
      className="rounded-lg border p-2.5 cursor-pointer transition-colors hover:bg-gray-50 dark:hover:bg-gray-800/50"
      style={{ borderColor: 'var(--border-color)' }}
      onClick={onToggle}
    >
      <div className="flex items-start gap-2">
        <FileText size={14} className="shrink-0 mt-0.5" style={{ color: 'var(--accent)' }} />
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium leading-snug" style={{ color: 'var(--text-primary)' }}>
            {paper.title}
          </div>
          <div className="text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>
            {paper.authors.slice(0, 2).join(', ')}
            {paper.authors.length > 2 && ' et al.'}
            {paper.year && ` · ${paper.year}`}
          </div>
        </div>
        {paper.url && (
          <a
            href={paper.url}
            target="_blank"
            rel="noopener noreferrer"
            className="shrink-0 p-0.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
            onClick={(e) => e.stopPropagation()}
          >
            <ExternalLink size={12} style={{ color: 'var(--text-tertiary)' }} />
          </a>
        )}
      </div>

      {expanded && paper.abstract && (
        <div
          className="mt-2 text-xs leading-relaxed"
          style={{ color: 'var(--text-secondary)' }}
        >
          {paper.abstract.slice(0, 300)}
          {paper.abstract.length > 300 && '...'}
        </div>
      )}

      {expanded && (
        <div className="mt-2 flex items-center gap-2 text-xs" style={{ color: 'var(--text-tertiary)' }}>
          <span>引用: {paper.citation_count}</span>
          <span>·</span>
          <span>{paper.source}</span>
        </div>
      )}
    </div>
  )
}
