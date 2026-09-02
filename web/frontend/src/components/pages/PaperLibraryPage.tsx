/** 论文库页面 */

import { useState, useEffect, useRef, type CSSProperties, type FormEvent } from 'react'
import { type Paper, type PaperFullText, fetchPaperFullText, fetchPapers, deletePaper } from '@/lib/api'
import { Search, FileText, ExternalLink, Loader2, RefreshCw, BookOpen, X, AlertCircle, Trash2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { MarkdownRenderer } from '@/components/shared/MarkdownRenderer'

export function PaperLibraryPage() {
  const [papers, setPapers] = useState<Paper[]>([])
  const [loading, setLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [filter, setFilter] = useState<'all' | 'parsed' | 'unparsed'>('all')
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [readerPaper, setReaderPaper] = useState<Paper | null>(null)
  const [readerData, setReaderData] = useState<PaperFullText | null>(null)
  const [readerLoading, setReaderLoading] = useState(false)
  const [readerError, setReaderError] = useState<string | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<Paper | null>(null)
  const readerRequestId = useRef(0)

  const loadPapers = async () => {
    setLoading(true)
    try {
      const data = await fetchPapers({ q: searchQuery || undefined })
      setPapers(data)
    } catch (e) {
      console.error('Failed to load papers:', e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadPapers()
  }, [])

  const handleSearch = (e: FormEvent) => {
    e.preventDefault()
    loadPapers()
  }

  const filtered = papers.filter((p) => {
    if (filter === 'parsed') return p.is_parsed
    if (filter === 'unparsed') return !p.is_parsed
    return true
  })

  const openReader = async (paper: Paper) => {
    const requestId = readerRequestId.current + 1
    readerRequestId.current = requestId
    setReaderPaper(paper)
    setExpandedId(null)
    setReaderData(null)
    setReaderError(null)
    setReaderLoading(true)
    try {
      const data = await fetchPaperFullText(paper.id)
      if (requestId !== readerRequestId.current) return
      setReaderData(data)
    } catch (e) {
      if (requestId !== readerRequestId.current) return
      console.error('Failed to load paper full text:', e)
      setReaderError('全文暂时无法加载，可能还没有可读取的解析文本。')
    } finally {
      if (requestId === readerRequestId.current) {
        setReaderLoading(false)
      }
    }
  }

  const handlePaperClick = (paper: Paper) => {
    if (paper.is_parsed) {
      void openReader(paper)
      return
    }
    setExpandedId(expandedId === paper.id ? null : paper.id)
  }

  const handleConfirmDelete = async () => {
    if (!deleteTarget) return
    try {
      await deletePaper(deleteTarget.id)
      setPapers((prev) => prev.filter((p) => p.id !== deleteTarget.id))
      if (readerPaper?.id === deleteTarget.id) {
        setReaderPaper(null)
        setReaderData(null)
      }
    } catch (e) {
      console.error('Failed to delete paper:', e)
    }
    setDeleteTarget(null)
  }

  return (
    <div className="flex flex-col h-full" style={{ backgroundColor: 'var(--bg-primary)' }}>
      {/* 顶栏 */}
      <div className="px-6 py-4 border-b shrink-0" style={{ borderColor: 'var(--border-color)' }}>
        <div className="flex items-center justify-between mb-3">
          <h1 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>📄 论文库</h1>
          <button
            onClick={loadPapers}
            className="p-1.5 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            title="刷新"
          >
            <RefreshCw size={16} style={{ color: 'var(--text-secondary)' }} />
          </button>
        </div>

        <div className="flex items-center gap-3">
          {/* 搜索框 */}
          <form onSubmit={handleSearch} className="flex-1 relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-tertiary)' }} />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="搜索论文标题..."
              className="w-full pl-9 pr-3 py-2 text-sm rounded-lg border outline-none focus:ring-1"
              style={{
                borderColor: 'var(--border-color)',
                backgroundColor: 'var(--bg-secondary)',
                color: 'var(--text-primary)',
              }}
            />
          </form>

          {/* 筛选 */}
          <div className="flex items-center gap-1 border rounded-lg p-0.5" style={{ borderColor: 'var(--border-color)' }}>
            {([
              { key: 'all', label: '全部' },
              { key: 'parsed', label: '已解析' },
              { key: 'unparsed', label: '未解析' },
            ] as const).map((opt) => (
              <button
                key={opt.key}
                onClick={() => setFilter(opt.key)}
                className={cn(
                  'px-3 py-1.5 text-xs rounded-md transition-colors',
                  filter === opt.key ? 'font-medium' : '',
                )}
                style={{
                  backgroundColor: filter === opt.key ? 'var(--bg-tertiary)' : 'transparent',
                  color: filter === opt.key ? 'var(--text-primary)' : 'var(--text-secondary)',
                }}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="flex-1 min-h-0 flex flex-col md:flex-row overflow-hidden">
        {/* 论文列表 */}
        <div className="flex-1 min-w-0 overflow-y-auto px-6 py-4">
          {loading ? (
            <div className="flex items-center justify-center py-20">
              <Loader2 size={20} className="animate-spin" style={{ color: 'var(--text-tertiary)' }} />
            </div>
          ) : filtered.length === 0 ? (
            <div className="text-center py-20" style={{ color: 'var(--text-tertiary)' }}>
              <FileText size={40} className="mx-auto mb-3 opacity-30" />
              <div className="text-sm">暂无论文</div>
              <div className="text-xs mt-1">使用对话中的 paper_search 工具检索论文</div>
            </div>
          ) : (
            <div className="space-y-2 max-w-4xl mx-auto">
              <div className="text-xs mb-3" style={{ color: 'var(--text-tertiary)' }}>
                共 {filtered.length} 篇论文
              </div>
              {filtered.map((paper) => (
                <div
                  key={paper.id}
                  className={cn(
                    'border rounded-lg p-4 cursor-pointer transition-colors hover:bg-gray-50 dark:hover:bg-gray-800/50',
                    readerPaper?.id === paper.id ? 'ring-1' : '',
                  )}
                  style={{
                    borderColor: readerPaper?.id === paper.id ? 'var(--accent)' : 'var(--border-color)',
                    '--tw-ring-color': 'var(--accent)',
                  } as CSSProperties}
                  onClick={() => handlePaperClick(paper)}
                >
                <div className="flex items-start gap-3">
                  <div
                    className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 mt-0.5"
                    style={{ backgroundColor: paper.is_parsed ? 'var(--accent-light)' : 'var(--bg-tertiary)' }}
                  >
                    <FileText size={14} style={{ color: paper.is_parsed ? 'var(--accent)' : 'var(--text-tertiary)' }} />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-medium leading-snug" style={{ color: 'var(--text-primary)' }}>
                      {paper.title}
                    </div>
                    <div className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
                      {paper.authors.slice(0, 3).join(', ')}
                      {paper.authors.length > 3 && ' et al.'}
                      {paper.year && ` · ${paper.year}`}
                      {paper.citation_count > 0 && ` · 引用 ${paper.citation_count}`}
                    </div>
                    <div className="flex items-center gap-2 mt-2">
                      <span
                        className="text-xs px-1.5 py-0.5 rounded"
                        style={{
                          backgroundColor: paper.is_parsed ? 'rgba(34,197,94,0.1)' : 'var(--bg-tertiary)',
                          color: paper.is_parsed ? 'var(--success)' : 'var(--text-tertiary)',
                        }}
                      >
                        {paper.is_parsed ? '✅ 已解析' : '⏳ 仅元数据'}
                      </span>
                      <span className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
                        {paper.source}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-1 shrink-0">
                    {paper.is_parsed && (
                      <button
                        type="button"
                        className="p-1.5 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
                        onClick={(e) => {
                          e.stopPropagation()
                          void openReader(paper)
                        }}
                        title="阅读全文"
                      >
                        <BookOpen size={14} style={{ color: 'var(--accent)' }} />
                      </button>
                    )}
                    {paper.url && (
                      <a
                        href={paper.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-1.5 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <ExternalLink size={14} style={{ color: 'var(--text-tertiary)' }} />
                      </a>
                    )}
                    <button
                      type="button"
                      className="p-1.5 rounded-md hover:bg-red-100 dark:hover:bg-red-900/30 transition-colors"
                      onClick={(e) => {
                        e.stopPropagation()
                        setDeleteTarget(paper)
                      }}
                      title="删除"
                    >
                      <Trash2 size={14} style={{ color: 'var(--error)' }} />
                    </button>
                  </div>
                </div>

                {/* 展开详情 */}
                {expandedId === paper.id && paper.abstract && (
                  <div className="mt-3 pt-3 border-t" style={{ borderColor: 'var(--border-color)' }}>
                    <div className="text-xs font-medium mb-1.5" style={{ color: 'var(--text-tertiary)' }}>摘要</div>
                    <div className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                      {paper.abstract}
                    </div>
                    <div className="mt-2 flex items-center gap-4 text-xs" style={{ color: 'var(--text-tertiary)' }}>
                      <span>ID: {paper.id}</span>
                    </div>
                  </div>
                )}
                </div>
              ))}
            </div>
          )}
        </div>

        {readerPaper && (
          <aside
            className="w-full md:w-[460px] lg:w-[540px] min-h-[320px] md:h-full border-t md:border-t-0 md:border-l flex flex-col"
            style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}
          >
            <div className="px-4 py-3 border-b shrink-0" style={{ borderColor: 'var(--border-color)' }}>
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0" style={{ backgroundColor: 'var(--accent-light)' }}>
                  <BookOpen size={15} style={{ color: 'var(--accent)' }} />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-medium leading-snug line-clamp-2" style={{ color: 'var(--text-primary)' }}>
                    {readerPaper.title}
                  </div>
                  <div className="text-xs mt-1 truncate" style={{ color: 'var(--text-tertiary)' }}>
                    {readerPaper.authors.slice(0, 3).join(', ')}
                    {readerPaper.authors.length > 3 && ' et al.'}
                    {readerPaper.year && ` · ${readerPaper.year}`}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    readerRequestId.current += 1
                    setReaderPaper(null)
                    setReaderData(null)
                    setReaderError(null)
                  }}
                  className="p-1.5 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors shrink-0"
                  title="关闭"
                >
                  <X size={16} style={{ color: 'var(--text-secondary)' }} />
                </button>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto px-5 py-4">
              {readerLoading ? (
                <div className="flex items-center justify-center py-16">
                  <Loader2 size={20} className="animate-spin" style={{ color: 'var(--text-tertiary)' }} />
                </div>
              ) : readerError ? (
                <div className="flex items-start gap-2 text-sm rounded-md border p-3" style={{ borderColor: 'var(--border-color)', color: 'var(--text-secondary)' }}>
                  <AlertCircle size={16} className="mt-0.5 shrink-0" style={{ color: 'var(--warning)' }} />
                  <span>{readerError}</span>
                </div>
              ) : readerData ? (
                <div className="text-sm leading-7" style={{ color: 'var(--text-secondary)' }}>
                  <MarkdownRenderer content={readerData.content} />
                </div>
              ) : null}
            </div>
          </aside>
        )}
      </div>
      {/* 删除确认弹窗 */}
      {deleteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ backgroundColor: 'rgba(0,0,0,0.4)' }} onClick={() => setDeleteTarget(null)}>
          <div
            className="w-80 rounded-xl border shadow-xl p-5"
            style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-primary)' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>删除论文</div>
              <button onClick={() => setDeleteTarget(null)} className="p-1 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
                <X size={14} style={{ color: 'var(--text-tertiary)' }} />
              </button>
            </div>
            <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>确定要从论文库中移除以下论文吗？</p>
            <p className="text-xs mb-5 truncate font-medium" style={{ color: 'var(--text-primary)' }}>{deleteTarget.title}</p>
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
