/** 知识图谱页面 — 力导向图可视化 */

import { useState, useEffect, useRef, useCallback } from 'react'
import { type GraphData, type GraphNode, fetchGraph, fetchGraphStats, type GraphStats } from '@/lib/api'
import { RefreshCw, Loader2, ZoomIn, ZoomOut, Maximize2, Users } from 'lucide-react'

// 动态导入的组件引用
let ForceGraph2D: any = null

const NODE_COLORS: Record<string, string> = {
  Paper: '#3b82f6',
  Author: '#10b981',
  Method: '#8b5cf6',
  Task: '#ec4899',
  Dataset: '#f59e0b',
  Metric: '#06b6d4',
  Contribution: '#22c55e',
  Limitation: '#ef4444',
  Concept: '#f59e0b',
  Unknown: '#6b7280',
}

const NODE_SIZES: Record<string, number> = {
  Paper: 8,
  Author: 5,
  Method: 7,
  Task: 7,
  Dataset: 7,
  Metric: 6,
  Contribution: 6,
  Limitation: 6,
  Concept: 7,
  Unknown: 4,
}

export function KnowledgeGraphPage() {
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] })
  const [stats, setStats] = useState<GraphStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [fgReady, setFgReady] = useState(false)
  const graphRef = useRef<any>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })
  const [showAuthors, setShowAuthors] = useState(false)

  // 动态导入 react-force-graph-2d
  useEffect(() => {
    import('react-force-graph-2d').then((mod) => {
      ForceGraph2D = mod.default
      setFgReady(true)
    }).catch((e) => console.error('Failed to load graph lib:', e))
  }, [])

  // 监听容器尺寸（始终挂载，不依赖数据）
  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    const update = () => {
      const rect = el.getBoundingClientRect()
      if (rect.width > 10 && rect.height > 10) {
        setDimensions((prev) => {
          if (prev.width === Math.floor(rect.width) && prev.height === Math.floor(rect.height)) return prev
          return { width: Math.floor(rect.width), height: Math.floor(rect.height) }
        })
      }
    }

    update()
    const obs = new ResizeObserver(update)
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  // 加载数据
  const loadGraph = useCallback(() => {
    setLoading(true)
    const exclude = showAuthors ? '' : 'Author'
    Promise.all([fetchGraph(exclude), fetchGraphStats(exclude)])
      .then(([graph, s]) => {
        setGraphData(graph)
        setStats(s)
      })
      .catch((e) => console.error('Failed to load graph:', e))
      .finally(() => setLoading(false))
  }, [showAuthors])

  useEffect(() => { loadGraph() }, [loadGraph])

  const handleRefresh = () => {
    loadGraph()
  }

  const nodeColor = useCallback((node: any) => NODE_COLORS[node.type] || NODE_COLORS.Unknown, [])
  const nodeVal = useCallback((node: any) => NODE_SIZES[node.type] || NODE_SIZES.Unknown, [])
  const nodeLabel = useCallback((node: any) => node.label || node.id, [])
  const linkColor = useCallback(() => 'rgba(150,150,150,0.4)', [])
  const linkLabel = useCallback((link: any) => link.type || '', [])

  const handleNodeClick = useCallback((node: any) => {
    setSelectedNode(node as GraphNode)
    if (graphRef.current) {
      graphRef.current.centerAt(node.x, node.y, 400)
      graphRef.current.zoom(3, 400)
    }
  }, [])

  const handleBackgroundClick = useCallback(() => setSelectedNode(null), [])

  const isEmpty = graphData.nodes.length === 0
  const canRenderGraph = fgReady && ForceGraph2D && dimensions.width > 10 && !isEmpty

  return (
    <div className="flex h-full" style={{ backgroundColor: 'var(--bg-primary)' }}>
      {/* 图谱区域 */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* 顶栏 */}
        <div className="px-6 py-3 border-b flex items-center justify-between shrink-0" style={{ borderColor: 'var(--border-color)' }}>
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>🕸️ 知识图谱</h1>
            {stats && (
              <span className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
                {stats.total_nodes} 节点 · {stats.total_edges} 关系
              </span>
            )}
          </div>
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setShowAuthors((v) => !v)}
              className={`p-1.5 rounded-md transition-colors ${showAuthors ? 'bg-blue-100 dark:bg-blue-900/30' : 'hover:bg-gray-100 dark:hover:bg-gray-800'}`}
              title={showAuthors ? '隐藏作者节点' : '显示作者节点'}
            >
              <Users size={16} style={{ color: showAuthors ? '#3b82f6' : 'var(--text-secondary)' }} />
            </button>
            <button onClick={() => graphRef.current?.zoom(1.5, 300)} className="p-1.5 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors" title="放大">
              <ZoomIn size={16} style={{ color: 'var(--text-secondary)' }} />
            </button>
            <button onClick={() => graphRef.current?.zoom(0.67, 300)} className="p-1.5 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors" title="缩小">
              <ZoomOut size={16} style={{ color: 'var(--text-secondary)' }} />
            </button>
            <button onClick={() => graphRef.current?.zoomToFit(400, 50)} className="p-1.5 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors" title="适应画布">
              <Maximize2 size={16} style={{ color: 'var(--text-secondary)' }} />
            </button>
            <button onClick={handleRefresh} disabled={loading} className="p-1.5 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors" title="刷新">
              <RefreshCw size={16} className={loading ? 'animate-spin' : ''} style={{ color: 'var(--text-secondary)' }} />
            </button>
          </div>
        </div>

        {/* 画布容器 — 始终渲染以便 ResizeObserver 测量 */}
        <div ref={containerRef} className="flex-1 relative" style={{ minHeight: 0 }}>
          {loading ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <Loader2 size={24} className="animate-spin" style={{ color: 'var(--text-tertiary)' }} />
            </div>
          ) : isEmpty ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="text-4xl mb-3 opacity-20">🕸️</div>
                <div className="text-sm" style={{ color: 'var(--text-tertiary)' }}>图谱为空</div>
                <div className="text-xs mt-1" style={{ color: 'var(--text-tertiary)' }}>在对话中使用 knowledge_graph 工具构建图谱</div>
              </div>
            </div>
          ) : canRenderGraph ? (
            <ForceGraph2D
              ref={graphRef}
              graphData={graphData}
              width={dimensions.width}
              height={dimensions.height}
              backgroundColor={getComputedStyle(document.documentElement).getPropertyValue('--bg-primary').trim() || '#ffffff'}
              nodeColor={nodeColor}
              nodeVal={nodeVal}
              nodeLabel={nodeLabel}
              nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
                const label = node.label || node.id
                const fontSize = Math.max(10, 14 / globalScale)
                const r = (NODE_SIZES[node.type] || 4) + 2

                ctx.beginPath()
                ctx.arc(node.x, node.y, r, 0, 2 * Math.PI)
                ctx.fillStyle = NODE_COLORS[node.type] || NODE_COLORS.Unknown
                ctx.fill()

                if (selectedNode && node.id === selectedNode.id) {
                  ctx.strokeStyle = '#fff'
                  ctx.lineWidth = 2 / globalScale
                  ctx.stroke()
                }

                if (globalScale > 1.2) {
                  ctx.font = `${fontSize}px sans-serif`
                  ctx.textAlign = 'center'
                  ctx.textBaseline = 'top'
                  ctx.fillStyle = document.documentElement.classList.contains('dark') ? '#e5e7eb' : '#374151'
                  ctx.fillText(label.length > 20 ? label.slice(0, 20) + '...' : label, node.x, node.y + r + 3)
                }
              }}
              linkColor={linkColor}
              linkLabel={linkLabel}
              linkDirectionalArrowLength={4}
              linkDirectionalArrowRelPos={1}
              linkWidth={1}
              onNodeClick={handleNodeClick}
              onBackgroundClick={handleBackgroundClick}
              cooldownTicks={100}
              d3AlphaDecay={0.02}
              d3VelocityDecay={0.3}
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center">
              <Loader2 size={20} className="animate-spin" style={{ color: 'var(--text-tertiary)' }} />
            </div>
          )}
        </div>
      </div>

      {/* 右侧详情面板 */}
      {(selectedNode || stats) && (
        <div className="w-72 border-l overflow-y-auto shrink-0" style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}>
          {selectedNode ? (
            <div className="p-4">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: NODE_COLORS[selectedNode.type] || NODE_COLORS.Unknown }} />
                <span className="text-xs font-medium px-2 py-0.5 rounded" style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}>{selectedNode.type}</span>
              </div>
              <h2 className="text-sm font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>{selectedNode.label}</h2>
              {selectedNode.description && <p className="text-xs mb-3 leading-relaxed" style={{ color: 'var(--text-secondary)' }}>{selectedNode.description}</p>}
              <div className="space-y-1.5 text-xs" style={{ color: 'var(--text-tertiary)' }}>
                <div>ID: <span className="font-mono">{selectedNode.id}</span></div>
                {selectedNode.canonical_name && <div>规范名: {selectedNode.canonical_name}</div>}
                {selectedNode.aliases && selectedNode.aliases.length > 0 && <div>别名: {selectedNode.aliases.slice(0, 4).join(', ')}</div>}
                {selectedNode.year && <div>年份: {selectedNode.year}</div>}
                {selectedNode.citation_count > 0 && <div>引用: {selectedNode.citation_count}</div>}
              </div>
              {graphData.links.length > 0 && (
                <div className="mt-4 pt-3 border-t" style={{ borderColor: 'var(--border-color)' }}>
                  <div className="text-xs font-medium mb-2" style={{ color: 'var(--text-tertiary)' }}>关联关系</div>
                  <div className="space-y-1.5">
                    {graphData.links
                      .filter((l) => l.source === selectedNode.id || l.target === selectedNode.id)
                      .slice(0, 20)
                      .map((l, i) => {
                        const isOutgoing = l.source === selectedNode.id
                        const otherId = isOutgoing ? l.target : l.source
                        const otherNode = graphData.nodes.find((n) => n.id === otherId)
                        return (
                          <div key={i} className="text-xs p-2 rounded" style={{ backgroundColor: 'var(--bg-tertiary)' }}>
                            <span style={{ color: 'var(--text-secondary)' }}>{isOutgoing ? '→' : '←'}</span>
                            <span className="mx-1 font-medium" style={{ color: 'var(--accent)' }}>{l.type}</span>
                            <span style={{ color: 'var(--text-secondary)' }}>→ {otherNode?.label || otherId}</span>
                          </div>
                        )
                      })}
                  </div>
                </div>
              )}
            </div>
          ) : stats && (
            <div className="p-4">
              <h2 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>图谱统计</h2>
              <div className="space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <span style={{ color: 'var(--text-secondary)' }}>节点总数</span>
                  <span className="font-medium" style={{ color: 'var(--text-primary)' }}>{stats.total_nodes}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span style={{ color: 'var(--text-secondary)' }}>关系总数</span>
                  <span className="font-medium" style={{ color: 'var(--text-primary)' }}>{stats.total_edges}</span>
                </div>
                <div className="pt-3 border-t" style={{ borderColor: 'var(--border-color)' }}>
                  <div className="text-xs font-medium mb-2" style={{ color: 'var(--text-tertiary)' }}>节点类型</div>
                  {Object.entries(stats.node_types).map(([type, count]) => (
                    <div key={type} className="flex items-center gap-2 py-1 text-xs">
                      <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: NODE_COLORS[type] || NODE_COLORS.Unknown }} />
                      <span style={{ color: 'var(--text-secondary)' }}>{type}</span>
                      <span className="ml-auto font-medium" style={{ color: 'var(--text-primary)' }}>{count}</span>
                    </div>
                  ))}
                </div>
                <div className="pt-3 border-t" style={{ borderColor: 'var(--border-color)' }}>
                  <div className="text-xs font-medium mb-2" style={{ color: 'var(--text-tertiary)' }}>关系类型</div>
                  {Object.entries(stats.edge_types).map(([type, count]) => (
                    <div key={type} className="flex items-center justify-between py-1 text-xs">
                      <span style={{ color: 'var(--text-secondary)' }}>{type}</span>
                      <span className="font-medium" style={{ color: 'var(--text-primary)' }}>{count}</span>
                    </div>
                  ))}
                </div>
                <div className="pt-3 border-t" style={{ borderColor: 'var(--border-color)' }}>
                  <div className="text-xs font-medium mb-2" style={{ color: 'var(--text-tertiary)' }}>图例</div>
                  {Object.entries(NODE_COLORS).filter(([k]) => k !== 'Unknown').map(([type, color]) => (
                    <div key={type} className="flex items-center gap-2 py-1 text-xs">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
                      <span style={{ color: 'var(--text-secondary)' }}>{type}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
