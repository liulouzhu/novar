/** 空会话欢迎页 */

import { useSessionStore } from '@/stores/sessionStore'
import { Search, FileText, Brain, Code } from 'lucide-react'

export function WelcomeScreen() {
  const createSession = useSessionStore((s) => s.createSession)

  const handleNewSession = async () => {
    await createSession()
  }

  const features = [
    { icon: <Search size={20} />, title: '论文检索', desc: '从 arXiv、Semantic Scholar 搜索学术论文' },
    { icon: <FileText size={20} />, title: 'PDF 解析', desc: '解析论文 PDF，建立向量索引' },
    { icon: <Brain size={20} />, title: '语义问答', desc: '基于 RAG 在已解析论文中检索答案' },
    { icon: <Code size={20} />, title: '数据分析', desc: '执行 Python 代码进行统计分析' },
  ]

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-8" style={{ backgroundColor: 'var(--bg-primary)' }}>
      <div className="text-center max-w-lg">
        <div className="text-4xl mb-4">🔬</div>
        <h1 className="text-2xl font-bold mb-2" style={{ color: 'var(--text-primary)' }}>
          Novare 研究助手
        </h1>
        <p className="text-sm mb-8" style={{ color: 'var(--text-secondary)' }}>
          你的智能科研助手，帮你检索论文、解析文献、构建知识图谱
        </p>

        <div className="grid grid-cols-2 gap-3 mb-8">
          {features.map((f) => (
            <div
              key={f.title}
              className="p-4 rounded-xl border text-left transition-colors hover:bg-gray-50 dark:hover:bg-gray-800/50"
              style={{ borderColor: 'var(--border-color)' }}
            >
              <div className="mb-2" style={{ color: 'var(--accent)' }}>{f.icon}</div>
              <div className="text-sm font-medium mb-1" style={{ color: 'var(--text-primary)' }}>{f.title}</div>
              <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>{f.desc}</div>
            </div>
          ))}
        </div>

        <button
          onClick={handleNewSession}
          className="px-6 py-2.5 rounded-lg text-sm font-medium text-white transition-colors hover:opacity-90"
          style={{ backgroundColor: 'var(--accent)' }}
        >
          开始对话
        </button>
      </div>
    </div>
  )
}
