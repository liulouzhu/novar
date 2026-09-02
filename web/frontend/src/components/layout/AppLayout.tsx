/** 三栏布局容器 */

import { useState } from 'react'
import { SessionSidebar, type PageKey } from './SessionSidebar'
import { ReferencePanel } from './ReferencePanel'
import { ChatArea } from '../chat/ChatArea'
import { WelcomeScreen } from '../chat/WelcomeScreen'
import { PaperLibraryPage } from '../pages/PaperLibraryPage'
import { KnowledgeGraphPage } from '../pages/KnowledgeGraphPage'
import { MemoryPage } from '../pages/MemoryPage'
import { useSessionStore } from '@/stores/sessionStore'

export function AppLayout() {
  const currentId = useSessionStore((s) => s.currentId)
  const [activePage, setActivePage] = useState<PageKey>('chat')
  const [showPanel, setShowPanel] = useState(true)

  const renderMainContent = () => {
    switch (activePage) {
      case 'papers':
        return <PaperLibraryPage />
      case 'graph':
        return <KnowledgeGraphPage />
      case 'memory':
        return <MemoryPage />
      case 'chat':
      default:
        return currentId ? (
          <ChatArea
            sessionId={currentId}
            panelOpen={showPanel}
            onTogglePanel={() => setShowPanel(!showPanel)}
          />
        ) : (
          <WelcomeScreen />
        )
    }
  }

  return (
    <div className="flex h-screen overflow-hidden" style={{ backgroundColor: 'var(--bg-primary)' }}>
      {/* 左侧导航栏 */}
      <SessionSidebar activePage={activePage} onNavigate={setActivePage} />

      {/* 中间主内容 */}
      <div className="flex-1 flex flex-col min-w-0">
        {renderMainContent()}
      </div>

      {/* 右侧引用面板（仅对话页） */}
      {activePage === 'chat' && currentId && showPanel && (
        <ReferencePanel sessionId={currentId} />
      )}
    </div>
  )
}
