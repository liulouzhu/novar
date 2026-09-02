/** 多行输入框 */

import { useState, useRef, useCallback, useEffect } from 'react'
import { Send, Paperclip, Loader2, Search, FileText, HelpCircle, Square, Sparkles, BookOpen, Zap } from 'lucide-react'
import { uploadFile, fetchSkills, type SkillMeta } from '@/lib/api'

/** 已知 skill 的图标映射，未知 skill 使用 Zap */
const SKILL_ICONS: Record<string, React.ReactNode> = {
  research: <Search size={14} />,
  parse: <FileText size={14} />,
  ask: <HelpCircle size={14} />,
  compile: <BookOpen size={14} />,
  innovation: <Sparkles size={14} />,
}

const DEFAULT_ICON = <Zap size={14} />

interface Props {
  onSend: (content: string, refs?: Array<{ type: string; id: string; title?: string }>) => void
  onStop: () => void
  disabled: boolean
}

export function InputBox({ onSend, onStop, disabled }: Props) {
  const [text, setText] = useState('')
  const [attachments, setAttachments] = useState<Array<{ name: string; uploadId: string }>>([])
  const [uploading, setUploading] = useState(false)
  const [showSkills, setShowSkills] = useState(false)
  const [skillFilter, setSkillFilter] = useState('')
  const [selectedSkillIdx, setSelectedSkillIdx] = useState(0)
  const [skills, setSkills] = useState<SkillMeta[]>([])
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const skillListRef = useRef<HTMLDivElement>(null)

  // 动态加载 skill 列表
  useEffect(() => {
    fetchSkills().then(setSkills).catch(() => {})
  }, [])

  // 检测 "/" 触发 Skill 列表
  useEffect(() => {
    if (text === '/' || (text.startsWith('/') && !text.includes(' ') && text.length < 20)) {
      setShowSkills(true)
      setSkillFilter(text.slice(1))
      setSelectedSkillIdx(0)
    } else {
      setShowSkills(false)
    }
  }, [text])

  const filteredSkills = skillFilter
    ? skills.filter((s) => s.name.includes(skillFilter) || s.description.includes(skillFilter))
    : skills

  // 确保选中项在有效范围内
  useEffect(() => {
    if (selectedSkillIdx >= filteredSkills.length) {
      setSelectedSkillIdx(Math.max(0, filteredSkills.length - 1))
    }
  }, [filteredSkills.length, selectedSkillIdx])

  const selectSkill = (skillName: string) => {
    setText(`/${skillName} `)
    setShowSkills(false)
    textareaRef.current?.focus()
  }

  const handleSend = useCallback(() => {
    if (!text.trim() || disabled) return

    let content = text.trim()
    if (attachments.length > 0) {
      content += '\n\n附件：\n' + attachments.map((a) => `- ${a.name} (upload_id: ${a.uploadId})`).join('\n')
    }

    onSend(content)
    setText('')
    setAttachments([])
    // 重置 textarea 高度
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }, [text, disabled, attachments, onSend])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Skill 列表打开时的键盘导航
    if (showSkills && filteredSkills.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setSelectedSkillIdx((prev) => (prev + 1) % filteredSkills.length)
        return
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault()
        setSelectedSkillIdx((prev) => (prev - 1 + filteredSkills.length) % filteredSkills.length)
        return
      }
      if (e.key === 'Enter' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault()
        selectSkill(filteredSkills[selectedSkillIdx].name)
        return
      }
      if (e.key === 'Escape') {
        e.preventDefault()
        setShowSkills(false)
        return
      }
    }

    // Enter 发送（或停止），Shift+Enter 换行
    if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
      e.preventDefault()
      if (disabled) {
        onStop()
      } else {
        handleSend()
      }
    }
  }

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value)
    // 自动扩展高度
    const el = e.target
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 200) + 'px'
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files?.length) return

    setUploading(true)
    for (const file of Array.from(files)) {
      try {
        const res = await uploadFile(file)
        setAttachments((prev) => [...prev, { name: res.filename, uploadId: res.upload_id }])
      } catch (err) {
        console.error('Upload failed:', err)
      }
    }
    setUploading(false)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault()
    const files = e.dataTransfer.files
    if (!files?.length) return

    setUploading(true)
    for (const file of Array.from(files)) {
      try {
        const res = await uploadFile(file)
        setAttachments((prev) => [...prev, { name: res.filename, uploadId: res.upload_id }])
      } catch (err) {
        console.error('Upload failed:', err)
      }
    }
    setUploading(false)
  }

  const removeAttachment = (idx: number) => {
    setAttachments((prev) => prev.filter((_, i) => i !== idx))
  }

  return (
    <div
      className="border-t px-4 py-3 shrink-0"
      style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-primary)' }}
    >
      <div className="max-w-3xl mx-auto">
        {/* 附件标签 */}
        {attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-2">
            {attachments.map((a, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1 px-2 py-1 text-xs rounded-md border"
                style={{ borderColor: 'var(--border-color)', color: 'var(--text-secondary)' }}
              >
                📄 {a.name}
                <button
                  onClick={() => removeAttachment(i)}
                  className="ml-1 hover:text-red-500"
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        )}

        {/* Skill 下拉列表 */}
        {showSkills && filteredSkills.length > 0 && (
          <div
            ref={skillListRef}
            className="mb-2 rounded-lg border overflow-hidden shadow-sm"
            style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-primary)' }}
          >
            <div className="px-3 py-1.5 text-xs font-medium border-b" style={{ borderColor: 'var(--border-color)', color: 'var(--text-tertiary)' }}>
              Skill 命令 <span className="font-normal">↑↓ 选择 · Enter 确认 · Esc 关闭</span>
            </div>
            {filteredSkills.map((skill, idx) => (
              <button
                key={skill.name}
                onClick={() => selectSkill(skill.name)}
                onMouseEnter={() => setSelectedSkillIdx(idx)}
                className="w-full flex items-center gap-3 px-3 py-2 text-sm text-left transition-colors"
                style={{
                  backgroundColor: idx === selectedSkillIdx ? 'var(--bg-tertiary)' : 'transparent',
                }}
              >
                <span style={{ color: 'var(--accent)' }}>{SKILL_ICONS[skill.name] || DEFAULT_ICON}</span>
                <div className="min-w-0 flex-1">
                  <div className="font-medium" style={{ color: 'var(--text-primary)' }}>
                    /{skill.name}
                    <span className="ml-2 text-xs font-normal" style={{ color: 'var(--text-secondary)' }}>
                      {skill.description}
                    </span>
                  </div>
                  <div className="text-xs truncate" style={{ color: 'var(--text-tertiary)' }}>
                    {skill.description}
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}

        {/* 输入区域 */}
        <div
          className="flex items-end gap-2 rounded-xl border px-3 py-2 transition-colors focus-within:ring-1"
          style={{
            borderColor: 'var(--border-color)',
            backgroundColor: 'var(--bg-secondary)',
            '--tw-ring-color': 'var(--border-focus)',
          } as React.CSSProperties}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          {/* 附件按钮 */}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading || disabled}
            className="p-1.5 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors shrink-0 disabled:opacity-50"
            title="上传文件"
          >
            {uploading ? (
              <Loader2 size={16} className="animate-spin" style={{ color: 'var(--text-secondary)' }} />
            ) : (
              <Paperclip size={16} style={{ color: 'var(--text-secondary)' }} />
            )}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.csv,.xlsx,.txt,.json,.md"
            className="hidden"
            onChange={handleFileChange}
          />

          {/* 文本输入 */}
          <textarea
            ref={textareaRef}
            value={text}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder={disabled ? 'Agent 正在工作中...' : '输入研究问题... (Enter 发送，Shift+Enter 换行)'}
            disabled={disabled}
            rows={1}
            className="flex-1 resize-none bg-transparent outline-none text-sm leading-6 max-h-[200px] disabled:opacity-50"
            style={{ color: 'var(--text-primary)' }}
          />

          {/* 发送/停止按钮 */}
          {disabled ? (
            <button
              onClick={onStop}
              className="p-1.5 rounded-md transition-colors shrink-0 animate-pulse"
              style={{
                backgroundColor: 'var(--error)',
                color: 'white',
              }}
              title="停止生成"
            >
              <Square size={16} fill="currentColor" />
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={!text.trim()}
              className="p-1.5 rounded-md transition-colors shrink-0 disabled:opacity-30"
              style={{
                backgroundColor: text.trim() ? 'var(--accent)' : 'transparent',
                color: text.trim() ? 'white' : 'var(--text-tertiary)',
              }}
              title="发送 (Enter)"
            >
              <Send size={16} />
            </button>
          )}
        </div>

        <div className="text-xs mt-1.5 text-center" style={{ color: 'var(--text-tertiary)' }}>
          {disabled ? 'Enter 或点击按钮停止生成' : 'Enter 发送 · Shift+Enter 换行 · 支持拖拽上传文件 · 输入 / 使用 Skill'}
        </div>
      </div>
    </div>
  )
}
