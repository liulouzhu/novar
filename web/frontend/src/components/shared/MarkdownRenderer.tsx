/** Markdown + LaTeX + 代码高亮渲染器 */

import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import remarkGfm from 'remark-gfm'
import rehypeKatex from 'rehype-katex'
import rehypeHighlight from 'rehype-highlight'
import { useState } from 'react'
import { Copy, Check } from 'lucide-react'

interface Props {
  content: string
}

export function MarkdownRenderer({ content }: Props) {
  const normalizedContent = normalizeHtmlTables(content)

  return (
    <ReactMarkdown
      remarkPlugins={[remarkMath, remarkGfm]}
      rehypePlugins={[rehypeKatex, rehypeHighlight]}
      components={{
        // 代码块：添加复制按钮
        pre: ({ children, ...props }) => (
          <div className="relative group my-3">
            <pre {...props} className="overflow-x-auto rounded-lg text-sm">
              {children}
            </pre>
            <CopyButton />
          </div>
        ),
        // 表格样式
        table: ({ children, ...props }) => (
          <div className="overflow-x-auto my-3">
            <table {...props} className="min-w-full border-collapse text-sm">
              {children}
            </table>
          </div>
        ),
        th: ({ children, ...props }) => (
          <th
            {...props}
            className="px-3 py-2 text-left text-xs font-medium border"
            style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-tertiary)' }}
          >
            {children}
          </th>
        ),
        td: ({ children, ...props }) => (
          <td
            {...props}
            className="px-3 py-2 text-sm border"
            style={{ borderColor: 'var(--border-color)' }}
          >
            {children}
          </td>
        ),
        // 链接在新窗口打开
        a: ({ children, href, ...props }) => (
          <a
            {...props}
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="underline hover:no-underline"
            style={{ color: 'var(--accent)' }}
          >
            {children}
          </a>
        ),
        // 引用样式
        blockquote: ({ children, ...props }) => (
          <blockquote
            {...props}
            className="border-l-4 pl-4 my-3 italic"
            style={{ borderColor: 'var(--accent)', color: 'var(--text-secondary)' }}
          >
            {children}
          </blockquote>
        ),
      }}
    >
      {normalizedContent}
    </ReactMarkdown>
  )
}

function normalizeHtmlTables(content: string): string {
  if (typeof DOMParser === 'undefined' || !content.includes('<table')) {
    return content
  }
  return content.replace(/<table[\s\S]*?<\/table>/gi, (tableHtml) => {
    const markdownTable = htmlTableToMarkdown(tableHtml)
    return markdownTable || tableHtml
  })
}

function htmlTableToMarkdown(tableHtml: string): string | null {
  const parser = new DOMParser()
  const doc = parser.parseFromString(tableHtml, 'text/html')
  const rows = Array.from(doc.querySelectorAll('tr'))
    .map((row) =>
      Array.from(row.children)
        .filter((cell) => ['TD', 'TH'].includes(cell.tagName))
        .map((cell) => escapeTableCell(cell.textContent || '')),
    )
    .filter((row) => row.length > 0)

  if (rows.length === 0) return null

  const columnCount = Math.max(...rows.map((row) => row.length))
  const paddedRows = rows.map((row) => [
    ...row,
    ...Array.from({ length: columnCount - row.length }, () => ''),
  ])
  const header = paddedRows[0]
  const body = paddedRows.slice(1)

  return [
    '',
    `| ${header.join(' | ')} |`,
    `| ${header.map(() => '---').join(' | ')} |`,
    ...body.map((row) => `| ${row.join(' | ')} |`),
    '',
  ].join('\n')
}

function escapeTableCell(value: string): string {
  return value
    .replace(/\s+/g, ' ')
    .replace(/\|/g, '\\|')
    .trim()
}

function CopyButton() {
  const [copied, setCopied] = useState(false)

  const handleCopy = (e: React.MouseEvent) => {
    const pre = (e.currentTarget as HTMLElement).closest('div')?.querySelector('pre')
    if (!pre) return
    const code = pre.textContent || ''
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <button
      onClick={handleCopy}
      className="absolute top-2 right-2 p-1.5 rounded-md opacity-0 group-hover:opacity-100 transition-opacity"
      style={{ backgroundColor: 'var(--bg-tertiary)' }}
      title="复制代码"
    >
      {copied ? (
        <Check size={12} style={{ color: 'var(--success)' }} />
      ) : (
        <Copy size={12} style={{ color: 'var(--text-tertiary)' }} />
      )}
    </button>
  )
}
