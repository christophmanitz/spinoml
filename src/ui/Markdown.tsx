// Shared GitHub-flavored-Markdown renderer with the SpinoML dark-theme element
// styling. Used by the chat panel and the model-explanation modal so both render
// LLM markdown identically.

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export default function Markdown({ children }: { children: string }) {
  return (
    <div className="space-y-2 leading-relaxed [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p className="my-1.5">{children}</p>,
          ul: ({ children }) => <ul className="my-1.5 ml-4 list-disc space-y-1">{children}</ul>,
          ol: ({ children }) => <ol className="my-1.5 ml-4 list-decimal space-y-1">{children}</ol>,
          li: ({ children }) => <li className="leading-snug">{children}</li>,
          h1: ({ children }) => <h1 className="mb-1 mt-2 text-[1.08em] font-semibold text-[#e6e8eb]">{children}</h1>,
          h2: ({ children }) => <h2 className="mb-1 mt-2 text-[1em] font-semibold text-[#e6e8eb]">{children}</h2>,
          h3: ({ children }) => <h3 className="mb-1 mt-2 text-[0.92em] font-semibold text-[#cdd3da]">{children}</h3>,
          strong: ({ children }) => <strong className="font-semibold text-[#e6e8eb]">{children}</strong>,
          a: ({ children, href }) => (
            <a href={href} target="_blank" rel="noreferrer" className="text-[var(--accent)] underline hover:text-[var(--accent-hover)]">{children}</a>
          ),
          pre: ({ children }) => (
            <pre className="my-2 overflow-x-auto rounded-md border border-[#1f2429] bg-[#0b0e11] p-2.5 font-mono text-[0.85em] leading-snug text-[#cdd3da]">{children}</pre>
          ),
          code: ({ className, children }) => {
            const isBlock = (className ?? '').startsWith('language-') || String(children).includes('\n')
            if (isBlock) return <code className={className}>{children}</code>
            return <code className="rounded bg-[#1f2429] px-1 py-0.5 font-mono text-[0.85em] text-[#e6c87a]">{children}</code>
          },
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-[#3a4148] pl-3 text-[#9aa1a8]">{children}</blockquote>
          ),
          table: ({ children }) => (
            <div className="my-2 overflow-x-auto"><table className="w-full border-collapse text-[0.85em]">{children}</table></div>
          ),
          th: ({ children }) => <th className="border border-[#1f2429] px-2 py-1 text-left font-semibold text-[#cdd3da]">{children}</th>,
          td: ({ children }) => <td className="border border-[#1f2429] px-2 py-1">{children}</td>,
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  )
}
