import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy } from 'lucide-react';

interface MarkdownRendererProps {
  content: string;
}

export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <ReactMarkdown
      components={{
        code({ node, inline, className, children, ...props }: any) {
          const match = /language-(\w+)/.exec(className || '');
          return !inline && match ? (
            <div className="rounded-md overflow-hidden my-4 border border-white/10 bg-black/40">
              <div className="px-4 py-1.5 text-xs text-slate-400 border-b border-white/5 flex items-center justify-between bg-black/40">
                <span className="uppercase tracking-wider font-bold">{match[1]}</span>
                <button
                  onClick={() => navigator.clipboard.writeText(String(children).replace(/\n$/, ''))}
                  className="hover:text-white transition-colors flex items-center gap-1"
                  title="Copy code"
                >
                  <Copy className="w-3 h-3" />
                </button>
              </div>
              <SyntaxHighlighter
                {...props}
                style={vscDarkPlus as any}
                language={match[1]}
                PreTag="div"
                customStyle={{ margin: 0, background: 'transparent', padding: '1rem' }}
              >
                {String(children).replace(/\n$/, '')}
              </SyntaxHighlighter>
            </div>
          ) : (
            <code {...props} className="bg-black/30 px-1.5 py-0.5 rounded text-primary/90 font-mono text-[0.9em]">
              {children}
            </code>
          );
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );
}