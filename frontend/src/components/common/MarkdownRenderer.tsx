import ReactMarkdown from 'react-markdown';
import type { Components } from 'react-markdown';
import type { ComponentProps } from 'react';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy } from 'lucide-react';

interface MarkdownRendererProps {
  content: string;
}

type CodeComponentProps = ComponentProps<'code'> & {
  inline?: boolean;
  node?: unknown;
};

export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  const components = {
    h1({ children, ...props }) {
      return <h1 {...props} className="text-2xl font-bold text-slate-100 mt-6 mb-4">{children}</h1>;
    },
    h2({ children, ...props }) {
      return <h2 {...props} className="text-xl font-bold text-slate-100 mt-5 mb-3">{children}</h2>;
    },
    h3({ children, ...props }) {
      return <h3 {...props} className="text-lg font-bold text-slate-200 mt-4 mb-2">{children}</h3>;
    },
    h4({ children, ...props }) {
      return <h4 {...props} className="text-base font-bold text-slate-200 mt-3 mb-2">{children}</h4>;
    },
    p({ children, ...props }) {
      return <p {...props} className="mb-3 text-slate-300 leading-relaxed">{children}</p>;
    },
    ul({ children, ...props }) {
      return <ul {...props} className="list-disc list-outside ml-6 mb-4 text-slate-300 space-y-1">{children}</ul>;
    },
    ol({ children, ...props }) {
      return <ol {...props} className="list-decimal list-outside ml-6 mb-4 text-slate-300 space-y-1">{children}</ol>;
    },
    li({ children, ...props }) {
      return <li {...props} className="leading-relaxed">{children}</li>;
    },
    blockquote({ children, ...props }) {
      return <blockquote {...props} className="border-l-4 border-primary/50 pl-4 py-1 my-4 bg-primary/5 rounded-r text-slate-300 italic">{children}</blockquote>;
    },
    a({ children, ...props }) {
      return <a {...props} className="text-primary hover:text-primary-light underline decoration-primary/30 underline-offset-2 transition-colors" target="_blank" rel="noopener noreferrer">{children}</a>;
    },
    strong({ children, ...props }) {
      return <strong {...props} className="text-primary font-bold">{children}</strong>;
    },
    em({ children, ...props }) {
      return <em {...props} className="italic text-slate-300">{children}</em>;
    },
    del({ children, ...props }) {
      return <del {...props} className="line-through text-slate-500">{children}</del>;
    },
    table({ children, ...props }) {
      return (
        <div className="w-full overflow-x-auto my-4 rounded-lg border border-white/10">
          <table {...props} className="w-full text-left border-collapse text-sm">{children}</table>
        </div>
      );
    },
    thead({ children, ...props }) {
      return <thead {...props} className="bg-black/40 border-b border-white/10">{children}</thead>;
    },
    tbody({ children, ...props }) {
      return <tbody {...props} className="divide-y divide-white/5">{children}</tbody>;
    },
    tr({ children, ...props }) {
      return <tr {...props} className="hover:bg-white/5 transition-colors">{children}</tr>;
    },
    th({ children, ...props }) {
      return <th {...props} className="px-4 py-3 font-semibold text-slate-200">{children}</th>;
    },
    td({ children, ...props }) {
      return <td {...props} className="px-4 py-3 text-slate-300">{children}</td>;
    },
    hr({ ...props }) {
      return <hr {...props} className="my-6 border-white/10" />;
    },
    code({ inline, className, children, ...props }: CodeComponentProps) {
      const match = /language-(\w+)/.exec(className || '');
      
      return !inline && match ? (
        <div className="code-block-wrapper rounded-md overflow-hidden my-4 border border-white/10 bg-black/40">
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
            style={vscDarkPlus}
            language={match[1]}
            PreTag="div"
            className="scrollbar-thin scrollbar-thumb-white/20 scrollbar-track-transparent"
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
  } satisfies Components;

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={components}
    >
      {content}
    </ReactMarkdown>
  );
}
