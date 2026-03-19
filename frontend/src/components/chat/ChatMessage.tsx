import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import { cn } from '@/lib/utils';
import { MtpActionCard } from './MTPCard';
import type { Message } from '@/types';

interface ChatMessageProps {
  message: Message;
  isStreaming?: boolean;
}

/** Prose wrapper for a text block */
function TextSegment({ text, isLast, isStreaming }: { text: string; isLast: boolean; isStreaming?: boolean }) {
  if (!text) return null;

  return (
    <div className={cn(
      'prose prose-invert prose-sm max-w-none',
      // 行高放宽，阅读更舒适
      'prose-p:my-2.5 prose-p:last:mb-0 prose-p:leading-7',
      'prose-li:leading-7',
      // 代码块面板
      'prose-pre:bg-black/50 prose-pre:backdrop-blur-md',
      'prose-pre:border prose-pre:border-white/5 prose-pre:rounded-xl',
      'prose-pre:shadow-[inset_0_4px_12px_rgba(0,0,0,0.5)]',
      // 行内代码
      'prose-code:text-fuchsia-200 prose-code:bg-fuchsia-500/20 prose-code:border prose-code:border-fuchsia-500/20'
    )}>
      <ReactMarkdown
        rehypePlugins={[rehypeHighlight]}
        components={{
          p: ({ children }) => (
            <p>
              {children}
              {isLast && isStreaming && (
                <span className="inline-block w-1.5 h-4 bg-primary align-middle animate-pulse ml-1 rounded-full shadow-[0_0_8px_rgba(147,51,234,0.5)]" />
              )}
            </p>
          )
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

export function ChatMessage({ message, isStreaming }: ChatMessageProps) {
  const isUser = message.role === 'user';
  const blocks = message.contentBlocks;

  // Find the index of the last text block (for streaming cursor placement)
  let lastTextIdx = -1;
  for (let i = blocks.length - 1; i >= 0; i--) {
    if (blocks[i].kind === 'text') { lastTextIdx = i; break; }
  }

  return (
    <div
      className={cn(
        'flex gap-4 mb-8',
        isUser ? 'justify-end' : 'justify-start'
      )}
    >
      {!isUser && (
        <div className={cn(
          "w-9 h-9 rounded-xl shrink-0 flex items-center justify-center relative",
          "bg-background/40 backdrop-blur-md border border-primary/40",
        )}>
          <span className="text-sm font-bold text-primary drop-shadow-md">P</span>
        </div>
      )}

      <div
        className={cn(
          isUser
            ? [
                'max-w-[85%] rounded-2xl p-5 transition-all duration-300',
                'bg-amber-400/25',
                'backdrop-blur-3xl',
                'border border-white/10 border-b-white/5 border-r-white/5',
                'rounded-tr-sm',
                'shadow-[0_8px_32px_rgba(0,0,0,0.3)]',
                'text-foreground font-medium'
              ]
            : [
                'relative max-w-[85%] rounded-2xl p-5 transition-all duration-300',
                'bg-purple-500/10',
                'backdrop-blur-3xl',
                'border border-white/10 border-b-white/5 border-r-white/5',
                'rounded-tl-sm',
                'shadow-[0_8px_32px_rgba(0,0,0,0.4)]',
                'text-foreground'
              ]
        )}
      >
        {/* Render content blocks in order */}
        {blocks.map((block, idx) => {
          if (block.kind === 'text') {
            return (
              <TextSegment
                key={idx}
                text={block.text}
                isLast={idx === lastTextIdx}
                isStreaming={isStreaming}
              />
            );
          }
          // MTP block
          return (
            <div key={idx} className="my-3">
              <MtpActionCard action={block.action} />
            </div>
          );
        })}

        {/* Timestamp */}
        <div className="mt-2 text-xs text-muted-foreground/50 flex justify-end">
          {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </div>
      </div>

      {isUser && (
        <div className="relative w-9 h-9 rounded-full bg-white/5 backdrop-blur-md flex items-center justify-center shrink-0 border border-white/10 shadow-lg z-10">
          <span className="text-sm font-bold text-muted-foreground">U</span>
        </div>
      )}
    </div>
  );
}
