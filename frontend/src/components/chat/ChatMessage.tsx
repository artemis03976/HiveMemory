import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import { cn } from '@/lib/utils';
import { MtpActionCard } from './MTPCard';
import type { Message } from '@/types';

interface ChatMessageProps {
  message: Message;
  isStreaming?: boolean;
}

export function ChatMessage({ message, isStreaming }: ChatMessageProps) {
  const isUser = message.role === 'user';

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
                // 1. 无色、微亮的半透明白
                'bg-amber-400/25',
                // 2. 极致模糊，粉碎背后的图案
                'backdrop-blur-3xl',
                // 3. 极细的高光切面边框
                'border border-white/10 border-b-white/5 border-r-white/5', 
                'rounded-tr-sm',
                // 4. 纯净的深色投影（去掉金色发光）
                'shadow-[0_8px_32px_rgba(0,0,0,0.3)]',
                'text-foreground font-medium' // 提升文字清晰度
              ]
            : [
                'relative max-w-[85%] rounded-2xl p-5 transition-all duration-300',
                // 1. 紫粉水晶渐变 (帕秋莉色系，极低透明度保证可读性)
                'bg-purple-500/10',
                // 2. 同样极致的模糊
                'backdrop-blur-3xl',
                // 3. 统一的物理边框
                'border border-white/10 border-b-white/5 border-r-white/5',
                'rounded-tl-sm',
                // 4. 更深的悬浮阴影
                'shadow-[0_8px_32px_rgba(0,0,0,0.4)]',
                'text-foreground'
              ]
        )}
      >
        {/* Message Content */}
        <div className={cn(
          'prose prose-invert prose-sm max-w-none',
          // 优化段落间距
          'prose-p:my-2 prose-p:last:mb-0',
          // 代码块面板：使其在玻璃内部凹陷
          'prose-pre:bg-black/50 prose-pre:backdrop-blur-md',
          'prose-pre:border prose-pre:border-white/5 prose-pre:rounded-xl',
          'prose-pre:shadow-[inset_0_4px_12px_rgba(0,0,0,0.5)]', // 强烈的内部凹陷阴影
          // 行内代码：轻量高亮
          'prose-code:text-fuchsia-200 prose-code:bg-fuchsia-500/20 prose-code:border prose-code:border-fuchsia-500/20'
        )}>
          <ReactMarkdown 
            rehypePlugins={[rehypeHighlight]}
            components={{
              // Custom renderer to append cursor if streaming
              p: ({children}) => (
                <p>
                  {children}
                  {isStreaming && (
                    <span className="inline-block w-1.5 h-4 bg-primary align-middle animate-pulse ml-1 rounded-full shadow-[0_0_8px_rgba(147,51,234,0.5)]" />
                  )}
                </p>
              )
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>

        {/* MTP Actions */}
        {message.mtpActions && message.mtpActions.length > 0 && (
          <div className="mt-4 space-y-2">
            {message.mtpActions.map((action) => (
              <MtpActionCard key={action.id} action={action} />
            ))}
          </div>
        )}

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
