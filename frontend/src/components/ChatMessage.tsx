import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import { cn } from '@/lib/utils';
import { MtpActionCard } from './MtpActionCard';
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
          "shadow-[0_0_15px_rgba(139,92,246,0.3)]" // 专属的紫色魔法光晕
        )}>
          <span className="text-sm font-bold text-primary drop-shadow-md">P</span>
        </div>
      )}

      <div
        className={cn(
          isUser
            ? [
                'max-w-[85%] rounded-3xl p-5 transition-all duration-300',
                // 1. 暖色水晶渐变底色 (极低透明度)
                'bg-linear-to-br from-amber-500/20 to-rose-500/20',
                // 2. 强模糊度，透出极光背景
                'backdrop-blur-xl',
                // 3. 晶体边框
                'border border-amber-500/30',
                // 4. 右上角对话尖角
                'rounded-tr-sm',
                // 5. 光影：Inset高光 + 琥珀色外发光
                'shadow-[inset_0_1px_1px_rgba(255,255,255,0.25),0_8px_24px_rgba(245,158,11,0.1)]',
                'text-foreground font-medium' // 提升文字清晰度
              ]
            : [
                'relative max-w-[85%] rounded-3xl p-5 transition-all duration-300',
                // 1. 紫粉水晶渐变 (帕秋莉色系，极低透明度保证可读性)
                'bg-linear-to-br from-purple-500/10 via-fuchsia-500/10 to-transparent',
                // 2. 极强模糊度 (让后面的极光在这里糊化，托住黑色文字)
                'backdrop-blur-2xl',
                // 3. 晶莹剔透的银色边框
                'border border-white/10',
                // 4. 左上角对话尖角
                'rounded-tl-sm',
                // 5. 光影：Inset高光 + 悬浮深色重阴影 (托住视线)
                'shadow-[inset_0_1px_1px_rgba(255,255,255,0.15),0_8px_32px_rgba(0,0,0,0.4)]',
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
