import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import { cn } from '@/lib/utils';
import { MtpActionCard } from './MtpActionCard';
import type { Message } from '@/types';

interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div
      className={cn(
        'flex gap-3 mb-4',
        isUser ? 'justify-end' : 'justify-start'
      )}
    >
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0">
          <span className="text-xs font-medium text-primary">P</span>
        </div>
      )}

      <div
        className={cn(
          'max-w-[80%] rounded-lg p-4',
          isUser
            ? 'bg-primary/20 text-foreground'
            : 'glass-card'
        )}
      >
        {/* Message Content */}
        <div className={cn(
          'prose prose-invert prose-sm max-w-none',
          'prose-p:my-2 prose-p:leading-relaxed',
          'prose-pre:bg-black/40 prose-pre:border prose-pre:border-white/10',
          'prose-code:text-primary prose-code:bg-black/20 prose-code:px-1 prose-code:rounded'
        )}>
          <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
            {message.content}
          </ReactMarkdown>
        </div>

        {/* MTP Actions */}
        {message.mtpActions && message.mtpActions.length > 0 && (
          <div className="mt-3 space-y-2">
            {message.mtpActions.map((action) => (
              <MtpActionCard key={action.id} action={action} />
            ))}
          </div>
        )}

        {/* Timestamp */}
        <div className="mt-2 text-xs text-muted-foreground">
          {new Date(message.timestamp).toLocaleTimeString()}
        </div>
      </div>

      {isUser && (
        <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center flex-shrink-0">
          <span className="text-xs font-medium">U</span>
        </div>
      )}
    </div>
  );
}
