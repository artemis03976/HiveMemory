import { useState } from 'react';
import TextareaAutosize from 'react-textarea-autosize';
import { Send, Paperclip, Hash } from 'lucide-react';
import { cn } from '@/lib/utils';

interface OmniInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
}

export function OmniInput({ onSend, disabled }: OmniInputProps) {
  const [message, setMessage] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSend(message);
      setMessage('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="glass-input rounded-2xl p-2 flex flex-col focus-within:ring-2 ring-primary transition-all duration-200">
      {/* Toolbar */}
      <div className="flex gap-2 pb-2 border-b border-white/10 mb-2">
        <button
          type="button"
          className="p-2 rounded-lg hover:bg-white/5 transition-colors cursor-pointer"
          aria-label="Attach file"
        >
          <Paperclip className="w-4 h-4 text-muted-foreground" />
        </button>
        <button
          type="button"
          className="p-2 rounded-lg hover:bg-white/5 transition-colors cursor-pointer"
          aria-label="Reference memory"
        >
          <Hash className="w-4 h-4 text-muted-foreground" />
        </button>
      </div>

      {/* Input Area */}
      <form onSubmit={handleSubmit} className="flex items-end gap-2">
        <TextareaAutosize
          id="message-input"
          name="message"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          maxRows={10}
          placeholder="向帕秋莉提问，或输入 / 唤出指令..."
          disabled={disabled}
          className={cn(
            'flex-1 resize-none bg-transparent outline-none p-2',
            'text-foreground placeholder:text-muted-foreground',
            'min-h-[40px]'
          )}
        />

        <button
          type="submit"
          disabled={!message.trim() || disabled}
          className={cn(
            'p-2 rounded-lg transition-all duration-200',
            'flex items-center justify-center',
            message.trim() && !disabled
              ? 'bg-primary hover:bg-primary/80 text-primary-foreground cursor-pointer'
              : 'bg-muted text-muted-foreground cursor-not-allowed'
          )}
          aria-label="Send message"
        >
          <Send className="w-4 h-4" />
        </button>
      </form>

      {/* Hint */}
      <div className="mt-2 text-xs text-muted-foreground">
        <kbd className="px-1.5 py-0.5 rounded bg-black/20 border border-white/10">Enter</kbd> 发送
        <span className="mx-2">•</span>
        <kbd className="px-1.5 py-0.5 rounded bg-black/20 border border-white/10">Shift + Enter</kbd> 换行
      </div>
    </div>
  );
}
