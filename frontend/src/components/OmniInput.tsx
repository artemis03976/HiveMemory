import { useState } from 'react';
import TextareaAutosize from 'react-textarea-autosize';
import { Send, Paperclip, Hash } from 'lucide-react';

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
    <div className="mx-auto max-w-3xl mb-4 w-full px-4"> 
      {/* 核心输入框本体 */}
      <div className="glass-input rounded-2xl p-3 flex flex-col gap-2">
        
        {/* 1. 输入区放在最上面，彻底无边框 */}
        <TextareaAutosize
          className="w-full resize-none bg-transparent outline-none text-[15px] text-foreground placeholder:text-muted-foreground/75 leading-relaxed px-1"
          placeholder="向帕秋莉提问，或输入 / 唤出指令..."
          minRows={1}
          maxRows={8}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
        />
        
        {/* 2. 底部工具栏与发送按钮并排 */}
        <div className="flex items-center justify-between pt-1">
          {/* 左侧工具：附件、记忆引用 */}
          <div className="flex items-center gap-1 text-muted-foreground/75">
            <button className="p-2 hover:text-foreground hover:bg-white/5 rounded-lg transition-colors">
              <Paperclip className="w-4 h-4" />
            </button>
            <button className="p-2 hover:text-foreground hover:bg-white/5 rounded-lg transition-colors">
              <Hash className="w-4 h-4" />
            </button>
          </div>
          
          {/* 右侧：发送按钮 (有输入内容时点亮为主色调) */}
          <button
            onClick={handleSubmit}
            disabled={!message.trim() || disabled}
            className="p-2 rounded-xl bg-primary/20 text-primary hover:bg-primary hover:text-white transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* 3. 快捷键提示移到输入框外面，极度淡化 */}
      <div className="text-center mt-3 text-[11px] text-muted-foreground/60 font-medium tracking-wide">
        <span className="bg-white/5 px-1.5 py-0.5 rounded text-[10px] mr-1">Enter</span> 发送 
        <span className="mx-2">•</span> 
        <span className="bg-white/5 px-1.5 py-0.5 rounded text-[10px] mr-1">Shift + Enter</span> 换行
      </div>
    </div>
  );
}
