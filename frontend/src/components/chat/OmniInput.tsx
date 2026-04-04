"use client";

import { useState } from 'react';
import { Paperclip, Hash, Send, BrainCircuit } from 'lucide-react';
import { useChatStore } from '@/stores/chatStore';
import { useChatRuntimeConfigStore } from '@/stores/chatRuntimeConfigStore';
import { Toggle } from '../common/FormControls';

export default function OmniInput() {
  const [message, setMessage] = useState('');
  const [enableMemory, setEnableMemory] = useState(true);
  const { sendMessage, isStreaming } = useChatStore();
  const generationOptions = useChatRuntimeConfigStore((state) => state.generationOptions);

  const handleSend = () => {
    if (message.trim() && !isStreaming) {
      sendMessage(message, {
        enable_memory_retrieval: enableMemory,
        generation_options: generationOptions,
      });
      setMessage('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="p-6 pb-10 w-full max-w-4xl mx-auto shrink-0">
      <div className="glass-panel ghost-border rounded-2xl p-2 flex flex-col shadow-[0_20px_50px_rgba(0,0,0,0.5)] transition-all duration-300 focus-within:shadow-[0_0_5px_rgba(149,71,247,0.4),0_12px_40px_rgba(0,0,0,0.5)] focus-within:border-primary/50 focus-within:ring-1 focus-within:ring-primary/20">
        {/* 输入区 */}
        <textarea
          className="w-full bg-transparent border-none focus:ring-0 text-sm py-3 px-4 resize-none placeholder-slate-500 outline-none"
          placeholder={isStreaming ? "正在思考..." : "向智能体提问..."}
          rows={1}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isStreaming}
        ></textarea>

        {/* 底部工具栏，与发送按钮并排 */}
        <div className="flex items-center justify-between px-1 pt-1">
          {/* 左侧工具：附件、记忆引用、记忆检索 */}
          <div className="flex items-center gap-1">
            <button className="p-2 text-slate-400 hover:text-white hover:bg-white/5 rounded-lg transition-all" title="附件" disabled={isStreaming}>
              <Paperclip className="w-4 h-4" />
            </button>
            <button className="p-2 text-slate-400 hover:text-white hover:bg-white/5 rounded-lg transition-all" title="话题引用">
              <Hash className="w-4 h-4" />
            </button>

            <div className="w-px h-4 bg-white/10 mx-2"></div>

            {/* 记忆预检索开关 */}
            <div 
              className={`flex items-center gap-2 px-2 py-1 rounded-lg transition-all cursor-pointer select-none ${isStreaming ? 'opacity-50 cursor-not-allowed' : 'hover:bg-white/5'}`}
              onClick={() => !isStreaming && setEnableMemory(!enableMemory)}
              title="在对话前进行记忆预检索"
            >
              <BrainCircuit className={`w-4 h-4 transition-colors ${enableMemory ? 'text-primary' : 'text-slate-500'}`} />
              <span className={`text-xs font-medium transition-colors ${enableMemory ? 'text-slate-300' : 'text-slate-500'}`}>
                记忆预检索
              </span>
              <div className="scale-[0.65] origin-left flex items-center" onClick={(e) => e.stopPropagation()}>
                <Toggle checked={enableMemory} onChange={setEnableMemory} disabled={isStreaming} />
              </div>
            </div>
          </div>

          {/* 右侧：发送按钮 (有输入内容时点亮为主色调) */}
          <button 
            disabled={!message.trim() || isStreaming}
            onClick={handleSend}
            className={`p-2.5 rounded-xl flex items-center justify-center transition-all duration-300 active:scale-95 ${
              message.trim() && !isStreaming
                ? 'bg-primary-dim text-on-primary-container shadow-[0_0_10px_rgba(149,71,247,0.4)] hover:shadow-[0_0_10px_rgba(149,71,247,0.6)] hover:bg-primary/90' 
                : 'bg-white/5 text-slate-500 hover:bg-white/10 hover:text-slate-300'
            }`}
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
      
      {/* 快捷键提示 */}
      <div className="text-center mt-4 text-[11px] text-slate-500/60 font-medium tracking-wide">
        <span className="bg-white/5 px-1.5 py-0.5 rounded text-[10px] mr-1 border border-white/5">Enter</span> 发送 
        <span className="mx-2">•</span> 
        <span className="bg-white/5 px-1.5 py-0.5 rounded text-[10px] mr-1 border border-white/5">Shift + Enter</span> 换行
      </div>
    </div>
  );
}
