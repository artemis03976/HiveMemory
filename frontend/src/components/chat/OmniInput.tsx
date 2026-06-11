"use client";

import { useState, useRef, useEffect } from 'react';
import { Paperclip, Hash, Send, Square, BrainCircuit, ChevronDown } from 'lucide-react';
import { useChatStore } from '@/stores/chatStore';
import { useChatRuntimeConfigStore } from '@/stores/chatRuntimeConfigStore';
import { useChatUiStore } from '@/stores/chatUiStore';
import { Toggle } from '../common/FormControls';
import { useAgents } from '@/hooks/useAgents';
import { motion, AnimatePresence } from 'motion/react';

export default function OmniInput() {
  const [message, setMessage] = useState('');

  const { enableMemory, setEnableMemory } = useChatUiStore();
  const [isAgentMenuOpen, setIsAgentMenuOpen] = useState(false);
  const [mentionQuery, setMentionQuery] = useState<string | null>(null);
  const [mentionIndex, setMentionIndex] = useState(0);

  const { sendMessage, stopStreaming, isStreaming, runStatus, currentAgentId, setCurrentAgentId } = useChatStore();

  const generationOptions = useChatRuntimeConfigStore((state) => state.generationOptions);
  const { agents, loading: agentsLoading, fetchError: agentsFetchError } = useAgents();

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  const currentAgent = agents.find(a => a.id === currentAgentId) || agents[0];

  const filteredAgents = agents.filter(a =>
    mentionQuery === null || a.name.toLowerCase().includes(mentionQuery.toLowerCase())
  );

  // 若本地持久化的 agent_id 不在后端返回列表中，自动回退到 omni_doll
  useEffect(() => {
    if (agentsLoading) return;
    const validAgentIds = new Set(agents.map((a) => a.id));
    const shouldFallback =
      currentAgentId !== 'omni_doll'
      && (agentsFetchError || agents.length === 0 || !validAgentIds.has(currentAgentId));

    if (shouldFallback) {
      console.warn('[AgentDebug][OmniInput] invalid persisted currentAgentId, fallback to omni_doll', {
        currentAgentId,
      });
      setCurrentAgentId('omni_doll');
    }
  }, [agents, agentsLoading, agentsFetchError, currentAgentId, setCurrentAgentId]);

  // Handle outside click for agent menu
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setIsAgentMenuOpen(false);
      }
    };
    if (isAgentMenuOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isAgentMenuOpen]);

  const handleMessageChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value;
    setMessage(val);

    // simple @ mention detection
    const cursorPos = e.target.selectionStart;
    const textBeforeCursor = val.slice(0, cursorPos);
    const lastAtPos = textBeforeCursor.lastIndexOf('@');
    
    if (lastAtPos !== -1) {
      // Check if @ is preceded by space or start of string
      if (lastAtPos === 0 || /\s/.test(textBeforeCursor[lastAtPos - 1])) {
        const query = textBeforeCursor.slice(lastAtPos + 1);
        if (!/\s/.test(query)) { // No space in query
          setMentionQuery(query);
          setMentionIndex(0);
          return;
        }
      }
    }
    setMentionQuery(null);
  };

  const applyMention = (agentId: string) => {
    setCurrentAgentId(agentId);
    
    if (textareaRef.current && mentionQuery !== null) {
      const cursorPos = textareaRef.current.selectionStart;
      const textBeforeCursor = message.slice(0, cursorPos);
      const lastAtPos = textBeforeCursor.lastIndexOf('@');
      
      const newText = message.slice(0, lastAtPos) + message.slice(cursorPos);
      setMessage(newText);
      setMentionQuery(null);
      
      // refocus textarea
      setTimeout(() => {
        if (textareaRef.current) {
          textareaRef.current.focus();
          textareaRef.current.setSelectionRange(lastAtPos, lastAtPos);
        }
      }, 0);
    }
  };

  const handleSend = () => {
    if (message.trim() && !isStreaming) {
      sendMessage(message, {
        enable_memory_retrieval: enableMemory,
        generation_options: generationOptions,
      });
      setMessage('');
    }
  };

  const isCancelling = runStatus === 'cancelling';
  const isFinalizing = runStatus === 'finalizing';
  const inputPlaceholder = isCancelling
    ? '正在取消生成...'
    : isFinalizing
      ? '正在整理本轮结果...'
      : isStreaming
        ? '正在思考...'
        : '向智能体提问... (输入 @ 呼叫特定角色)';
  const stopTitle = isCancelling
    ? '正在请求停止'
    : isFinalizing
      ? '正在收尾，可继续请求停止'
      : '停止生成';

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (mentionQuery !== null) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (filteredAgents.length > 0) {
          setMentionIndex((prev) => (prev + 1) % filteredAgents.length);
        }
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (filteredAgents.length > 0) {
          setMentionIndex((prev) => (prev - 1 + filteredAgents.length) % filteredAgents.length);
        }
        return;
      }
      if (e.key === 'Enter') {
        e.preventDefault();
        if (filteredAgents[mentionIndex]) {
          applyMention(filteredAgents[mentionIndex].id);
        }
        return;
      }
      if (e.key === 'Escape') {
        setMentionQuery(null);
        return;
      }
    }

    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="p-6 pb-10 w-full max-w-4xl mx-auto shrink-0 relative">
      
      {/* Agent Capsule */}
      <div className="mb-2 relative" ref={menuRef}>
        <button 
          onClick={() => setIsAgentMenuOpen(!isAgentMenuOpen)}
          className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-surface-container-high border border-white/10 hover:border-primary/50 hover:bg-white/5 transition-all text-xs font-medium text-slate-300 shadow-lg"
        >
          <span className="text-slate-500">当前对话:</span>
          {currentAgent && <currentAgent.avatarIcon className={`w-3.5 h-3.5 ${currentAgent.colorClass}`} />}
          <span>{currentAgent?.name ?? '...'}</span>
          <ChevronDown className={`w-3 h-3 text-slate-500 transition-transform ${isAgentMenuOpen ? 'rotate-180' : ''}`} />
        </button>

        <AnimatePresence>
          {isAgentMenuOpen && (
            <motion.div
              initial={{ opacity: 0, y: 10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 10, scale: 0.95 }}
              transition={{ duration: 0.15 }}
              className="absolute bottom-full left-0 mb-2 w-64 bg-surface-container-highest border border-white/10 rounded-xl shadow-2xl overflow-hidden z-50 origin-bottom-left"
            >
              <div className="p-1">
                {agents.length > 0 ? (
                  agents.map((agent) => (
                    <button
                      key={agent.id}
                      onClick={() => {
                        setCurrentAgentId(agent.id);
                        setIsAgentMenuOpen(false);
                      }}
                      className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-colors ${
                        currentAgentId === agent.id ? 'bg-primary/20 text-white' : 'hover:bg-white/5 text-slate-300'
                      }`}
                    >
                      <div className={`p-1.5 rounded-md ${currentAgentId === agent.id ? 'bg-primary/20' : 'bg-white/5'}`}>
                        <agent.avatarIcon className={`w-4 h-4 ${agent.colorClass}`} />
                      </div>
                      <div className="flex-1 overflow-hidden">
                        <div className="text-sm font-medium truncate">{agent.name}</div>
                        <div className="text-xs text-slate-500 truncate mt-0.5">{agent.description}</div>
                      </div>
                    </button>
                  ))
                ) : (
                  <div className="px-3 py-4 text-center text-sm text-slate-400">
                    当前还没有自定义Agent
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="glass-panel ghost-border rounded-2xl p-2 flex flex-col shadow-[0_20px_50px_rgba(0,0,0,0.5)] transition-all duration-300 focus-within:shadow-[0_0_5px_rgba(149,71,247,0.4),0_12px_40px_rgba(0,0,0,0.5)] focus-within:border-primary/50 focus-within:ring-1 focus-within:ring-primary/20 relative">
        
        {/* @ Mention Popup */}
        <AnimatePresence>
          {mentionQuery !== null && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              className="absolute bottom-full left-4 mb-2 w-64 bg-surface-container-highest border border-white/10 rounded-xl shadow-2xl overflow-hidden z-50 origin-bottom-left"
            >
              <div className="px-3 py-2 border-b border-white/10 text-xs font-medium text-slate-400 bg-black/20 flex items-center gap-2">
                <span>呼叫智能体...</span>
                {mentionQuery && <span className="text-primary truncate">@{mentionQuery}</span>}
              </div>
              <div className="p-1 max-h-[240px] overflow-y-auto scrollbar-hide">
                {agents.length === 0 ? (
                  <div className="px-3 py-4 text-center text-sm text-slate-400">
                    当前还没有自定义Agent
                  </div>
                ) : filteredAgents.length > 0 ? (
                  filteredAgents.map((agent, idx) => (
                    <button
                      key={agent.id}
                      onClick={() => applyMention(agent.id)}
                      className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-colors ${
                        mentionIndex === idx ? 'bg-primary/20 text-white' : 'hover:bg-white/5 text-slate-300'
                      }`}
                    >
                      <div className={`p-1.5 rounded-md ${mentionIndex === idx ? 'bg-primary/20' : 'bg-white/5'}`}>
                        <agent.avatarIcon className={`w-4 h-4 ${agent.colorClass}`} />
                      </div>
                      <div className="flex-1 overflow-hidden">
                        <div className="text-sm font-medium truncate">{agent.name}</div>
                        <div className="text-xs text-slate-500 truncate mt-0.5">{agent.description}</div>
                      </div>
                    </button>
                  ))
                ) : (
                  <div className="px-3 py-4 text-center text-sm text-slate-400">
                    未找到匹配的Agent
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* 输入区 */}
        <textarea
          ref={textareaRef}
          className="w-full bg-transparent border-none focus:ring-0 text-sm py-3 px-4 resize-none placeholder-slate-500 outline-none"
          placeholder={inputPlaceholder}
          rows={1}
          value={message}
          onChange={handleMessageChange}
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

          {/* 右侧：发送按钮 / 停止按钮 */}
          {isStreaming ? (
            <button
              onClick={stopStreaming}
              disabled={isCancelling}
              className={`p-2.5 rounded-xl flex items-center justify-center transition-all duration-300 active:scale-95 ${
                isCancelling
                  ? 'bg-magic-metal/20 text-magic-metal cursor-wait animate-pulse shadow-[0_0_10px_rgba(245,158,11,0.25)]'
                  : 'bg-red-500/20 text-red-400 hover:bg-red-500/30 hover:text-red-300 shadow-[0_0_10px_rgba(239,68,68,0.3)]'
              }`}
              title={stopTitle}
              aria-label={stopTitle}
            >
              <Square className="w-4 h-4" />
            </button>
          ) : (
            <button
              disabled={!message.trim()}
              onClick={handleSend}
              className={`p-2.5 rounded-xl flex items-center justify-center transition-all duration-300 active:scale-95 ${
                message.trim()
                  ? 'bg-primary-dim text-on-primary-container shadow-[0_0_10px_rgba(149,71,247,0.4)] hover:shadow-[0_0_10px_rgba(149,71,247,0.6)] hover:bg-primary/90'
                  : 'bg-white/5 text-slate-500 hover:bg-white/10 hover:text-slate-300'
              }`}
            >
              <Send className="w-4 h-4" />
            </button>
          )}
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
