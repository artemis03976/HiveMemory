"use client";

import { useState, useRef, useEffect } from 'react';
import { Paperclip, Hash, Send, Square, BrainCircuit, ChevronDown } from 'lucide-react';
import { useChatRuntimeConfigStore, useChatStore, useChatUiStore } from '@/stores';
import { useAttachmentStore } from '@/stores/attachment';
import { buildAttachmentSelection } from '@/stores/attachment/attachmentStore';
import { Toggle } from '../common/FormControls';
import { useAgents } from '@/hooks/useAgents';
import { useAttachmentUpload } from '@/hooks/useAttachmentUpload';
import { ATTACHMENT_ACCEPT_ATTR } from '@/constants/attachments';
import type { ChatModelSelection } from '@/types/chat';
import { motion, AnimatePresence } from 'motion/react';
import AttachmentQueue from './AttachmentQueue';

export default function OmniInput() {
  const { draft: message, setDraft: setMessage, isLoadingHistory, historyReady, currentSessionId } = useChatStore();

  const { enableMemory, setEnableMemory } = useChatUiStore();
  const [isAgentMenuOpen, setIsAgentMenuOpen] = useState(false);
  const [mentionQuery, setMentionQuery] = useState<string | null>(null);
  const [mentionIndex, setMentionIndex] = useState(0);

  const { sendMessage, stopStreaming, isStreaming, runStatus, currentAgentId, setCurrentAgentId } = useChatStore();
  const { enqueueFiles } = useAttachmentUpload();
  const attachmentItems = useAttachmentStore((state) => state.items);
  const selectedAttachmentIds = useAttachmentStore((state) => state.selectedIds);
  const clearAttachmentSelection = useAttachmentStore((state) => state.clearSelection);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const modelSelection = useChatRuntimeConfigStore((state) => state.modelSelection);
  const overrideParams = useChatRuntimeConfigStore((state) => state.overrideParams);
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

  // 输入框高度随内容自适应，超过上限后内部滚动
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [message]);

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
    if (message.trim() && !isStreaming && !isLoadingHistory && (!currentSessionId || historyReady)) {
      // 会话级覆盖按需下发：
      // - model_id 为空 = 跟随 Agent 默认（不下发，避免后端 min_length=1 校验失败）
      // - overrideParams 关闭时不下发 temperature/top_p/max_tokens，跟随 profile/模型定义
      const selection: ChatModelSelection = {};
      if (modelSelection.model_id) {
        selection.model_id = modelSelection.model_id;
      }
      if (overrideParams) {
        selection.temperature = modelSelection.temperature;
        selection.top_p = modelSelection.top_p;
        selection.max_tokens = modelSelection.max_tokens;
      }
      // 发送时冻结附件选择快照：发送后再修改选择不影响已开始的 Chat run
      const attachments = buildAttachmentSelection(attachmentItems, selectedAttachmentIds);
      sendMessage(message, {
        enable_memory_retrieval: enableMemory,
        model_selection: selection,
        attachments,
      });
      setMessage('');
      if (attachments.length > 0) {
        clearAttachmentSelection();
      }
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
      ? '正在收尾，请等待完成'
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

      {/* 附件上传队列（拖拽与 Paperclip 共用同一入队逻辑） */}
      <AttachmentQueue />

      {/* Agent Capsule */}
      <div className="mb-2 relative" ref={menuRef}>
        <button 
          onClick={() => setIsAgentMenuOpen(!isAgentMenuOpen)}
          className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-surface-container-high border border-line hover:border-primary/50 hover:bg-ink/5 transition-all text-xs font-medium text-fg-2 shadow-lg"
        >
          <span className="text-fg-3">当前对话:</span>
          {currentAgent && <currentAgent.avatarIcon className={`w-3.5 h-3.5 text-primary`} />}
          <span>{currentAgent?.name ?? '...'}</span>
          <ChevronDown className={`w-3 h-3 text-fg-3 transition-transform ${isAgentMenuOpen ? 'rotate-180' : ''}`} />
        </button>

        <AnimatePresence>
          {isAgentMenuOpen && (
            <motion.div
              initial={{ opacity: 0, y: 10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 10, scale: 0.95 }}
              transition={{ duration: 0.15 }}
              className="absolute bottom-full left-0 mb-2 w-64 bg-surface-container-highest border border-line rounded-xl shadow-2xl overflow-hidden z-50 origin-bottom-left"
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
                        currentAgentId === agent.id ? 'bg-primary/20 text-fg' : 'hover:bg-ink/5 text-fg-2'
                      }`}
                    >
                      <div className={`p-1.5 rounded-md ${currentAgentId === agent.id ? 'bg-primary/20' : 'bg-ink/5'}`}>
                        <agent.avatarIcon className={`w-4 h-4 text-primary`} />
                      </div>
                      <div className="flex-1 overflow-hidden">
                        <div className="text-sm font-medium truncate">{agent.name}</div>
                        <div className="text-xs text-fg-3 truncate mt-0.5">{agent.description}</div>
                      </div>
                    </button>
                  ))
                ) : (
                  <div className="px-3 py-4 text-center text-sm text-fg-3">
                    当前还没有自定义Agent
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="bg-surface-container-low border border-line rounded-2xl p-2 flex flex-col shadow-pop transition-all duration-300 focus-within:border-primary/50 focus-within:ring-1 focus-within:ring-primary/20 relative">
        
        {/* @ Mention Popup */}
        <AnimatePresence>
          {mentionQuery !== null && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              className="absolute bottom-full left-4 mb-2 w-64 bg-surface-container-highest border border-line rounded-xl shadow-2xl overflow-hidden z-50 origin-bottom-left"
            >
              <div className="px-3 py-2 border-b border-line text-xs font-medium text-fg-3 bg-field flex items-center gap-2">
                <span>呼叫智能体...</span>
                {mentionQuery && <span className="text-primary truncate">@{mentionQuery}</span>}
              </div>
              <div className="p-1 max-h-60 overflow-y-auto scrollbar-hide">
                {agents.length === 0 ? (
                  <div className="px-3 py-4 text-center text-sm text-fg-3">
                    当前还没有自定义Agent
                  </div>
                ) : filteredAgents.length > 0 ? (
                  filteredAgents.map((agent, idx) => (
                    <button
                      key={agent.id}
                      onClick={() => applyMention(agent.id)}
                      className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-colors ${
                        mentionIndex === idx ? 'bg-primary/20 text-fg' : 'hover:bg-ink/5 text-fg-2'
                      }`}
                    >
                      <div className={`p-1.5 rounded-md ${mentionIndex === idx ? 'bg-primary/20' : 'bg-ink/5'}`}>
                        <agent.avatarIcon className={`w-4 h-4 text-primary`} />
                      </div>
                      <div className="flex-1 overflow-hidden">
                        <div className="text-sm font-medium truncate">{agent.name}</div>
                        <div className="text-xs text-fg-3 truncate mt-0.5">{agent.description}</div>
                      </div>
                    </button>
                  ))
                ) : (
                  <div className="px-3 py-4 text-center text-sm text-fg-3">
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
          className="w-full bg-transparent border-none focus:ring-0 text-sm py-3 px-4 resize-none placeholder-fg-3 outline-none scrollbar-hide"
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
            {/* 上传是独立于 Chat SSE run 的资产命令，生成期间仍可继续上传附件 */}
            <button
              onClick={() => fileInputRef.current?.click()}
              className="p-2 text-fg-3 hover:text-fg hover:bg-ink/5 rounded-lg transition-all"
              title="上传附件"
              aria-label="上传附件"
            >
              <Paperclip className="w-4 h-4" />
            </button>
            {/* 隐藏的文件选择入口：accept 为提示而非校验，格式以后端稳定错误为准 */}
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept={ATTACHMENT_ACCEPT_ATTR}
              className="hidden"
              onChange={(e) => {
                enqueueFiles(e.target.files);
                // 允许再次选择同一文件时仍触发 onChange
                e.target.value = '';
              }}
            />
            <button className="p-2 text-fg-3 hover:text-fg hover:bg-ink/5 rounded-lg transition-all" title="话题引用">
              <Hash className="w-4 h-4" />
            </button>

            <div className="w-px h-4 bg-ink/10 mx-2"></div>

            {/* 记忆预检索开关 */}
            <div 
              className={`flex items-center gap-2 px-2 py-1 rounded-lg transition-all cursor-pointer select-none ${isStreaming ? 'opacity-50 cursor-not-allowed' : 'hover:bg-ink/5'}`}
              onClick={() => !isStreaming && setEnableMemory(!enableMemory)}
              title="在对话前进行记忆预检索"
            >
              <BrainCircuit className={`w-4 h-4 transition-colors ${enableMemory ? 'text-primary' : 'text-fg-3'}`} />
              <span className={`text-xs font-medium transition-colors ${enableMemory ? 'text-fg-2' : 'text-fg-3'}`}>
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
              disabled={isCancelling || isFinalizing}
              className={`p-2.5 rounded-xl flex items-center justify-center transition-all duration-300 active:scale-95 ${
                isCancelling
                  ? 'bg-magic-metal/20 text-magic-metal cursor-wait animate-pulse'
                  : 'bg-magic-fire/20 text-magic-fire hover:bg-magic-fire/30 hover:text-magic-fire'
              }`}
              title={stopTitle}
              aria-label={stopTitle}
            >
              <Square className="w-4 h-4" />
            </button>
          ) : (
            <button
              disabled={!message.trim() || isLoadingHistory || (!!currentSessionId && !historyReady)}
              onClick={handleSend}
              className={`p-2.5 rounded-xl flex items-center justify-center transition-all duration-300 active:scale-95 ${
                message.trim()
                  ? 'bg-primary-dim text-on-primary-container hover:bg-primary/90'
                  : 'bg-ink/5 text-fg-3 hover:bg-ink/10 hover:text-fg-2'
              }`}
            >
              <Send className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>
      
      {/* 快捷键提示 */}
      <div className="text-center mt-4 text-xs text-fg-3/60 font-medium tracking-wide">
        <span className="bg-ink/5 px-1.5 py-0.5 rounded text-xs mr-1 border border-line">Enter</span> 发送
        <span className="mx-2">•</span> 
        <span className="bg-ink/5 px-1.5 py-0.5 rounded text-xs mr-1 border border-line">Shift + Enter</span> 换行
      </div>
    </div>
  );
}
