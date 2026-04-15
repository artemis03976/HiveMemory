import { X, ChevronLeft, Database, Terminal, Brain } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import type { MemoryAtom } from '@/types';
import KernelTerminalTab from './KernelTerminalTab';
import { useChatUiStore } from '@/stores/chatUiStore';

interface KernelVisionProps {
  memories: MemoryAtom[];
  isCollapsed: boolean;
  onToggleCollapse: () => void;
}

export default function KernelVision({
  memories,
  isCollapsed,
  onToggleCollapse
}: KernelVisionProps) {
  const { kernelVisionTab: activeTab, setKernelVisionTab: setActiveTab } = useChatUiStore();
  const safeMemories = Array.isArray(memories) ? memories : [];

  if (isCollapsed) {
    return (
      <aside className="w-12 glass-panel border-l border-white/5 flex flex-col h-full items-center py-4 transition-all duration-300">
        <button 
          onClick={onToggleCollapse}
          className="p-2 text-slate-400 hover:text-primary transition-colors hover:bg-white/5 rounded-lg"
        >
          <ChevronLeft className="w-5 h-5" />
        </button>
        <div className="mt-8 flex flex-col gap-8 items-center">
          <Database className="w-4 h-4 text-slate-600" />
          <div className="[writing-mode:vertical-rl] text-[10px] font-bold tracking-[0.2em] uppercase text-slate-500 select-none">
            内核视界
          </div>
        </div>
      </aside>
    );
  }

  return (
    <aside 
      className={`glass-panel border-l border-white/5 flex flex-col h-full transition-all duration-300 ${
        activeTab === 'terminal' ? 'w-[600px]' : 'w-80'
      }`}
    >
      <div className="p-6 pb-2 border-b border-white/5 flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <h3 className="font-manrope font-bold text-sm uppercase tracking-widest text-primary">内核视界</h3>
          <X 
            onClick={onToggleCollapse}
            className="w-4 h-4 text-slate-500 cursor-pointer hover:text-white transition-colors" 
          />
        </div>
        
        {/* Tabs 切换栏 */}
        <div className="flex w-full bg-white/5 p-1 rounded-full ghost-border backdrop-blur-md relative">
          <div
            className="absolute top-1 bottom-1 w-[calc(50%-4px)] bg-primary/20 rounded-full shadow-[0_0_10px_rgba(197,154,255,0.2)] transition-all duration-300 ease-[cubic-bezier(0.23,1,0.32,1)]"
            style={{
              left: activeTab === 'context' ? '4px' : 'calc(50% + 0px)'
            }}
          />
          <button 
            onClick={() => setActiveTab('context')}
            className={`relative z-10 flex-1 flex items-center justify-center gap-1.5 py-1.5 px-3 text-[12px] font-bold tracking-widest rounded-full transition-all duration-300 ${
              activeTab === 'context' ? 'text-primary' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Brain className="w-3.5 h-3.5" /> 引用记忆
          </button>
          <button 
            onClick={() => setActiveTab('terminal')}
            className={`relative z-10 flex-1 flex items-center justify-center gap-1.5 py-1.5 px-3 text-[12px] font-bold tracking-widest rounded-full transition-all duration-300 ${
              activeTab === 'terminal' ? 'text-primary' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Terminal className="w-3.5 h-3.5" /> 内核终端
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 scrollbar-hide">
        {activeTab === 'context' ? (
          <div className="space-y-4">
            <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest px-1">
              当前话题参考记忆 (Top {safeMemories.length})
            </h4>
            {safeMemories.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 gap-3 text-slate-500">
                <Brain className="w-8 h-8 opacity-30" />
                <p className="text-sm">暂无引用记忆</p>
                <p className="text-xs opacity-60">发送消息后，检索到的记忆将显示在此处</p>
              </div>
            ) : (
              <AnimatePresence mode="popLayout">
                {safeMemories.map((memory) => (
                  <motion.div
                    key={memory.id}
                    layout
                    initial={{ opacity: 0, y: -20, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.9, transition: { duration: 0.2 } }}
                    transition={{ duration: 0.4, type: "spring", bounce: 0.2 }}
                    className="mb-4 p-4 rounded-xl bg-surface-container-high ghost-border group hover:bg-surface-container-highest transition-all relative overflow-hidden"
                  >
                    <div className="flex items-start justify-between mb-3 relative z-10">
                      <code className="text-[11px] font-bold text-primary font-mono px-1.5 py-0.5 bg-primary/10 rounded">
                        {memory.alias}
                      </code>
                      {memory.confidence_score && (
                        <span className="text-[10px] text-tertiary font-mono">
                          {(memory.confidence_score * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                    <p className="text-[12px] leading-relaxed text-slate-300 font-inter mb-3 relative z-10">
                      {memory.summary}
                    </p>
                    <div className="flex flex-wrap gap-1.5 relative z-10">
                      {(Array.isArray(memory.tags) ? memory.tags : []).map((tag) => (
                        <span key={tag} className="px-2 py-0.5 rounded-md bg-white/5 text-slate-400 text-[9px] uppercase tracking-wider ghost-border">
                          #{tag}
                        </span>
                      ))}
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            )}
          </div>
        ) : (
          <KernelTerminalTab />
        )}
      </div>
    </aside>
  );
}
