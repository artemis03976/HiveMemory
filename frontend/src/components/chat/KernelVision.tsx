import { useEffect } from 'react';
import { X, ChevronLeft, Database, Terminal, Brain, ListChecks } from 'lucide-react';
import type { MemoryAtom } from '@/types';
import KernelTerminalTab from './KernelTerminalTab';
import MemoryRuntimeTab from './MemoryRuntimeTab';
import ReferencedMemoryTab from './ReferencedMemoryTab';
import { useChatUiStore, useKernelStore, useMemoryTaskStore } from '@/stores';

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
  const connectRuntimeEvents = useKernelStore((state) => state.connectRuntimeEvents);
  const activeMemoryTaskCount = useMemoryTaskStore((state) =>
    state.tasks.filter((task) => task.status === 'pending' || task.status === 'running').length,
  );
  const panelWidth =
    activeTab === 'terminal' ? 'w-[600px]' : activeTab === 'memory-runtime' ? 'w-[440px]' : 'w-80';
  const tabIndicatorLeft =
    activeTab === 'context' ? '4px' : activeTab === 'memory-runtime' ? 'calc(33.333% + 1px)' : 'calc(66.666% - 2px)';

  useEffect(() => {
    connectRuntimeEvents();
  }, [connectRuntimeEvents]);

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
      className={`glass-panel border-l border-white/5 flex flex-col h-full transition-all duration-300 ${panelWidth}`}
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
            className="absolute top-1 bottom-1 w-[calc(33.333%-4px)] bg-primary/20 rounded-full shadow-[0_0_10px_rgba(197,154,255,0.2)] transition-all duration-300 ease-[cubic-bezier(0.23,1,0.32,1)]"
            style={{ left: tabIndicatorLeft }}
          />
          <button 
            onClick={() => setActiveTab('context')}
            className={`relative z-10 flex-1 flex items-center justify-center gap-1.5 py-1.5 px-2 text-[11px] font-bold tracking-widest rounded-full transition-all duration-300 ${
              activeTab === 'context' ? 'text-primary' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Brain className="w-3.5 h-3.5" /> 引用记忆
          </button>
          <button
            onClick={() => setActiveTab('memory-runtime')}
            className={`relative z-10 flex-1 flex items-center justify-center gap-1.5 py-1.5 px-2 text-[11px] font-bold tracking-widest rounded-full transition-all duration-300 ${
              activeTab === 'memory-runtime' ? 'text-primary' : 'text-slate-400 hover:text-slate-200'
            }`}
            aria-label={`Memory Runtime${activeMemoryTaskCount > 0 ? `, ${activeMemoryTaskCount} active tasks` : ''}`}
          >
            <ListChecks className="w-3.5 h-3.5" />
            <span>Runtime</span>
            {activeMemoryTaskCount > 0 && (
              <span className="absolute -right-0.5 -top-1 inline-flex h-4 min-w-4 items-center justify-center rounded-full border border-magic-water/30 bg-surface-container-highest px-1 text-[9px] font-bold leading-none tracking-normal text-magic-water shadow-[0_0_8px_rgba(117,199,255,0.18)]">
                {activeMemoryTaskCount > 99 ? '99+' : activeMemoryTaskCount}
              </span>
            )}
          </button>
          <button 
            onClick={() => setActiveTab('terminal')}
            className={`relative z-10 flex-1 flex items-center justify-center gap-1.5 py-1.5 px-2 text-[11px] font-bold tracking-widest rounded-full transition-all duration-300 ${
              activeTab === 'terminal' ? 'text-primary' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Terminal className="w-3.5 h-3.5" /> 终端
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 scrollbar-hide">
        {activeTab === 'context' && <ReferencedMemoryTab memories={memories} />}
        {activeTab === 'memory-runtime' && <MemoryRuntimeTab />}
        {activeTab === 'terminal' && <KernelTerminalTab />}
      </div>
    </aside>
  );
}
