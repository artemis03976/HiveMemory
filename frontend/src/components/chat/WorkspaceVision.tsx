import { useEffect } from 'react';
import { X, ChevronLeft, Database, Brain, ListChecks } from 'lucide-react';
import type { MemoryAtom } from '@/types';
import MemoryRuntimeTab from './MemoryRuntimeTab';
import ReferencedMemoryTab from './ReferencedMemoryTab';
import { useChatUiStore, useWorkspaceStore, useMemoryTaskStore } from '@/stores';

interface WorkspaceVisionProps {
  memories: MemoryAtom[];
  isCollapsed: boolean;
  onToggleCollapse: () => void;
  /** 展开时作为抽屉覆盖在中间栏上，而不是占用行内宽度。 */
  overlay?: boolean;
}

export default function WorkspaceVision({
  memories,
  isCollapsed,
  onToggleCollapse,
  overlay = false,
}: WorkspaceVisionProps) {
  const { workspaceVisionTab: activeTab, setWorkspaceVisionTab: setActiveTab } = useChatUiStore();
  const connectRuntimeEvents = useWorkspaceStore((state) => state.connectRuntimeEvents);
  const activeMemoryTaskCount = useMemoryTaskStore((state) =>
    state.tasks.filter((task) => task.status === 'pending' || task.status === 'running').length,
  );
  const tabIndicatorLeft = activeTab === 'context' ? '4px' : '50%';

  useEffect(() => {
    connectRuntimeEvents();
  }, [connectRuntimeEvents]);

  if (isCollapsed) {
    return (
      <aside className="w-12 shrink-0 bg-surface-container-low border-l border-line flex flex-col h-full items-center py-4 transition-all duration-300">
        <button
          onClick={onToggleCollapse}
          aria-label="展开工作区视界"
          className="p-2 text-fg-3 hover:text-primary transition-colors hover:bg-ink/5 rounded-lg"
        >
          <ChevronLeft className="w-5 h-5" />
        </button>
        <div className="mt-8 flex flex-col gap-8 items-center">
          <Database className="w-4 h-4 text-fg-3" />
          <div className="[writing-mode:vertical-rl] text-xs font-bold text-fg-3 select-none">
            工作区视界
          </div>
        </div>
      </aside>
    );
  }

  return (
    <aside
      className={`w-[360px] shrink-0 bg-surface-container-low border-l border-line flex flex-col h-full ${
        overlay ? 'absolute right-0 top-0 z-40 shadow-pop' : ''
      }`}
    >
      <div className="p-6 pb-2 border-b border-line flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <h3 className="font-manrope font-bold text-sm text-primary">工作区视界</h3>
          <button
            onClick={onToggleCollapse}
            aria-label="收起工作区视界"
            className="rounded-md p-1 text-fg-3 transition-colors hover:bg-ink/5 hover:text-fg"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
        
        {/* Tabs 切换栏 */}
        <div className="flex w-full bg-ink/5 p-1 rounded-full border border-line backdrop-blur-md relative">
          <div
            className="absolute top-1 bottom-1 w-[calc(50%-4px)] bg-primary/20 rounded-full transition-all duration-300 ease-[cubic-bezier(0.23,1,0.32,1)]"
            style={{ left: tabIndicatorLeft }}
          />
          <button 
            onClick={() => setActiveTab('context')}
            className={`relative z-10 flex-1 flex items-center justify-center gap-1.5 py-1.5 px-2 text-xs font-bold tracking-widest rounded-full transition-all duration-300 ${
              activeTab === 'context' ? 'text-primary' : 'text-fg-3 hover:text-fg'
            }`}
          >
            <Brain className="w-3.5 h-3.5" /> 引用记忆
          </button>
          <button
            onClick={() => setActiveTab('memory-runtime')}
            className={`relative z-10 flex-1 flex items-center justify-center gap-1.5 py-1.5 px-2 text-xs font-bold tracking-widest rounded-full transition-all duration-300 ${
              activeTab === 'memory-runtime' ? 'text-primary' : 'text-fg-3 hover:text-fg'
            }`}
            aria-label={`记忆任务${activeMemoryTaskCount > 0 ? `，${activeMemoryTaskCount} 个进行中` : ''}`}
          >
            <ListChecks className="w-3.5 h-3.5" />
            <span>记忆任务</span>
            {activeMemoryTaskCount > 0 && (
              <span className="absolute -right-0.5 -top-1 inline-flex h-4 min-w-4 items-center justify-center rounded-full border border-magic-water/30 bg-surface-container-highest px-1 text-xs font-bold leading-none tracking-normal text-magic-water">
                {activeMemoryTaskCount > 99 ? '99+' : activeMemoryTaskCount}
              </span>
            )}
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 scrollbar-hide">
        {activeTab === 'context' && <ReferencedMemoryTab memories={memories} />}
        {activeTab === 'memory-runtime' && <MemoryRuntimeTab />}
      </div>
    </aside>
  );
}
