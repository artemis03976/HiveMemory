import { useEffect } from 'react';
import { X } from 'lucide-react';
import type { MemoryAtom } from '@/types';
import MemoryRuntimeTab from './MemoryRuntimeTab';
import ModelConfigTab from './ModelConfigTab';
import ReferencedMemoryTab from './ReferencedMemoryTab';
import { useChatUiStore, useWorkspaceStore, useMemoryTaskStore } from '@/stores';
import type { WorkspaceVisionTab } from '@/stores/ui/appUiStore';

interface WorkspaceVisionProps {
  memories: MemoryAtom[];
  /** 窗口放不下右栏时，它作为抽屉覆盖在中间栏上，并带关闭按钮。 */
  onClose?: () => void;
}

/** 右栏：本轮引用的记忆、记忆生成任务与本会话的模型配置。 */
export default function WorkspaceVision({ memories, onClose }: WorkspaceVisionProps) {
  const { workspaceVisionTab: activeTab, setWorkspaceVisionTab: setActiveTab } = useChatUiStore();
  const connectRuntimeEvents = useWorkspaceStore((state) => state.connectRuntimeEvents);
  const activeTaskCount = useMemoryTaskStore((state) =>
    state.tasks.filter((task) => task.status === 'pending' || task.status === 'running').length,
  );

  useEffect(() => {
    connectRuntimeEvents();
  }, [connectRuntimeEvents]);

  const tabs: { id: WorkspaceVisionTab; label: string }[] = [
    { id: 'context', label: memories.length > 0 ? `引用记忆 · ${memories.length}` : '引用记忆' },
    { id: 'memory-runtime', label: activeTaskCount > 0 ? `记忆任务 · ${activeTaskCount}` : '记忆任务' },
    { id: 'model', label: '模型配置' },
  ];

  return (
    <aside
      aria-label="本轮上下文"
      className={`flex h-full w-[360px] shrink-0 flex-col border-l border-line bg-surface-container-low ${
        onClose ? 'absolute right-0 top-0 z-40 shadow-pop' : ''
      }`}
    >
      <div className="flex items-center gap-2 px-3.5 pb-2.5 pt-3.5">
        <div role="tablist" aria-label="右栏" className="flex flex-1 gap-0.5 rounded-[11px] border border-line bg-field p-[3px]">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              role="tab"
              aria-selected={activeTab === tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`h-[30px] flex-1 whitespace-nowrap rounded-lg text-[13px] font-semibold transition-colors ${
                activeTab === tab.id ? 'bg-surface-container-high text-fg' : 'text-fg-3 hover:text-fg'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
        {onClose && (
          <button
            onClick={onClose}
            aria-label="关闭右栏"
            className="flex h-8 w-8 items-center justify-center rounded-[9px] text-fg-3 transition-colors hover:bg-surface-container-high hover:text-fg"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      <div className="scrollbar-thin flex min-h-0 flex-1 flex-col overflow-y-auto px-3.5 pb-[18px] pt-1">
        {activeTab === 'context' && <ReferencedMemoryTab memories={memories} />}
        {activeTab === 'memory-runtime' && <MemoryRuntimeTab />}
        {activeTab === 'model' && <ModelConfigTab />}
      </div>
    </aside>
  );
}
