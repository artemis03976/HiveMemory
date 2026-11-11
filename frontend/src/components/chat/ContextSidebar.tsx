import { Columns, ChevronRight } from 'lucide-react';
import SessionTab from './SessionTab';
import ModelConfigTab from './ModelConfigTab';
import { useChatUiStore } from '@/stores';

interface ContextSidebarProps {
  isCollapsed: boolean;
  onToggleCollapse: () => void;
  /** 展开时作为抽屉覆盖在中间栏上，而不是占用行内宽度。 */
  overlay?: boolean;
}

export default function ContextSidebar({
  isCollapsed,
  onToggleCollapse,
  overlay = false,
}: ContextSidebarProps) {
  const { contextSidebarTab: activeTab, setContextSidebarTab: setActiveTab } = useChatUiStore();

  if (isCollapsed) {
    return (
      <aside className="w-12 shrink-0 bg-surface-container-low border-r border-line flex flex-col h-full items-center py-4 transition-all duration-300">
        <button
          onClick={onToggleCollapse}
          aria-label="展开侧边栏"
          className="p-2 text-fg-3 hover:text-primary transition-colors hover:bg-ink/5 rounded-lg"
        >
          <ChevronRight className="w-5 h-5" />
        </button>
        <div className="mt-8 flex flex-col gap-8 items-center">
          <div className="[writing-mode:vertical-lr] text-xs font-bold text-fg-3 select-none">
            侧边栏
          </div>
        </div>
      </aside>
    );
  }

  return (
    <aside className={`w-72 shrink-0 bg-surface-container-low border-r border-line flex flex-col h-full ${
      overlay ? 'absolute left-0 top-0 z-40 shadow-pop' : ''
    }`}>
      <header className="flex items-center justify-between px-6 w-full h-14 bg-surface/10 border-b border-line font-manrope text-sm font-medium tracking-wide">
        <span className="font-black text-lg tracking-tighter text-primary">HiveMemory</span>
        <button
          onClick={onToggleCollapse}
          aria-label="收起侧边栏"
          className="rounded-md p-1 text-fg-3 transition-colors hover:bg-ink/5 hover:text-fg"
        >
          <Columns className="w-4 h-4" />
        </button>
      </header>

      {/* Tabs Header */}
      <div className="px-4 mt-6">
        <div className="flex w-full bg-ink/5 p-1 rounded-full border border-line backdrop-blur-md relative">
          {/* 滑动背景指示器 */}
          <div
            className="absolute top-1 bottom-1 w-[calc(50%-4px)] bg-primary/20 rounded-full transition-all duration-300 ease-[cubic-bezier(0.23,1,0.32,1)]"
            style={{
              left: activeTab === 'sessions' ? '4px' : 'calc(50% + 0px)'
            }}
          />
          
          <button 
            onClick={() => setActiveTab('sessions')}
            className={`relative z-10 flex-1 py-1.5 px-3 text-[12px] font-bold tracking-widest rounded-full transition-all duration-300 ${
              activeTab === 'sessions' ? 'text-primary' : 'text-fg-3 hover:text-fg'
            }`}
          >
            会话
          </button>
          <button 
            onClick={() => setActiveTab('config')}
            className={`relative z-10 flex-1 py-1.5 px-3 text-[12px] font-bold tracking-widest rounded-full transition-all duration-300 ${
              activeTab === 'config' ? 'text-primary' : 'text-fg-3 hover:text-fg'
            }`}
          >
            模型配置
          </button>
        </div>
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto px-2 mt-4 flex flex-col gap-1 scrollbar-hide pb-6">
        {activeTab === 'sessions' ? (
          <SessionTab />
        ) : (
          <ModelConfigTab />
        )}
      </div>
    </aside>
  );
}
