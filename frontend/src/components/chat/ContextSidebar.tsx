import { useState } from 'react';
import { Columns, ChevronRight } from 'lucide-react';
import type { Topic } from '@/types';
import TopicTab from './TopicTab';
import ModelConfigTab from './ModelConfigTab';

interface ContextSidebarProps {
  topics: Topic[];
  activeTopicId: string;
  onTopicSelect: (id: string) => void;
  isCollapsed: boolean;
  onToggleCollapse: () => void;
}

type TabType = 'topics' | 'config';

export default function ContextSidebar({
  topics,
  activeTopicId,
  onTopicSelect,
  isCollapsed,
  onToggleCollapse
}: ContextSidebarProps) {
  const [activeTab, setActiveTab] = useState<TabType>('topics');

  if (isCollapsed) {
    return (
      <aside className="w-12 glass-panel border-r border-white/5 flex flex-col h-full items-center py-4 transition-all duration-300">
        <button
          onClick={onToggleCollapse}
          className="p-2 text-slate-400 hover:text-primary transition-colors hover:bg-white/5 rounded-lg"
        >
          <ChevronRight className="w-5 h-5" />
        </button>
        <div className="mt-8 flex flex-col gap-8 items-center">
          <div className="[writing-mode:vertical-lr] text-[10px] font-bold tracking-[0.2em] uppercase text-slate-500 select-none">
            侧边栏
          </div>
        </div>
      </aside>
    );
  }

  return (
    <aside className="w-72 glass-panel border-r border-white/5 flex flex-col h-full transition-all duration-300">
      <header className="flex items-center justify-between px-6 w-full h-14 bg-surface/10 border-b border-white/5 font-manrope text-sm font-medium tracking-wide">
        <span className="font-black text-lg tracking-tighter text-primary drop-shadow-[0_0_8px_rgba(197,154,255,0.4)]">HiveMemory</span>
        <div className="flex gap-3 text-slate-400">
          <Columns
            onClick={onToggleCollapse}
            className="w-4 h-4 cursor-pointer hover:text-white transition-colors"
          />
        </div>
      </header>

      {/* Tabs Header */}
      <div className="px-4 mt-6">
        <div className="flex w-full bg-white/5 p-1 rounded-full ghost-border backdrop-blur-md relative">
          {/* 滑动背景指示器 */}
          <div
            className="absolute top-1 bottom-1 w-[calc(50%-4px)] bg-primary/20 rounded-full shadow-[0_0_10px_rgba(197,154,255,0.2)] transition-all duration-300 ease-[cubic-bezier(0.23,1,0.32,1)]"
            style={{
              left: activeTab === 'topics' ? '4px' : 'calc(50% + 0px)'
            }}
          />
          
          <button 
            onClick={() => setActiveTab('topics')}
            className={`relative z-10 flex-1 py-1.5 px-3 text-[12px] font-bold tracking-widest rounded-full transition-all duration-300 ${
              activeTab === 'topics' ? 'text-primary' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            话题
          </button>
          <button 
            onClick={() => setActiveTab('config')}
            className={`relative z-10 flex-1 py-1.5 px-3 text-[12px] font-bold tracking-widest rounded-full transition-all duration-300 ${
              activeTab === 'config' ? 'text-primary' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            模型配置
          </button>
        </div>
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto px-2 mt-4 space-y-1 scrollbar-hide pb-6">
        {activeTab === 'topics' ? (
          <TopicTab 
            topics={topics}
            activeTopicId={activeTopicId}
            onTopicSelect={onTopicSelect}
          />
        ) : (
          <ModelConfigTab />
        )}
      </div>
    </aside>
  );
}