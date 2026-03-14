import { useState } from 'react';
import { Folder, Settings, Archive, Trash2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { Topic } from '@/types';

const mockTopics: Topic[] = [
  {
    id: 't1',
    title: '前端架构讨论',
    summary: '讨论 HiveMemory 的 UI/UX 实现细节与七曜魔法主题...',
    status: 'active',
    lastActive: Date.now(),
    messageCount: 15,
  },
  {
    id: 't2',
    title: '后端 API 重构',
    summary: '重构 MTP 协议处理程序与异步接口...',
    status: 'dormant',
    lastActive: Date.now() - 3600000,
    messageCount: 8,
  },
];

type TabType = 'topics' | 'config';

export function ContextSidebar() {
  const [activeTab, setActiveTab] = useState<TabType>('topics');

  return (
    <div className="glass-sidebar h-screen flex flex-col relative z-40">
      {/* Tabs Header - 胶囊风格 */}
      <div className="p-3 border-b border-white/5">
        <div className="flex p-1 bg-black/20 backdrop-blur-sm rounded-xl border border-white/5">
          <TabButton active={activeTab === 'topics'} onClick={() => setActiveTab('topics')} icon={Folder} label="话题" />
          <TabButton active={activeTab === 'config'} onClick={() => setActiveTab('config')} icon={Settings} label="配置" />
        </div>
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        {activeTab === 'topics' && <TopicsTab topics={mockTopics} />}
        {activeTab === 'config' && <ConfigTab />}
      </div>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  icon: Icon,
  label,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ElementType;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'flex-1 flex items-center justify-center gap-2 px-4 py-3',
        'text-sm font-medium transition-all duration-200',
        active
          ? 'text-white bg-white/15 shadow-sm border border-white/10'
          : 'text-muted-foreground hover:text-white hover:bg-white/5 border border-transparent'
      )}
    >
      <Icon className="w-4 h-4" />
      {label}
    </button>
  );
}

function TopicsTab({ topics }: { topics: Topic[] }) {
  return (
    <div className="p-3 space-y-3">
      {topics.map((topic) => (
        <TopicCard key={topic.id} topic={topic} />
      ))}
    </div>
  );
}

function TopicCard({ topic }: { topic: Topic }) {
  const [isHovered, setIsHovered] = useState(false);

  const statusColors = {
    active: 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]',
    dormant: 'bg-amber-500',
    swapped: 'bg-slate-600',
  };

  return (
    <div
      className={cn(
        'glass-card p-3 rounded-lg cursor-pointer group overflow-hidden',
        'transition-all duration-300 ease-out border',
         topic.status === 'active' 
          ? 'bg-linear-to-r from-primary/20 to-transparent border-primary/30 shadow-[0_4px_20px_rgba(139,92,246,0.1)]' 
          : 'bg-white/5 border-white/5 hover:bg-white/10 hover:border-white/10'
      )}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {topic.status === 'active' && (
        <div className="absolute left-0 top-2 bottom-2 w-[3px] rounded-r-full bg-primary shadow-[0_0_8px_rgba(139,92,246,0.8)]" />
      )}

      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5">
            {/* 状态指示灯 */}
            <div className={cn('w-2 h-2 rounded-full transition-all duration-500 shrink-0', statusColors[topic.status], topic.status === 'active' && 'animate-pulse')} />
            {/* 标题 */}
            <h3 className={cn(
              "text-sm font-semibold truncate transition-colors", 
              topic.status === 'active' ? 'text-white/95' : 'text-foreground/85 group-hover:text-white'
            )}>
              {topic.title}
            </h3>
          </div>
          {/* 摘要 */}
          <p className="text-xs text-muted-foreground/80 line-clamp-2 leading-relaxed">
            {topic.summary}
          </p>
          {/* Meta 信息 */}
          <div className="flex items-center gap-2 mt-2.5 text-[11px] text-muted-foreground/60 font-medium">
            <span>{topic.messageCount} msgs</span>
            <span>•</span>
            <span>{formatTime(topic.lastActive)}</span>
          </div>
        </div>

        {/* Action buttons on hover */}
        <div className={cn(
          "flex gap-1 transition-all duration-200",
          isHovered ? "opacity-100 translate-x-0" : "opacity-0 translate-x-2 pointer-events-none"
        )}>
          <button className="p-1.5 rounded-md hover:bg-white/10 hover:text-white text-muted-foreground transition-colors" aria-label="Archive topic">
            <Archive className="w-3.5 h-3.5" />
          </button>
          <button className="p-1.5 rounded-md hover:bg-magic-fire/20 text-muted-foreground hover:text-magic-fire transition-colors" aria-label="Delete topic">
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}

function ConfigTab() {
  return (
    <div className="p-4 space-y-4">
      <div>
        <label htmlFor="model-select" className="text-sm font-medium text-foreground mb-2 block">
          模型 (Model)
        </label>
        {/* 使用更精致的玻璃输入框 */}
        <select className="w-full glass-input px-3 py-2.5 rounded-xl text-sm text-foreground appearance-none cursor-pointer">
          <option>GPT-4o</option>
          <option>Claude 3.5 Sonnet</option>
          <option>Gemini Pro</option>
        </select>
      </div>

      <div>
        <label htmlFor="temperature-slider" className="text-sm font-medium text-foreground mb-2 block">
          温度 (Temperature): 0.7
        </label>
        <input
          id="temperature-slider"
          name="temperature"
          type="range"
          min="0"
          max="2"
          step="0.1"
          defaultValue="0.7"
          className="w-full"
        />
      </div>

      <div>
        <label htmlFor="max-tokens" className="text-sm font-medium text-foreground mb-2 block">
          最大 Token 数 (Max Tokens)
        </label>
        <input
          id="max-tokens"
          name="maxTokens"
          type="number"
          defaultValue="4096"
          className="w-full glass-input px-3 py-2 rounded-lg text-sm"
        />
      </div>
    </div>
  );
}

function formatTime(timestamp: number): string {
  const diff = Date.now() - timestamp;
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);

  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}
