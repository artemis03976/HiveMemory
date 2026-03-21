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
    <div className="liquid-glass glass-sidebar h-screen flex flex-col relative z-40">
      {/* Tabs Header - 悬浮胶囊风格 (无边框分割) */}
      <div className="p-4 pb-2">
        <div className="relative flex p-1 bg-black/40 backdrop-blur-md rounded-2xl border border-white/5 shadow-inner">
          {/* 滑动背景指示器 */}
          <div 
            className="absolute top-1 bottom-1 w-[calc(50%-4px)] bg-gradient-to-r from-amber-500/20 to-amber-400/10 border border-amber-500/30 rounded-xl shadow-[0_0_15px_rgba(245,158,11,0.2)] transition-all duration-300 ease-[cubic-bezier(0.23,1,0.32,1)]"
            style={{ 
              left: activeTab === 'topics' ? '4px' : 'calc(50% + 0px)'
            }}
          />
          <TabButton active={activeTab === 'topics'} onClick={() => setActiveTab('topics')} icon={Folder} label="话题" />
          <TabButton active={activeTab === 'config'} onClick={() => setActiveTab('config')} icon={Settings} label="配置" />
        </div>
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto custom-scrollbar pt-2">
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
        'relative flex-1 flex items-center justify-center gap-2 px-4 py-2.5 z-10',
        'text-sm font-medium transition-colors duration-300',
        active
          ? 'text-amber-400'
          : 'text-muted-foreground hover:text-foreground'
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
    active: 'bg-magic-wood shadow-[0_0_8px_rgba(16,185,129,0.5)]',
    dormant: 'bg-magic-metal',
    swapped: 'bg-slate-600',
  };

  return (
    <div
      className={cn(
        // 基础样式：去掉所有的 blur 和 glass 类名！保持相对定位和过渡动画
        'relative p-3 rounded-xl cursor-pointer group overflow-hidden',
        'transition-all duration-200 ease-out border',
        
        // 【核心改造：选中态 vs 未选中态】
        topic.status === 'active' 
          ?[
              // 选中态：实体黑底 + 星云紫渐变 + 明显边框 + 魔法发光
              'bg-[#120f18]', // 实心暗紫黑色打底，彻底隔绝底层模糊背景
              'bg-gradient-to-r from-primary/30 via-primary/5 to-transparent', // 叠加鲜艳的魔法渐变
              'border-primary/40', // 清晰的实体边框
              'shadow-[0_4px_20px_rgba(139,92,246,0.2)]' // 强烈的外部紫色光晕
            ]
          :[
              // 未选中态：完全透明 (Ghost)，无边框
              'bg-transparent',
              'border-transparent',
              // Hover 时仅叠加一层淡淡的纯白色，不加 blur
              'hover:bg-white/5 hover:border-white/10'
            ]
      )}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* 选中态的左侧霓虹发光指示条 (保持不变，这是点睛之笔) */}
      {topic.status === 'active' && (
        <div className="absolute left-0 top-1/2 -translate-y-1/2 h-4/5 w-[3px] rounded-r-full bg-primary shadow-[0_0_10px_rgba(139,92,246,0.9)]" />
      )}

      <div className="flex items-start justify-between gap-2 relative z-10">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5">
            {/* 状态指示灯 */}
            <div className={cn('w-2 h-2 rounded-full transition-all duration-500 shrink-0', statusColors[topic.status], topic.status === 'active' && 'animate-pulse')} />
            {/* 标题 */}
            <h3 className={cn(
              "text-sm font-semibold truncate transition-colors", 
              topic.status === 'active' ? 'text-white' : 'text-foreground/80 group-hover:text-white/95'
            )}>
              {topic.title}
            </h3>
          </div>
          {/* 摘要 */}
          <p className={cn(
            "text-xs line-clamp-2 leading-relaxed transition-colors",
            topic.status === 'active' ? 'text-white/70' : 'text-muted-foreground/60'
          )}>
            {topic.summary}
          </p>
          {/* Meta 信息 */}
          <div className="flex items-center gap-2 mt-2.5 text-[11px] font-medium transition-colors"
               style={{ color: topic.status === 'active' ? 'rgba(255,255,255,0.4)' : 'var(--color-muted-foreground)' }}>
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
