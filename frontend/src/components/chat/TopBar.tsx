import { MoreVertical } from 'lucide-react';

interface TopBarProps {
  activeTopicTitle: string;
}

export default function TopBar({ activeTopicTitle }: TopBarProps) {
  return (
    <div className="h-14 flex items-center px-8 border-b border-white/5 justify-between bg-surface-dim shrink-0">
      {/* 活跃话题信息 */}
      <div className="flex items-center gap-3">
        <div className="w-2 h-2 rounded-full bg-primary animate-pulse shadow-[0_0_8px_#c59aff]"></div>
        <h2 className="font-manrope font-semibold text-lg tracking-tight">{activeTopicTitle}</h2>
      </div>
      
      {/* 上下文剩余信息 */}
      <div className="flex items-center gap-4">
        <span className="px-3 py-1 rounded-full bg-secondary-container text-on-secondary-container text-[10px] font-bold tracking-widest uppercase">
          上下文剩余: 84%
        </span>
        <button className="text-slate-400 hover:text-white transition-colors">
          <MoreVertical className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}