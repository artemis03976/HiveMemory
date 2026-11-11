import { PanelLeftOpen, PanelRightOpen } from 'lucide-react';
import { useChatStore } from '@/stores';

interface TopBarProps {
  onOpenSessions?: () => void;
  onOpenVision?: () => void;
  referenceCount: number;
}

const panelButtonCls =
  'flex h-[34px] shrink-0 items-center gap-2 rounded-[10px] border border-line-strong bg-surface-container text-[13px] font-semibold text-fg transition-colors hover:bg-surface-container-high';

/** 会话标题与轮数来自会话资源；两栏收起或收为抽屉时，标题两侧给出展开它们的按钮。 */
export default function TopBar({ onOpenSessions, onOpenVision, referenceCount }: TopBarProps) {
  const { sessions, currentSessionId } = useChatStore();
  const session = sessions.find(item => item.session_id === currentSessionId);
  const turns = session?.turn_count ?? 0;
  return (
    <div className="flex min-h-16 shrink-0 items-center gap-3 border-b border-line px-6 py-3">
      {onOpenSessions && (
        <button onClick={onOpenSessions} aria-label="展开会话栏" title="展开会话栏" className={`${panelButtonCls} w-[34px] justify-center`}>
          <PanelLeftOpen className="h-4 w-4" aria-hidden="true" />
        </button>
      )}
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <h2 className="truncate font-manrope text-lg font-bold text-fg">{session?.title || '新会话'}</h2>
        <span className="text-xs text-fg-3">{turns > 0 ? `${turns} 轮` : '还没有消息'}</span>
      </div>
      {onOpenVision && (
        <button onClick={onOpenVision} title="展开右栏" className={`${panelButtonCls} px-3`}>
          <PanelRightOpen className="h-4 w-4" aria-hidden="true" />
          {referenceCount > 0 ? `引用记忆 · ${referenceCount}` : '引用记忆'}
        </button>
      )}
    </div>
  );
}
