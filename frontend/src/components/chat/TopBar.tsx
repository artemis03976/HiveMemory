import { useChatStore } from '@/stores';

/** 会话标题与轮数来自会话资源，不再用 Topic token 数表示对话上下文。 */
export default function TopBar() {
  const { sessions, currentSessionId } = useChatStore();
  const session = sessions.find(item => item.session_id === currentSessionId);
  const turns = session?.turn_count ?? 0;
  return (
    <div className="flex min-h-16 shrink-0 flex-col justify-center gap-0.5 border-b border-line px-6 py-3">
      <h2 className="truncate font-manrope text-lg font-bold text-fg">{session?.title || '新会话'}</h2>
      <span className="text-xs text-fg-3">{turns > 0 ? `${turns} 轮` : '还没有消息'}</span>
    </div>
  );
}
