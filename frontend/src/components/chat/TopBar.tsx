import { useChatStore } from '@/stores';

/** 会话标题与轮数来自会话资源，不再用 Topic token 数表示对话上下文。 */
export default function TopBar() {
  const { sessions, currentSessionId } = useChatStore();
  const session = sessions.find(item => item.session_id === currentSessionId);
  return <div className="h-14 flex items-center px-8 border-b border-white/5 justify-between bg-surface-dim shrink-0">
    <h2 className="font-manrope font-semibold text-lg truncate">{session?.title || '新会话'}</h2>
    <span className="text-xs text-slate-500">{session?.turn_count ?? 0} 轮</span>
  </div>;
}
