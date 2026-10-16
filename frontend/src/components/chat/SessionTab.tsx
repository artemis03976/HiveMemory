import { useState } from 'react';
import { Plus, Pencil, Trash2, RefreshCw } from 'lucide-react';
import { useChatStore } from '@/stores';
import { ConfirmDialog } from '../common/ConfirmDialog';

/** 运行和历史读取期间锁定选择，避免消息串入其他会话。 */
export default function SessionTab() {
  const { sessions, currentSessionId, isStreaming, isLoadingHistory, sessionError,
    newSession, selectSession, renameSession, deleteSession, refreshSessions } = useChatStore();
  const [editing, setEditing] = useState<string | null>(null);
  const [title, setTitle] = useState('');
  const [deleting, setDeleting] = useState<string | null>(null);
  const busy = isStreaming || isLoadingHistory;
  return <div className="space-y-3 px-2">
    <div className="flex gap-2">
      <button disabled={busy} onClick={() => void newSession()} className="flex-1 flex items-center justify-center gap-2 p-2 rounded-lg bg-primary/15 text-primary disabled:opacity-40"><Plus size={16} />新建会话</button>
      <button title="刷新会话" onClick={() => void refreshSessions()} className="p-2 text-slate-400"><RefreshCw size={16} /></button>
    </div>
    {sessionError && <p role="alert" className="text-xs text-red-300">{sessionError}</p>}
    {isLoadingHistory && <p className="text-xs text-slate-400">正在读取会话…</p>}
    {sessions.length === 0 && <p className="py-8 text-center text-sm text-slate-500">暂无会话</p>}
    {sessions.map(session => <div key={session.session_id} className={`p-3 rounded-xl border ${currentSessionId === session.session_id ? 'border-primary/40 bg-primary/10' : 'border-white/5 bg-white/5'}`}>
      {editing === session.session_id ? <form onSubmit={event => {
        event.preventDefault(); void renameSession(session.session_id, title.trim()); setEditing(null);
      }} className="space-y-2">
        <input aria-label="会话标题" autoFocus maxLength={200} value={title} onChange={e => setTitle(e.target.value)} className="w-full rounded bg-black/20 p-2 text-sm" />
        <button disabled={busy} className="text-xs text-primary mr-3">保存</button>
        <button type="button" onClick={() => setEditing(null)} className="text-xs text-slate-400">取消</button>
      </form> : <>
        <button disabled={busy} onClick={() => void selectSession(session.session_id)} className="w-full text-left disabled:opacity-60">
          <div className="truncate text-sm text-slate-200">{session.title || '新会话'}</div>
          <div className="text-xs text-slate-500 mt-1">{session.turn_count} 轮{session.active_process_id ? ' · 运行中' : ''}</div>
        </button>
        <div className="flex justify-end gap-3 mt-2 text-slate-400">
          <button title="重命名会话" disabled={busy} onClick={() => { setEditing(session.session_id); setTitle(session.title); }}><Pencil size={13} /></button>
          <button title="删除会话" disabled={busy || !!session.active_process_id} onClick={() => setDeleting(session.session_id)}><Trash2 size={13} /></button>
        </div>
      </>}
    </div>)}
    <ConfirmDialog isOpen={deleting !== null} title="删除会话" message="删除后将无法恢复这段会话。已有的记忆材料会保留。" onConfirm={() => { if (deleting) void deleteSession(deleting); setDeleting(null); }} onCancel={() => setDeleting(null)} />
  </div>;
}
