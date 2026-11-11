import { useState } from 'react';
import { Check, PanelLeftClose, Pencil, Plus, Trash2, X } from 'lucide-react';
import { useChatStore } from '@/stores';
import type { SessionSummary } from '@/services/sessionApi';
import { dayGroup, formatListTime, type DayGroup } from '@/utils/dateTime';
import { useNow } from '@/hooks/useNow';
import { ConfirmDialog } from '../common/ConfirmDialog';

interface ContextSidebarProps {
  /** 窗口放不下会话栏时，它作为抽屉覆盖在中间栏上。 */
  overlay: boolean;
  /** 行内时收起会话栏，抽屉时关闭抽屉。 */
  onHide: () => void;
}

const DAY_GROUPS: DayGroup[] = ['今天', '昨天', '更早'];
/** 相对时间与日期分组按这个间隔刷新。 */
const CLOCK_TICK_MS = 60_000;

const iconButtonCls =
  'flex h-[26px] w-[26px] items-center justify-center rounded-[7px] text-fg-3 transition-colors hover:bg-surface-container-high hover:text-fg disabled:cursor-not-allowed disabled:opacity-40';

function SessionRow({ session, now, active, busy, onRename, onDelete }: {
  session: SessionSummary;
  now: number;
  active: boolean;
  busy: boolean;
  onRename: () => void;
  onDelete: () => void;
}) {
  const selectSession = useChatStore((s) => s.selectSession);
  const running = session.active_process_id !== null;
  return (
    <div className={`group relative flex items-center rounded-[10px] ${active ? 'bg-primary/15' : 'hover:bg-surface-container-high'}`}>
      {active && <span aria-hidden="true" className="absolute bottom-[9px] left-0 top-[9px] w-[3px] rounded-full bg-brand" />}
      <button
        disabled={busy}
        onClick={() => void selectSession(session.session_id)}
        aria-current={active ? 'page' : undefined}
        className="flex min-w-0 flex-1 flex-col gap-0.5 py-2 pl-3 pr-1.5 text-left disabled:cursor-not-allowed"
      >
        <span className={`truncate text-[13px] text-fg ${active ? 'font-semibold' : 'font-medium'}`}>{session.title || '新会话'}</span>
        <span className="flex items-center gap-1.5 text-xs text-fg-3">
          {formatListTime(Date.parse(session.updated_at), now)} · {session.turn_count} 轮
          {running && (
            <span className="inline-flex items-center gap-1 text-magic-water">
              <span aria-hidden="true" className="h-1.5 w-1.5 animate-pulse rounded-full bg-magic-water" />
              运行中
            </span>
          )}
        </span>
      </button>
      <span className="flex gap-0.5 pr-1.5 opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100">
        <button disabled={busy} onClick={onRename} aria-label="重命名会话" title="重命名会话" className={iconButtonCls}>
          <Pencil className="h-[13px] w-[13px]" />
        </button>
        <button disabled={busy || running} onClick={onDelete} aria-label="删除会话" title={running ? '会话运行中，结束后才能删除' : '删除会话'} className={iconButtonCls}>
          <Trash2 className="h-[13px] w-[13px]" />
        </button>
      </span>
    </div>
  );
}

function RenameRow({ initial, busy, onSave, onCancel }: {
  initial: string;
  busy: boolean;
  onSave: (title: string) => void;
  onCancel: () => void;
}) {
  const [title, setTitle] = useState(initial);
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        onSave(title.trim());
      }}
      className="flex items-center gap-1 rounded-[10px] bg-surface-container-high p-1.5"
    >
      <input
        aria-label="会话标题"
        autoFocus
        maxLength={200}
        value={title}
        onChange={(event) => setTitle(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Escape') onCancel();
        }}
        className="h-8 min-w-0 flex-1 rounded-lg border border-primary/40 bg-field px-2 text-[13px] text-fg focus:outline-none"
      />
      <button type="submit" disabled={busy} aria-label="保存标题" className={iconButtonCls}>
        <Check className="h-3.5 w-3.5" />
      </button>
      <button type="button" onClick={onCancel} aria-label="取消重命名" className={iconButtonCls}>
        <X className="h-3.5 w-3.5" />
      </button>
    </form>
  );
}

/** 左栏：按日期分组的会话列表。运行和历史读取期间锁定选择，避免消息串入其他会话。 */
export default function ContextSidebar({ overlay, onHide }: ContextSidebarProps) {
  const { sessions, currentSessionId, isStreaming, isLoadingHistory, sessionError,
    newSession, renameSession, deleteSession } = useChatStore();
  const [editing, setEditing] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const busy = isStreaming || isLoadingHistory;
  const now = useNow(CLOCK_TICK_MS);
  const groups = DAY_GROUPS
    .map((label) => ({ label, items: sessions.filter((s) => dayGroup(Date.parse(s.updated_at), now) === label) }))
    .filter((group) => group.items.length > 0);

  return (
    <aside
      aria-label="会话"
      className={`flex h-full w-[264px] shrink-0 flex-col border-r border-line bg-surface-container-low ${
        overlay ? 'absolute left-0 top-0 z-40 shadow-pop' : ''
      }`}
    >
      <div className="flex items-center gap-2 pb-2.5 pl-[18px] pr-3.5 pt-4">
        <h2 className="flex-1 font-manrope text-base font-bold text-fg">会话</h2>
        <button
          disabled={busy}
          onClick={() => void newSession()}
          className="flex h-8 items-center gap-1.5 rounded-[9px] border border-primary/40 bg-primary/15 px-3 text-[13px] font-semibold text-primary transition-colors hover:bg-primary/20 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Plus className="h-3.5 w-3.5" strokeWidth={2.4} />
          新会话
        </button>
        <button onClick={onHide} aria-label={overlay ? '关闭会话栏' : '收起会话栏'} title={overlay ? '关闭会话栏' : '收起会话栏'} className={`${iconButtonCls} h-8 w-8`}>
          {overlay ? <X className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
        </button>
      </div>

      <div className="scrollbar-thin flex-1 overflow-y-auto px-2 pb-4 pt-1">
        {sessionError && <p role="alert" className="px-2.5 pb-1 pt-2 text-xs text-magic-fire">{sessionError}</p>}
        {isLoadingHistory && <p className="px-2.5 pb-1 pt-2 text-xs text-fg-3">正在读取会话…</p>}
        {sessions.length === 0 && !isLoadingHistory && (
          <p className="pt-10 text-center text-[13px] text-fg-3">还没有会话</p>
        )}
        {groups.map((group) => (
          <div key={group.label}>
            <div className="px-2.5 pb-1.5 pt-3 text-xs font-semibold text-fg-3">{group.label}</div>
            {group.items.map((session) =>
              editing === session.session_id ? (
                <RenameRow
                  key={session.session_id}
                  initial={session.title}
                  busy={busy}
                  onSave={(title) => {
                    void renameSession(session.session_id, title);
                    setEditing(null);
                  }}
                  onCancel={() => setEditing(null)}
                />
              ) : (
                <SessionRow
                  key={session.session_id}
                  session={session}
                  now={now}
                  active={currentSessionId === session.session_id}
                  busy={busy}
                  onRename={() => setEditing(session.session_id)}
                  onDelete={() => setDeleting(session.session_id)}
                />
              ),
            )}
          </div>
        ))}
      </div>

      <ConfirmDialog
        isOpen={deleting !== null}
        title="删除会话"
        message="删除后将无法恢复这段会话。已有的记忆材料会保留。"
        onConfirm={() => { if (deleting) void deleteSession(deleting); setDeleting(null); }}
        onCancel={() => setDeleting(null)}
      />
    </aside>
  );
}
