import { useEffect, useState, type ReactNode } from 'react';
import { BookmarkCheck, Bot, Check, ChevronRight, Clock, Copy, Hash, Info, List, MessageSquare, RefreshCw, Trash2 } from 'lucide-react';
import { useChatStore, useChatUiStore, useToastStore, useTopicStore } from '@/stores';
import { fetchConfig } from '@/services/configApi';
import { readPath } from '@/constants/settingsSchema';
import { useNow } from '@/hooks/useNow';
import { formatDateTime, formatFullDateTime } from '@/utils/dateTime';
import { formatTimeAgo } from '@/utils/timeAgo';
import { formatTokens, minutesUntilAutoSettle } from '@/utils/topicPool';
import { ConfirmDialog } from '../common/ConfirmDialog';
import type { Topic, TopicProgressEntry, TopicTurnExcerpt } from '@/types';

/** 话题池的规则取自主配置；读不到时页面只显示话题本身，不写具体数值。 */
interface PoolSettings {
  capacity: number;
  settleAfterSeconds: number;
  foldTokenThreshold: number;
  foldRetainBlocks: number;
  progressRecordAfterSeconds: number;
}

const POOL_SETTING_PATHS = {
  capacity: 'patchouli.perception.max_resident_topics',
  settleAfterSeconds: 'patchouli.perception.settle_after_seconds',
  foldTokenThreshold: 'patchouli.perception.engine.fold_token_threshold',
  foldRetainBlocks: 'patchouli.perception.engine.fold_retain_recent_blocks',
  progressRecordAfterSeconds: 'patchouli.perception.progress_record_after_seconds',
} as const;

/** 剩余时间在这之内时，自动沉淀以警示色标出。 */
const SETTLE_SOON_MINUTES = 10;
/** 容量格子最多画这么多，更大的容量只写数字。 */
const MAX_SLOT_CELLS = 10;
/** 估算与相对时间按这个间隔刷新。 */
const CLOCK_TICK_MS = 60_000;

function usePoolSettings(): PoolSettings | null {
  const [settings, setSettings] = useState<PoolSettings | null>(null);
  useEffect(() => {
    fetchConfig()
      .then(({ config }) => {
        const values = Object.fromEntries(
          Object.entries(POOL_SETTING_PATHS).map(([key, path]) => [key, readPath(config, path)]),
        );
        if (Object.values(values).every((value) => typeof value === 'number')) setSettings(values as unknown as PoolSettings);
      })
      .catch(() => setSettings(null));
  }, []);
  return settings;
}

function topicTitle(topic: Topic): string {
  return topic.title || '未命名话题';
}

function TopicIcon({ size }: { size: 'sm' | 'lg' }) {
  return (
    <span
      aria-hidden="true"
      className={`flex shrink-0 items-center justify-center bg-accent/15 text-accent ${size === 'lg' ? 'h-11 w-11 rounded-[13px]' : 'h-8 w-8 rounded-[10px]'}`}
    >
      <Hash className={size === 'lg' ? 'h-5 w-5' : 'h-[15px] w-[15px]'} strokeWidth={2.2} />
    </span>
  );
}

function SideCard({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  return (
    <div title={hint} className="flex flex-col gap-2.5 rounded-[14px] border border-line bg-surface-container-low p-3.5">
      <span className="text-xs font-semibold text-fg-3">{title}</span>
      {children}
    </div>
  );
}

function turnRange(entry: TopicProgressEntry): string {
  return entry.fromTurn === entry.toTurn ? `第 ${entry.toTurn} 轮` : `第 ${entry.fromTurn}–${entry.toTurn} 轮`;
}

/** 还没有整理的轮次的节选；后端只交出最近几轮，更早的只写出轮数。 */
export function UnrecordedTurnList({ turns, total, recordAfter }: { turns: TopicTurnExcerpt[]; total: number; recordAfter: string }) {
  const earlier = total - turns.length;
  return (
    <div className="flex flex-col gap-2">
      {earlier > 0 && <span className="text-xs text-fg-3">只列出最近 {turns.length} 轮，更早的 {earlier} 轮没有展开</span>}
      {turns.map((turn) => (
        <div key={turn.turnNumber} className="flex flex-col gap-1.5 rounded-xl border border-line bg-surface-container px-3 py-2.5 text-[13px] leading-relaxed">
          <span className="text-xs font-semibold text-fg-3">第 {turn.turnNumber} 轮</span>
          {turn.user && (
            <span className="flex gap-2.5">
              <span className="w-7 shrink-0 font-semibold text-accent">用户</span>
              <span className="line-clamp-2 min-w-0 whitespace-pre-wrap text-fg">{turn.user}</span>
            </span>
          )}
          {turn.assistant && (
            <span className="flex gap-2.5">
              <span className="w-7 shrink-0 font-semibold text-fg-3">回复</span>
              <span className="line-clamp-2 min-w-0 whitespace-pre-wrap text-fg-2">{turn.assistant}</span>
            </span>
          )}
        </div>
      ))}
      <span className="text-xs leading-relaxed text-fg-3">{recordAfter}，这些轮会整理成新的一条进展记录</span>
    </div>
  );
}

/**
 * 话题的进展：已经写下的进展记录逐条列出，其后是还没有整理的轮数，可展开为最近几轮的节选。
 *
 * 进展记录只往后追加，最后一条覆盖到的位置之后就是还没有整理的原始交互；还没有写过一条时，
 * 话题的进展就是全部原始交互。展开的状态属于这一个话题，切换话题时由调用方以 key 重置。
 */
function ProgressSection({ topic, recordAfterSeconds, now }: { topic: Topic; recordAfterSeconds: number | null; now: number }) {
  const [unrecordedOpen, setUnrecordedOpen] = useState(false);
  const entries = topic.progressRecord;
  const recorded = entries.length > 0 ? entries[entries.length - 1].toTurn : 0;
  const unrecorded = topic.unrecordedTurnCount;
  const totalTurns = recorded + unrecorded;
  const summary = entries.length === 0
    ? `还没有进展记录 · ${unrecorded} 轮没有整理`
    : `${entries.length} 条进展记录 · ${unrecorded === 0 ? `已整理全部 ${totalTurns} 轮` : `已整理 ${recorded} / ${totalTurns} 轮`}`;
  const recordAfter = recordAfterSeconds !== null ? `话题停下 ${Math.round(recordAfterSeconds / 60)} 分钟后` : '话题停下一段时间后';

  return (
    <section aria-label="话题的进展" className="flex flex-col gap-2">
      <div className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
        <span className="text-xs font-semibold text-fg-3">话题的进展</span>
        <span className="flex-1" />
        <span className="text-xs text-fg-3">{summary}</span>
      </div>
      <div className="flex flex-col rounded-[14px] border border-line bg-surface-container-low px-4 pb-1 pt-4">
        {entries.length === 0 && (
          <div className="flex items-start gap-3 pb-3.5">
            <span aria-hidden="true" className="flex h-7 w-7 shrink-0 items-center justify-center rounded-[9px] bg-primary/15 text-primary">
              <Clock className="h-[15px] w-[15px]" />
            </span>
            <span className="flex flex-col gap-1">
              <b className="text-sm font-semibold text-fg">还没有进展记录</b>
              <span className="text-[13px] leading-relaxed text-fg-2">{recordAfter}写下第一条。在那之前，话题的进展就是这些还没有整理的轮次。</span>
            </span>
          </div>
        )}
        {entries.length > 0 && (
          <ol className="flex flex-col">
            {entries.map((entry, index) => {
              const latest = index === entries.length - 1;
              return (
                <li key={entry.toTurn} className="flex gap-3">
                  <span aria-hidden="true" className="flex w-3 shrink-0 flex-col items-center">
                    <span className={`mt-1.5 h-2.5 w-2.5 rounded-full ${latest ? 'bg-accent shadow-[0_0_10px_rgba(255,156,198,0.55)]' : 'bg-primary shadow-[0_0_8px_rgba(157,107,255,0.35)]'}`} />
                    <span className={`mt-[5px] w-0.5 flex-1 rounded-full ${latest && unrecorded === 0 ? 'bg-transparent' : 'bg-line-strong'}`} />
                  </span>
                  <div className="flex min-w-0 flex-1 flex-col gap-1.5 pb-4">
                    <span className="flex flex-wrap items-center gap-x-2.5 gap-y-1.5 text-xs text-fg-3">
                      <span className="inline-flex h-[22px] items-center rounded-full bg-primary/15 px-2.5 font-semibold text-primary">{turnRange(entry)}</span>
                      <span>覆盖 {entry.toTurn - entry.fromTurn + 1} 轮</span>
                      <span className="flex-1" />
                      <span title={formatFullDateTime(entry.writtenAt)}>{formatDateTime(entry.writtenAt, now)} 写下</span>
                    </span>
                    <p className="whitespace-pre-wrap text-sm leading-7 text-fg">{entry.text}</p>
                  </div>
                </li>
              );
            })}
          </ol>
        )}
        {unrecorded > 0 ? (
          <div className="flex gap-3 pb-3">
            <span aria-hidden="true" className="flex w-3 shrink-0 justify-center">
              <span className="mt-[11px] box-border h-2.5 w-2.5 rounded-full border-2 border-dashed border-accent" />
            </span>
            <div className="flex min-w-0 flex-1 flex-col gap-2.5">
              <button
                type="button"
                onClick={() => setUnrecordedOpen(!unrecordedOpen)}
                aria-expanded={unrecordedOpen}
                className="inline-flex h-8 items-center gap-1.5 self-start rounded-[9px] border border-line-strong bg-surface-container pl-2 pr-3 text-[13px] font-semibold text-accent transition-colors hover:bg-surface-container-high"
              >
                <ChevronRight className={`h-3.5 w-3.5 transition-transform ${unrecordedOpen ? 'rotate-90' : ''}`} strokeWidth={2.4} aria-hidden="true" />
                还有 {unrecorded} 轮没有整理
              </button>
              {unrecordedOpen && <UnrecordedTurnList turns={topic.unrecordedTurns} total={unrecorded} recordAfter={recordAfter} />}
            </div>
          </div>
        ) : (
          <span className="flex items-center gap-1.5 pb-3.5 pl-6 text-xs text-fg-3">
            <Check className="h-[13px] w-[13px] text-magic-wood" aria-hidden="true" />
            全部 {totalTurns} 轮都已整理
          </span>
        )}
      </div>
    </section>
  );
}

export function TopicRow({ topic, selected, settleMinutes, now, onSelect }: {
  topic: Topic;
  selected: boolean;
  settleMinutes: number | null;
  now: number;
  onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      aria-current={selected ? 'true' : undefined}
      className={`relative flex items-start gap-3 rounded-[14px] border p-3.5 pl-4 text-left wrap-break-word transition-colors ${
        selected ? 'border-primary/40 bg-primary/15' : 'border-line bg-surface-container hover:border-line-strong'
      }`}
    >
      {selected && <span aria-hidden="true" className="absolute -left-px bottom-3.5 top-3.5 w-[3px] rounded-full bg-brand" />}
      <TopicIcon size="sm" />
      <span className="flex min-w-0 flex-1 flex-col gap-1">
        <span className="truncate text-[15px] font-semibold text-fg">{topicTitle(topic)}</span>
        {(topic.foldSummary || topic.topicSummary) && (
          <span className="line-clamp-2 text-[13px] leading-normal text-fg-2">{topic.foldSummary || topic.topicSummary}</span>
        )}
        <span className="mt-0.5 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-xs text-fg-3">
          <span>{topic.blockCount} 个片段</span>
          <span>{formatTokens(topic.totalTokens)} tokens</span>
          <span>{formatTimeAgo(topic.lastActive, now)}写入</span>
          {topic.progressRecord.length > 0 && (
            <span title="这个话题已经写下的进展记录条数" className="inline-flex h-5 items-center gap-1 rounded-full bg-accent/15 px-[7px] font-semibold text-accent">
              <List className="h-[11px] w-[11px]" strokeWidth={2.4} aria-hidden="true" />
              {topic.progressRecord.length} 条进展
            </span>
          )}
          {settleMinutes !== null && settleMinutes <= SETTLE_SOON_MINUTES && (
            <span className="inline-flex items-center gap-1.5 font-semibold text-magic-metal">
              <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-magic-metal" />
              {settleMinutes > 0 ? `约 ${settleMinutes} 分钟后沉淀` : '即将沉淀'}
            </span>
          )}
        </span>
      </span>
    </button>
  );
}

export function TopicDetail({ topic, settings, settleMinutes, now, onReference, onSettle, onDiscard }: {
  topic: Topic;
  settings: PoolSettings | null;
  settleMinutes: number | null;
  now: number;
  onReference: () => void;
  onSettle: () => void;
  onDiscard: () => void;
}) {
  const addToast = useToastStore((s) => s.addToast);
  const shortId = topic.id.length > 12 ? `${topic.id.slice(0, 4)}…${topic.id.slice(-4)}` : topic.id;
  const soon = settleMinutes !== null && settleMinutes <= SETTLE_SOON_MINUTES;
  const hasProgress = topic.progressRecord.length > 0 || topic.unrecordedTurnCount > 0;
  const hasContent = topic.topicSummary || hasProgress || topic.foldSummary || topic.lastTurn;

  return (
    <section aria-label={topicTitle(topic)} className="flex min-w-0 flex-1 flex-col overflow-hidden wrap-break-word rounded-[18px] border border-line bg-surface-container shadow-pop">
      <div className="flex flex-wrap items-start gap-3.5 border-b border-line px-5.5 pb-4 pt-5">
        <TopicIcon size="lg" />
        <div className="flex min-w-[200px] flex-1 flex-col gap-2">
          <h2 className="font-manrope text-[22px] font-extrabold leading-snug text-fg">{topicTitle(topic)}</h2>
          <div className="flex flex-wrap items-center gap-2 text-xs text-fg-3">
            <button
              onClick={() => {
                navigator.clipboard.writeText(topic.id)
                  .then(() => addToast('话题 ID 已复制', 'success'))
                  .catch(() => addToast('复制失败', 'error'));
              }}
              aria-label="复制话题 ID"
              title={topic.id}
              className="inline-flex h-6 items-center gap-1.5 rounded-[7px] border border-line bg-surface-container-low px-2 text-fg-2 transition-colors hover:bg-surface-container-high"
            >
              <span className="font-mono text-xs">{shortId}</span>
              <Copy className="h-3 w-3" aria-hidden="true" />
            </button>
            <span>
              {topic.blockCount} 个片段 · {formatTokens(topic.totalTokens)} tokens{topic.model ? ` · ${topic.model}` : ''}
            </span>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <button
            onClick={onReference}
            className="inline-flex h-9 items-center gap-1.5 rounded-[10px] border border-line-strong bg-surface-container-low px-3 text-[13px] font-semibold text-fg transition-colors hover:bg-surface-container-high"
          >
            <MessageSquare className="h-[15px] w-[15px]" aria-hidden="true" />
            在对话中引用
          </button>
          <button
            onClick={onSettle}
            className="inline-flex h-9 items-center gap-1.5 rounded-[10px] bg-brand px-3.5 text-[13px] font-semibold text-white shadow-glow transition-transform active:scale-95"
          >
            <BookmarkCheck className="h-[15px] w-[15px]" aria-hidden="true" />
            结算为长期记忆
          </button>
          <span aria-hidden="true" className="mx-0.5 h-[22px] w-px bg-line-strong" />
          <button
            onClick={onDiscard}
            className="inline-flex h-9 items-center gap-1.5 rounded-[10px] border border-line-strong px-3 text-[13px] font-semibold text-fg-2 transition-colors hover:border-transparent hover:bg-magic-fire/12 hover:text-magic-fire"
          >
            <Trash2 className="h-[15px] w-[15px]" aria-hidden="true" />
            丢弃
          </button>
        </div>
      </div>

      {/* 窄窗口下信息卡排在正文之后，与正文一起滚动 */}
      <div className="scrollbar-thin flex min-h-0 flex-1 flex-col overflow-y-auto xl:flex-row xl:overflow-hidden">
        <div className="scrollbar-thin flex min-w-0 flex-col gap-4.5 pb-5.5 pl-5.5 pr-4.5 pt-4.5 xl:flex-1 xl:overflow-y-auto">
          {!hasContent && <p className="text-[13px] text-fg-3">这个话题还没有摘要。</p>}
          {topic.topicSummary && (
            <div className="flex flex-col gap-2">
              <span className="text-xs font-semibold text-fg-3">在聊什么</span>
              <p className="text-[15px] leading-relaxed text-fg">{topic.topicSummary}</p>
            </div>
          )}
          {hasProgress && <ProgressSection key={topic.id} topic={topic} recordAfterSeconds={settings?.progressRecordAfterSeconds ?? null} now={now} />}
          {topic.foldSummary && (
            <div className="flex flex-col gap-2">
              <span className="text-xs font-semibold text-fg-3">较早内容的摘要</span>
              <p className="rounded-[14px] border border-line bg-surface-container-low px-4 py-3.5 text-sm leading-7 text-fg-2">{topic.foldSummary}</p>
            </div>
          )}
          {topic.lastTurn && (
            <div className="flex flex-col gap-2.5">
              <span className="text-xs font-semibold text-fg-3">最近一轮</span>
              {topic.lastTurn.user && (
                <div className="max-w-[78%] self-end whitespace-pre-wrap rounded-[14px] rounded-br-[4px] bg-bubble px-3.5 py-2.5 text-sm leading-relaxed text-fg">
                  {topic.lastTurn.user}
                </div>
              )}
              {topic.lastTurn.assistant && (
                <div className="flex items-start gap-2.5">
                  <span aria-hidden="true" className="flex h-7 w-7 shrink-0 items-center justify-center rounded-[9px] bg-surface-container-high text-fg-2">
                    <Bot className="h-[15px] w-[15px]" />
                  </span>
                  <p className="mt-0.5 line-clamp-4 whitespace-pre-wrap text-sm leading-7 text-fg-2">{topic.lastTurn.assistant}</p>
                </div>
              )}
            </div>
          )}
        </div>

        <aside aria-label="话题信息" className="flex shrink-0 flex-col gap-3 px-5.5 pb-5.5 xl:w-[288px] xl:pl-1 xl:pt-4.5">
          <SideCard
            title="上下文"
            hint={settings ? `超过折叠阈值后，较早的片段会折叠成摘要，只保留最近 ${settings.foldRetainBlocks} 个片段的原文` : undefined}
          >
            <span className="flex items-baseline gap-1.5">
              <b className="font-manrope text-xl font-extrabold text-fg">{formatTokens(topic.totalTokens)}</b>
              <span className="text-[13px] text-fg-3">{settings ? `/ ${formatTokens(settings.foldTokenThreshold)} tokens` : 'tokens'}</span>
            </span>
            {settings && (
              <>
                <span aria-hidden="true" className="h-1.5 overflow-hidden rounded-full bg-surface-container-high">
                  <span
                    className="block h-full rounded-full bg-primary"
                    style={{ width: `${Math.min(100, (topic.totalTokens / settings.foldTokenThreshold) * 100)}%` }}
                  />
                </span>
                <span className="text-xs leading-relaxed text-fg-3">超过 {formatTokens(settings.foldTokenThreshold)} 后，较早的片段折叠成摘要</span>
              </>
            )}
          </SideCard>
          {settings && settleMinutes !== null && (
            <SideCard title="自动沉淀" hint="按最近一次写入估算；话题再次写入后重新计时">
              <span className={`flex items-center gap-1.5 text-sm font-semibold ${soon ? 'text-magic-metal' : 'text-fg'}`}>
                <span aria-hidden="true" className={`h-[7px] w-[7px] rounded-full ${soon ? 'bg-magic-metal' : 'bg-fg-3'}`} />
                {settleMinutes > 0 ? `约 ${settleMinutes} 分钟后` : '即将沉淀'}
              </span>
              <span className="text-xs leading-relaxed text-fg-3">
                话题停下 {Math.round(settings.settleAfterSeconds / 60)} 分钟后，新内容沉淀为长期记忆，话题仍留在池中；继续在这个话题里对话会重新计时
              </span>
            </SideCard>
          )}
          <SideCard title="信息">
            <span className="flex gap-2 text-[13px]"><span className="flex-1 text-fg-2">片段</span><span className="font-mono text-xs text-fg">{topic.blockCount}</span></span>
            {topic.model && (
              <span className="flex gap-2 text-[13px]"><span className="flex-1 text-fg-2">模型</span><span className="font-mono text-xs text-fg">{topic.model}</span></span>
            )}
            <span className="flex gap-2 text-[13px]">
              <span className="flex-1 text-fg-2">最近写入</span>
              <span title={formatFullDateTime(topic.lastActive)} className="text-fg">
                {formatDateTime(topic.lastActive, now)} · {formatTimeAgo(topic.lastActive, now)}
              </span>
            </span>
          </SideCard>
        </aside>
      </div>
    </section>
  );
}

/** 记忆图书馆的短期话题池：话题列表与选中话题的详情，可结算、丢弃或在对话中引用。 */
export default function TopicPool({ tabs }: { tabs: ReactNode }) {
  const { topics, isLoading, error, selectedTopicId, fetchTopics, selectTopic, settleTopic, deleteTopic } = useTopicStore();
  const setDesignatedTopic = useChatStore((s) => s.setDesignatedTopic);
  const setActiveNavTab = useChatUiStore((s) => s.setActiveNavTab);
  const addToast = useToastStore((s) => s.addToast);
  const settings = usePoolSettings();
  const now = useNow(CLOCK_TICK_MS);
  const [discarding, setDiscarding] = useState<Topic | null>(null);

  const selected = topics.find((t) => t.id === selectedTopicId) ?? topics[0];
  const settleMinutes = (topic: Topic) =>
    settings ? minutesUntilAutoSettle(topic, settings.settleAfterSeconds, now) : null;

  const settle = async (topic: Topic) => {
    try {
      await settleTopic(topic.id);
      addToast(`已结算“${topicTitle(topic)}”，有还没沉淀的内容时，生成进度在对话页右栏的“记忆任务”中查看`, 'success');
    } catch (e) {
      addToast(`结算失败：${e instanceof Error ? e.message : '未知错误'}`, 'error');
    }
  };
  const discard = async (topic: Topic) => {
    try {
      await deleteTopic(topic.id);
      addToast(`已丢弃“${topicTitle(topic)}”`, 'success');
    } catch (e) {
      addToast(`丢弃失败：${e instanceof Error ? e.message : '未知错误'}`, 'error');
    }
  };
  const reference = (topic: Topic) => {
    setDesignatedTopic({ id: topic.id, title: topicTitle(topic) });
    setActiveNavTab('chat');
  };

  const slots = settings && settings.capacity <= MAX_SLOT_CELLS ? Array.from({ length: settings.capacity }, (_, i) => i < topics.length) : null;

  return (
    <>
      <div className="flex flex-wrap items-center gap-3.5 border-b border-line bg-surface-container-low/50 px-8 py-4">
        {tabs}
        {settings && (
          <div
            role="group"
            aria-label="话题池容量"
            title="同时驻留的话题上限，在高级设置中调整"
            className="flex h-[38px] items-center gap-2 rounded-[11px] border border-line bg-surface-container-low px-3"
          >
            {slots && (
              <span aria-hidden="true" className="flex gap-[3px]">
                {slots.map((filled, i) => (
                  <span key={i} className={`h-2 w-3.5 rounded-[3px] ${filled ? 'bg-primary' : 'bg-surface-container-high'}`} />
                ))}
              </span>
            )}
            <span className="text-[13px] text-fg-2">驻留 <b className="font-semibold text-fg">{topics.length}</b> / {settings.capacity}</span>
          </div>
        )}
        <span className="flex min-w-0 flex-1 items-center gap-2 text-[13px] text-fg-3">
          <Info className="h-[15px] w-[15px] shrink-0" aria-hidden="true" />
          <span>
            {settings ? `话题停下 ${Math.round(settings.settleAfterSeconds / 60)} 分钟时` : '话题停下一段时间后'}，新内容会沉淀为长期记忆，话题仍留在池中；池满需要腾出位置时，最久没用的话题先沉淀再退出
          </span>
        </span>
        <button
          onClick={() => void fetchTopics()}
          disabled={isLoading}
          aria-label="刷新话题池"
          title="刷新"
          className="flex h-[38px] w-[38px] items-center justify-center rounded-[11px] border border-line bg-surface-container-low text-fg-2 transition-colors hover:bg-surface-container-high hover:text-fg disabled:cursor-wait"
        >
          <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {topics.length === 0 ? (
        <div className="flex flex-1 items-center justify-center px-8 pb-16 pt-6">
          {isLoading ? (
            <p className="text-[13px] text-fg-3">正在读取话题…</p>
          ) : error ? (
            <p className="text-[13px] text-fg-3">{error}</p>
          ) : (
            <div className="flex max-w-[460px] flex-col items-center gap-3.5 text-center">
              <span aria-hidden="true" className="flex h-14 w-14 items-center justify-center rounded-2xl bg-accent/15 text-accent">
                <Hash className="h-[26px] w-[26px]" />
              </span>
              <h2 className="font-manrope text-[22px] font-extrabold text-fg">短期话题池是空的</h2>
              <p className="leading-7 text-fg-2">
                对话开始后，相关的几轮会聚成一个话题暂存在这里。话题停下一段时间后，帕秋莉把新的内容沉淀为长期记忆，话题仍留在池中；池满或手动结算时，话题沉淀后离开话题池。
              </p>
              <button
                onClick={() => setActiveNavTab('chat')}
                className="h-[38px] rounded-[11px] border border-line-strong bg-surface-container px-4 font-semibold text-fg transition-colors hover:bg-surface-container-high"
              >
                去对话
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 gap-4 px-8 pb-7 pt-5">
          <div aria-label="短期话题" className="scrollbar-thin flex w-[300px] shrink-0 flex-col gap-2 overflow-y-auto pr-1 xl:w-[400px]">
            {topics.map((topic) => (
              <TopicRow
                key={topic.id}
                topic={topic}
                selected={topic.id === selected?.id}
                settleMinutes={settleMinutes(topic)}
                now={now}
                onSelect={() => selectTopic(topic.id)}
              />
            ))}
          </div>
          {selected && (
            <TopicDetail
              topic={selected}
              settings={settings}
              settleMinutes={settleMinutes(selected)}
              now={now}
              onReference={() => reference(selected)}
              onSettle={() => void settle(selected)}
              onDiscard={() => setDiscarding(selected)}
            />
          )}
        </div>
      )}

      <ConfirmDialog
        isOpen={discarding !== null}
        title={discarding ? `丢弃“${topicTitle(discarding)}”？` : ''}
        message={discarding ? `话题中的 ${discarding.blockCount} 个片段会被删除，无法恢复，其中还没有沉淀的内容不会再提炼为长期记忆。已经沉淀的内容照常生成记忆，对话中已经写入的记忆也不受影响。` : ''}
        confirmText="丢弃话题"
        onConfirm={() => { if (discarding) void discard(discarding); }}
        onCancel={() => setDiscarding(null)}
      />
    </>
  );
}
