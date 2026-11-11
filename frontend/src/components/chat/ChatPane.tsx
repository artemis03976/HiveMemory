import { Fragment, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { ArrowDown, Bot, FileUp } from 'lucide-react';
import TopBar from './TopBar';
import ChatMessage from './ChatMessage';
import OmniInput from './OmniInput';
import SessionCompactionBoundary from './SessionCompactionBoundary';
import { useChatStore, useChatUiStore, useMemoryViewStore, useToastStore, useTopicStore } from '@/stores';
import { useAgents, type AgentProfile } from '@/hooks/useAgents';
import { useTopicTitles } from '@/hooks/useTopicTitles';
import { recordMemoryFeedback } from '@/services/memoryApi';
import { useAttachmentUpload } from '@/hooks/useAttachmentUpload';
import { isNearBottom } from '@/utils/scrollFollow';

const EXAMPLE_PROMPTS = [
  { title: '回顾上周的改动', prompt: '总结一下上周对生成链路做的修改' },
  { title: '查一条记忆', prompt: 'Qdrant 升级前要备份哪些东西？' },
  { title: '记下一个结论', prompt: '把这次讨论的结论写进记忆' },
];

/** 空会话的欢迎区：当前 Agent、记忆预检索的状态与几条示例问题。 */
function EmptySession({ agents }: { agents: AgentProfile[] }) {
  const { currentAgentId, setDraft } = useChatStore();
  const enableMemory = useChatUiStore((s) => s.enableMemory);
  const agentName = agents.find((a) => a.id === currentAgentId)?.name ?? currentAgentId;

  return (
    <div className="flex h-full items-center justify-center p-6">
      <div className="flex w-full max-w-[620px] flex-col items-center gap-4 text-center">
        <div aria-hidden="true" className="flex h-16 w-16 items-center justify-center rounded-[20px] bg-brand text-white shadow-glow">
          <Bot className="h-8 w-8" />
        </div>
        <h2 className="font-manrope text-2xl font-extrabold text-fg">今天要请人偶帮忙做什么？</h2>
        <p className="leading-relaxed text-fg-2">
          当前是<b className="font-semibold text-fg">{agentName}</b>。
          {enableMemory
            ? '记忆预检索已开启：发送之前，先从记忆图书馆中检索与这条消息相关的记忆。'
            : '记忆预检索已关闭：这一轮不会先检索记忆。'}
        </p>
        <div className="mt-1.5 grid w-full grid-cols-1 gap-2.5 sm:grid-cols-3">
          {EXAMPLE_PROMPTS.map((example) => (
            <button
              key={example.title}
              onClick={() => setDraft(example.prompt)}
              className="flex flex-col gap-1.5 rounded-2xl border border-line-strong bg-surface-container p-3.5 text-left transition-colors hover:border-primary/40"
            >
              <span className="font-semibold text-fg">{example.title}</span>
              <span className="text-[13px] leading-normal text-fg-3">{example.prompt}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

interface ChatPaneProps {
  /** 会话栏收起或收为抽屉时，头部给出展开它的按钮。 */
  onOpenSessions?: () => void;
  /** 右栏收起或收为抽屉时，头部给出展开它的按钮。 */
  onOpenVision?: () => void;
}

export default function ChatPane({ onOpenSessions, onOpenVision }: ChatPaneProps) {
  const { messages, compactionCheckpoint, compactionProgress, runStatus, isLoadingHistory, currentSessionId, isStreaming, retrievedMemories, retryMessage } = useChatStore();
  const addToast = useToastStore((s) => s.addToast);
  const { agents } = useAgents();
  const topicTitles = useTopicTitles(messages);
  const selectTopic = useTopicStore((s) => s.selectTopic);
  const setLibraryTab = useMemoryViewStore((s) => s.setLibraryTab);
  const setActiveNavTab = useChatUiStore((s) => s.setActiveNavTab);
  const openTopic = (topicId: string) => {
    selectTopic(topicId);
    setLibraryTab('topics');
    setActiveNavTab('database');
  };
  const lastReplyId = [...messages].reverse().find((m) => m.role === 'agent' || m.role === 'assistant')?.id;

  const copyReply = (text: string) => {
    navigator.clipboard.writeText(text)
      .then(() => addToast('已复制', 'success'))
      .catch(() => addToast('复制失败', 'error'));
  };

  // 预检索命中只随流式事件下发、不写入会话记录，所以只有本页完成的最后一轮能对它们统一反馈
  const feedbackOnTurn = (positive: boolean) => {
    Promise.all(retrievedMemories.map((memory) => recordMemoryFeedback(memory.id, positive)))
      .then(() => addToast(`已记录对 ${retrievedMemories.length} 条记忆的反馈`, 'success'))
      .catch(() => addToast('记录反馈失败', 'error'));
  };
  const { enqueueFiles } = useAttachmentUpload();
  const [isDragActive, setIsDragActive] = useState(false);
  // 拖拽进入子元素会连续触发 dragenter/dragleave，用计数判断是否真正离开对话区
  const dragDepthRef = useRef(0);

  // 停在底部时新内容自动跟随；向上翻过之后不再跟随，并记下新增的消息数
  const listRef = useRef<HTMLDivElement>(null);
  const [following, setFollowing] = useState(true);
  const [unseen, setUnseen] = useState(0);
  const seenCountRef = useRef(messages.length);

  const scrollToLatest = () => {
    const list = listRef.current;
    if (list) list.scrollTop = list.scrollHeight;
    setFollowing(true);
    setUnseen(0);
  };

  // 打开会话时定位到最新一轮
  useEffect(() => {
    if (isLoadingHistory) return;
    seenCountRef.current = messages.length;
    scrollToLatest();
    // 只在会话切换或历史加载完成时触发
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSessionId, isLoadingHistory]);

  // 新消息与流式输出：跟随时贴底，否则累计未看到的消息数
  useLayoutEffect(() => {
    const added = messages.length - seenCountRef.current;
    seenCountRef.current = messages.length;
    if (following) {
      const list = listRef.current;
      if (list) list.scrollTop = list.scrollHeight;
    } else if (added > 0) {
      setUnseen((count) => count + added);
    }
  }, [messages, following]);

  const handleScroll = () => {
    const list = listRef.current;
    if (!list) return;
    const atBottom = isNearBottom(list.scrollTop, list.scrollHeight, list.clientHeight);
    if (atBottom !== following) setFollowing(atBottom);
    if (atBottom) setUnseen(0);
  };

  const hasFiles = (e: React.DragEvent) =>
    Array.from(e.dataTransfer?.types ?? []).includes('Files');

  const handleDragEnter = (e: React.DragEvent) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepthRef.current += 1;
    setIsDragActive(true);
  };

  const handleDragOver = (e: React.DragEvent) => {
    if (!hasFiles(e)) return;
    // 必须 preventDefault，否则浏览器会直接打开拖入的文件
    e.preventDefault();
  };

  const handleDragLeave = (e: React.DragEvent) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (dragDepthRef.current === 0) {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepthRef.current = 0;
    setIsDragActive(false);
    // 只接收 DataTransfer.files：目录和无文件 payload 直接忽略
    enqueueFiles(e.dataTransfer.files);
  };

  return (
    <main
      className="relative flex h-full min-w-0 flex-1 flex-col overflow-hidden bg-background bg-atmo"
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <TopBar onOpenSessions={onOpenSessions} onOpenVision={onOpenVision} referenceCount={retrievedMemories.length} />

      {messages.length === 0 ? (
        <div className="flex-1 overflow-y-auto">
          {isLoadingHistory ? (
            <div className="flex h-full items-center justify-center text-sm text-fg-3">少女祈祷中...</div>
          ) : (
            <EmptySession agents={agents} />
          )}
        </div>
      ) : (
        <div ref={listRef} onScroll={handleScroll} className="scrollbar-thin flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-[780px] space-y-7 px-7 pb-8 pt-6">
            {messages.map((msg) => (
              <Fragment key={msg.id}>
                <ChatMessage
                  message={msg}
                  agents={agents}
                  topicTitle={msg.topicId ? topicTitles[msg.topicId] : undefined}
                  onOpenTopic={openTopic}
                  onCopy={copyReply}
                  onRegenerate={msg.id === lastReplyId && !isStreaming ? () => void retryMessage(msg.id) : undefined}
                  onFeedback={msg.id === lastReplyId && !isStreaming && retrievedMemories.length > 0 ? feedbackOnTurn : undefined}
                />
                <SessionCompactionBoundary interactionId={msg.interactionId} checkpoint={compactionCheckpoint} />
              </Fragment>
            ))}
          </div>
        </div>
      )}

      {!following && messages.length > 0 && (
        <div className="pointer-events-none relative z-10 -mt-12 flex justify-center pb-3">
          <button
            onClick={scrollToLatest}
            className="pointer-events-auto flex h-[34px] items-center gap-1.5 rounded-full border border-line-strong bg-surface-container px-3.5 text-[13px] font-semibold text-fg shadow-pop"
          >
            <ArrowDown className="h-3.5 w-3.5" />
            回到最新
            {unseen > 0 && <span className="text-primary">· {unseen} 条新消息</span>}
          </button>
        </div>
      )}

      {compactionProgress?.status === 'started' && runStatus !== 'cancelling' && (
        <div role="status" aria-live="polite" className="px-8 py-3 text-sm text-primary">
          正在压缩较早的上下文… 原始对话会保留
        </div>
      )}
      <OmniInput />

      {/* 拖入文件时的 drop 状态提示；pointer-events-none 保证 drop 命中对话区 */}
      {isDragActive && (
        <div className="pointer-events-none absolute inset-0 z-40 flex items-center justify-center rounded-lg border-2 border-dashed border-primary/50 bg-primary/10 backdrop-blur-sm">
          <div className="flex flex-col items-center gap-2 text-primary">
            <FileUp className="h-10 w-10" />
            <span className="text-sm font-medium">松开以上传附件</span>
          </div>
        </div>
      )}
    </main>
  );
}
