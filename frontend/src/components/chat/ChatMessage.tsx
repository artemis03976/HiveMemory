import { useState } from 'react';
import { Copy, ThumbsUp, ThumbsDown, RefreshCw, BrainCircuit, Hash } from 'lucide-react';
import type { Message, ContentBlock, InlineBlock, SubAgentBlock } from '@/types';
import { motion } from 'motion/react';
import SubAgentCard from './SubAgentCard';
import InlineBlockList from './InlineBlockList';
import { resolveChatAgent, type AgentProfile } from '@/hooks/useAgents';
import { chatFailureMessage } from '@/stores/chat/failureMessage';

interface ChatMessageProps {
  message: Message;
  /** Profile 列表，用来取回复者与子 Agent 的名称和图标。 */
  agents: AgentProfile[];
  /** 消息所在话题的标题；取不到时显示“未命名话题”。 */
  topicTitle?: string;
  /** 最后一条回复才能重新生成。 */
  onRegenerate?: () => void;
  /** 本页知道这一轮预检索命中了哪些记忆时，可以对它们统一反馈。 */
  onFeedback?: (positive: boolean) => void;
  onCopy?: (text: string) => void;
}

type MessageSegment =
  | { kind: 'inline'; blocks: InlineBlock[] }
  | { kind: 'sub_agent'; block: SubAgentBlock };

function getMessageBlocks(message: Message): ContentBlock[] {
  if (message.contentBlocks) {
    return message.contentBlocks;
  }

  if (message.mtpAction) {
    return [
      { kind: 'text', text: message.content },
      { kind: 'mtp', action: message.mtpAction },
    ];
  }

  return [{ kind: 'text', text: message.content }];
}

function buildMessageSegments(blocks: ContentBlock[]): MessageSegment[] {
  const segments: MessageSegment[] = [];
  let currentInline: InlineBlock[] = [];

  const flushInline = () => {
    if (currentInline.length > 0) {
      segments.push({ kind: 'inline', blocks: currentInline });
      currentInline = [];
    }
  };

  for (const block of blocks) {
    if (block.kind === 'sub_agent') {
      flushInline();
      segments.push({ kind: 'sub_agent', block });
      continue;
    }
    currentInline.push(block);
  }

  flushInline();
  return segments;
}

export default function ChatMessage({ message, agents, topicTitle, onRegenerate, onFeedback, onCopy }: ChatMessageProps) {
  const isAgent = message.role === 'agent' || message.role === 'assistant';
  const isUser = message.role === 'user';
  const [feedback, setFeedback] = useState<boolean | null>(null);

  const blocks = getMessageBlocks(message);
  const segments = buildMessageSegments(blocks);
  const lastInlineSegmentIdx = segments.map((segment) => segment.kind).lastIndexOf('inline');
  const hasContent = blocks.some(
    (b) => (b.kind === 'text' && b.text) || b.kind === 'mtp' || b.kind === 'sub_agent',
  );
  const isProcessing = isAgent && message.isStreaming && !hasContent;
  const replyText = blocks.flatMap((b) => (b.kind === 'text' && b.text ? [b.text] : [])).join('\n\n');

  if (isUser) {
    return (
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="flex justify-end">
        <div className="max-w-[78%] whitespace-pre-wrap rounded-2xl rounded-br-md border border-line bg-bubble px-3.5 py-2.5 text-sm leading-relaxed text-fg">
          {message.content}
        </div>
      </motion.div>
    );
  }

  if (!isAgent) return null;
  const agent = resolveChatAgent(agents, message.agent_id);
  const time = message.timestamp
    ? new Date(message.timestamp).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', hour12: false })
    : null;

  const pick = (positive: boolean) => {
    setFeedback(positive);
    onFeedback?.(positive);
  };

  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="group flex flex-col gap-2.5">
      <div className="flex items-center gap-2.5">
        <span aria-hidden="true" className="flex h-7 w-7 items-center justify-center rounded-[9px] bg-primary/15 text-primary">
          <agent.avatarIcon className="h-4 w-4" />
        </span>
        <span className="text-[13px] font-semibold text-fg">{agent.name}</span>
        {message.isStreaming ? (
          <span className="flex items-center gap-1.5 text-xs text-magic-water">
            <span aria-hidden="true" className="h-1.5 w-1.5 animate-pulse rounded-full bg-magic-water" />
            回复中
          </span>
        ) : (
          time && <span className="text-xs text-fg-3">{time}</span>
        )}
      </div>

      {isProcessing && (
        <div className="flex items-center gap-2 text-sm font-medium text-primary">
          <BrainCircuit className="h-4 w-4 animate-pulse" />
          <span>思考中<span className="thinking-dots"></span></span>
        </div>
      )}
      {segments.map((segment, idx) => (
        segment.kind === 'inline' ? (
          <InlineBlockList
            key={idx}
            blocks={segment.blocks}
            isStreaming={message.isStreaming}
            animateLastTextCursor={idx === lastInlineSegmentIdx}
            textClassName="text-sm leading-relaxed text-fg"
          />
        ) : (
          <SubAgentCard key={idx} block={segment.block} agent={resolveChatAgent(agents, segment.block.agentId)} />
        )
      ))}

      {!message.isStreaming && (
        <div className="flex flex-wrap items-center gap-3 text-xs text-fg-3">
          {message.topicId && (
            <span className="flex items-center gap-1" title={topicTitle ? undefined : message.topicId}>
              <Hash className="h-3 w-3" aria-hidden="true" />
              {topicTitle ?? '未命名话题'}
            </span>
          )}
          {message.outcome === 'cancelled' && <span className="text-magic-metal">已中断 · 回复未完成</span>}
          {message.outcome === 'failed' && <span className="text-magic-fire">{chatFailureMessage(message.failureReason)} · 回复可能不完整</span>}
          <span
            className={`ml-auto flex gap-0.5 transition-opacity ${onRegenerate ? '' : 'opacity-0 group-hover:opacity-100 group-focus-within:opacity-100'}`}
          >
            {replyText && (
              <button
                onClick={() => onCopy?.(replyText)}
                aria-label="复制回复"
                title="复制回复"
                className="rounded-lg p-1.5 text-fg-3 transition-colors hover:bg-ink/5 hover:text-fg"
              >
                <Copy className="h-3.5 w-3.5" />
              </button>
            )}
            {onFeedback && (
              <>
                <button
                  onClick={() => pick(true)}
                  aria-pressed={feedback === true}
                  aria-label="本轮命中的记忆有帮助"
                  title="本轮预检索命中的记忆有帮助"
                  className={`rounded-lg p-1.5 transition-colors hover:bg-ink/5 ${feedback === true ? 'bg-primary/15 text-primary' : 'text-fg-3 hover:text-fg'}`}
                >
                  <ThumbsUp className="h-3.5 w-3.5" />
                </button>
                <button
                  onClick={() => pick(false)}
                  aria-pressed={feedback === false}
                  aria-label="本轮命中的记忆不准确"
                  title="本轮预检索命中的记忆不准确"
                  className={`rounded-lg p-1.5 transition-colors hover:bg-ink/5 ${feedback === false ? 'bg-primary/15 text-primary' : 'text-fg-3 hover:text-fg'}`}
                >
                  <ThumbsDown className="h-3.5 w-3.5" />
                </button>
              </>
            )}
            {onRegenerate && (
              <button
                onClick={onRegenerate}
                aria-label="重新生成"
                title="以同一条消息再发送一轮（原附件不重发），原来的一轮保留在历史中"
                className="rounded-lg p-1.5 text-fg-3 transition-colors hover:bg-ink/5 hover:text-fg"
              >
                <RefreshCw className="h-3.5 w-3.5" />
              </button>
            )}
          </span>
        </div>
      )}
    </motion.div>
  );
}
