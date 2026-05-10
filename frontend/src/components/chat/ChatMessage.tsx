import { User, Copy, ThumbsUp, RefreshCw, BrainCircuit } from 'lucide-react';
import type { Message, ContentBlock, InlineBlock, SubAgentBlock } from '@/types';
import { motion } from 'motion/react';
import SubAgentCard from './SubAgentCard';
import InlineBlockList from './InlineBlockList';
import { MOCK_AGENTS } from '@/constants/agents';

interface ChatMessageProps {
  message: Message;
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

export default function ChatMessage({ message }: ChatMessageProps) {
  const isAgent = message.role === 'agent' || message.role === 'assistant';
  const isUser = message.role === 'user';

  const agent = isAgent
    ? (MOCK_AGENTS.find(a => a.id === message.agent_id) || MOCK_AGENTS[0])
    : null;

  const blocks = getMessageBlocks(message);
  const segments = buildMessageSegments(blocks);
  const lastInlineSegmentIdx = segments.map((segment) => segment.kind).lastIndexOf('inline');
  const hasContent = blocks.some(
    (b) => (b.kind === 'text' && b.text) || b.kind === 'mtp' || b.kind === 'sub_agent',
  );
  const isProcessing = isAgent && message.isStreaming && !hasContent;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex gap-4 items-start ${isUser ? 'flex-row-reverse' : 'group'}`}
    >
      {/* 消息头像 - 仅在 User 时显示，Agent 时将头像和名字显示在气泡上方 */}
      {isUser && (
        <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 mt-1 bg-surface-container-highest border border-white/10">
          <User className="w-4 h-4 text-slate-400" />
        </div>
      )}

      <div className={`flex-1 space-y-2 ${isUser ? 'max-w-[80%]' : 'w-full overflow-hidden'}`}>
        {/* Agent 头像与名字 (气泡上方) */}
        {isAgent && agent && (
          <div className="flex items-center gap-2.5 mb-1.5 px-1">
            <div className={`w-7 h-7 rounded-md flex items-center justify-center shrink-0 bg-primary/10 ghost-border`}>
              <agent.avatarIcon className={`w-3.5 h-3.5 ${agent.colorClass}`} />
            </div>
            <span className="text-sm font-semibold text-slate-300">{agent.name}</span>
          </div>
        )}

        <div className={`p-5 rounded-xl ghost-border flex flex-col gap-4 ${
          isAgent ? 'bg-primary-container/10' : 'bg-surface-container-highest'
        } ${isAgent && message.isStreaming ? 'agent-processing-border' : ''}`}>
          {isProcessing && (
            <div className="flex items-center gap-2 text-primary/80 text-sm font-medium">
              <BrainCircuit className="w-4 h-4 animate-pulse" />
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
                textClassName={`text-sm leading-relaxed text-on-surface ${isAgent ? 'nebula-glow' : ''}`}
              />
            ) : (
              <SubAgentCard key={idx} block={segment.block} />
            )
          ))}
        </div>

        {/* Agent 消息工具按钮 */}
        {isAgent && !message.isStreaming && (
          <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
            {/* 复制消息 */}
            <button className="p-1.5 rounded bg-white/5 hover:bg-white/10 text-slate-400">
              <Copy className="w-3 h-3" />
            </button>
            {/* 反馈 */}
            <button className="p-1.5 rounded bg-white/5 hover:bg-white/10 text-slate-400">
              <ThumbsUp className="w-3 h-3" />
            </button>
            {/* 重新生成 */}
            <button className="p-1.5 rounded bg-white/5 hover:bg-white/10 text-slate-400">
              <RefreshCw className="w-3 h-3" />
            </button>
          </div>
        )}
      </div>
    </motion.div>
  );
}
