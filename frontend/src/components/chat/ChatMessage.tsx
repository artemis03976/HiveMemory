import { Bot, User, Copy, ThumbsUp, RefreshCw, BrainCircuit } from 'lucide-react';
import type { Message, ContentBlock } from '@/types';
import { motion } from 'motion/react';
import MTPCard from './MTPCard';
import MarkdownRenderer from '../common/MarkdownRenderer';

interface ChatMessageProps {
  message: Message;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isAgent = message.role === 'agent' || message.role === 'assistant';
  const isUser = message.role === 'user';
  
  const blocks = message.contentBlocks || (
    message.mtpAction 
      ? [{ kind: 'text', text: message.content }, { kind: 'mtp', action: message.mtpAction }]
      : [{ kind: 'text', text: message.content }]
  ) as ContentBlock[];

  const hasContent = blocks.some(b => (b.kind === 'text' && b.text) || (b.kind === 'mtp' && b.action));
  const isProcessing = isAgent && message.isStreaming && !hasContent;

  let lastTextIdx = -1;
  for (let i = blocks.length - 1; i >= 0; i--) {
    if (blocks[i].kind === 'text') {
      lastTextIdx = i;
      break;
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex gap-4 items-start ${isUser ? 'flex-row-reverse' : 'group'}`}
    >
      {/* 消息头像 */}
      <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 mt-1 ${
        isAgent ? 'bg-primary/10 ghost-border' : 'bg-surface-container-highest border border-white/10'
      }`}>
        {isAgent ? (
          <Bot className="w-4 h-4 text-primary" />
        ) : (
          <User className="w-4 h-4 text-slate-400" />
        )}
      </div>

      <div className={`flex-1 space-y-4 ${isUser ? 'max-w-[80%]' : 'w-full overflow-hidden'}`}>
        <div className={`p-5 rounded-xl ghost-border flex flex-col gap-4 ${
          isAgent ? 'bg-primary-container/10' : 'bg-surface-container-highest'
        } ${isAgent && message.isStreaming ? 'agent-processing-border' : ''}`}>
          {isProcessing && (
            <div className="flex items-center gap-2 text-primary/80 text-sm font-medium">
              <BrainCircuit className="w-4 h-4 animate-pulse" />
              <span>思考中<span className="thinking-dots"></span></span>
            </div>
          )}
          {blocks.map((block, idx) => {
            if (block.kind === 'text') {
              if (!block.text) return null;
              return (
                <div key={idx} className={`text-sm leading-relaxed text-on-surface ${isAgent ? 'nebula-glow' : ''} ${message.isStreaming && idx === lastTextIdx ? 'typing-cursor' : ''}`}>
                  <MarkdownRenderer content={message.isStreaming && idx === lastTextIdx ? block.text + '\u200B' : block.text} />
                </div>
              );
            }
            if (block.kind === 'mtp' && block.action) {
              return <MTPCard key={idx} action={block.action} />;
            }
            return null;
          })}
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