import { Bot, User, Copy, ThumbsUp, RefreshCw } from 'lucide-react';
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
        }`}>
          {blocks.map((block, idx) => {
            if (block.kind === 'text') {
              if (!block.text) return null;
              return (
                <div key={idx} className={`text-sm leading-relaxed text-on-surface ${isAgent ? 'nebula-glow' : ''}`}>
                  <MarkdownRenderer content={block.text} />
                  {/* 如果正在流式输出，并且是最后一个文本块，显示光标 */}
                  {message.isStreaming && idx === lastTextIdx && (
                    <span className="inline-block w-1.5 h-4 bg-primary align-middle animate-pulse ml-1 rounded-full shadow-[0_0_8px_rgba(147,51,234,0.5)]" />
                  )}
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