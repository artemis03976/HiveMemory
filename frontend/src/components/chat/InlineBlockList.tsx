import type { InlineBlock } from '@/types';
import MarkdownRenderer from '../common/MarkdownRenderer';
import MTPCard from './MTPCard';

interface InlineBlockListProps {
  blocks: InlineBlock[];
  isStreaming?: boolean;
  textClassName?: string;
  animateLastTextCursor?: boolean;
}

export default function InlineBlockList({
  blocks,
  isStreaming = false,
  textClassName = 'text-sm leading-relaxed text-on-surface',
  animateLastTextCursor = false,
}: InlineBlockListProps) {
  let lastTextIdx = -1;
  for (let i = blocks.length - 1; i >= 0; i--) {
    if (blocks[i].kind === 'text') {
      lastTextIdx = i;
      break;
    }
  }

  return (
    <>
      {blocks.map((block, idx) => {
        if (block.kind === 'text') {
          if (!block.text) return null;
          const showCursor = animateLastTextCursor && isStreaming && idx === lastTextIdx;
          return (
            <div key={idx} className={`${textClassName} ${showCursor ? 'typing-cursor' : ''}`}>
              <MarkdownRenderer content={showCursor ? block.text + '\u200B' : block.text} />
            </div>
          );
        }

        return <MTPCard key={idx} action={block.action} />;
      })}
    </>
  );
}
