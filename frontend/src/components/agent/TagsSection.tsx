import { useState } from 'react';

interface TagsSectionProps {
  tags: string[];
  onChange: (tags: string[]) => void;
}

/** 标签编辑：回车加入，统一为小写；重复的标签不再加入。 */
export function TagsSection({ tags, onChange }: TagsSectionProps) {
  const [input, setInput] = useState('');

  const addTag = () => {
    const tag = input.trim().toLowerCase();
    if (tag && !tags.includes(tag)) onChange([...tags, tag]);
    setInput('');
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      {tags.map((tag) => (
        <span key={tag} className="inline-flex items-center gap-1 rounded-full bg-surface-container-high py-0.5 pl-2.5 pr-1.5 text-xs text-fg-2">
          #{tag}
          <button
            onClick={() => onChange(tags.filter((t) => t !== tag))}
            aria-label={`移除标签 ${tag}`}
            className="rounded-full px-0.5 leading-none text-fg-3 transition-colors hover:text-fg"
          >
            ×
          </button>
        </span>
      ))}
      <input
        type="text"
        aria-label="添加标签"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key !== 'Enter') return;
          e.preventDefault();
          addTag();
        }}
        placeholder="添加标签，回车确认"
        className="h-7 w-40 rounded-full border border-dashed border-line-strong bg-transparent px-2.5 text-xs text-fg placeholder:text-fg-3 focus:border-primary/50 focus:outline-none"
      />
    </div>
  );
}
