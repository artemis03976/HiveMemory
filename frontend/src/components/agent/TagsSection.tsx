import { useState, useCallback } from 'react';
import { Tag, Plus, X } from 'lucide-react';

interface TagsSectionProps {
  tags: string[];
  onChange: (tags: string[]) => void;
}

export function TagsSection({ tags, onChange }: TagsSectionProps) {
  const [input, setInput] = useState('');

  const addTag = useCallback(() => {
    const trimmed = input.trim().toLowerCase();
    if (trimmed && !tags.includes(trimmed)) {
      onChange([...tags, trimmed]);
    }
    setInput('');
  }, [input, tags, onChange]);

  const removeTag = useCallback((tag: string) => {
    onChange(tags.filter(t => t !== tag));
  }, [tags, onChange]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addTag();
    }
  }, [addTag]);

  return (
    <section className="space-y-3">
      <label className="text-sm font-bold text-slate-200 flex items-center gap-2">
        <Tag className="w-4 h-4 text-primary" />
        Tags
      </label>

      <div className="flex flex-wrap items-center gap-2">
        {tags.map(tag => (
          <span
            key={tag}
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-primary/10 border border-primary/20 text-xs text-primary font-mono group"
          >
            {tag}
            <button
              onClick={() => removeTag(tag)}
              className="opacity-0 group-hover:opacity-100 transition-opacity hover:text-red-400"
            >
              <X className="w-3 h-3" />
            </button>
          </span>
        ))}

        <div className="inline-flex items-center gap-1 bg-black/20 border border-white/10 rounded-lg px-2 py-1 focus-within:border-primary/50 transition-colors">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="add tag..."
            className="bg-transparent border-none text-xs text-white placeholder-slate-500 focus:outline-none w-20 font-mono"
          />
          <button
            onClick={addTag}
            disabled={!input.trim()}
            className="text-slate-500 hover:text-primary disabled:opacity-30 transition-colors"
          >
            <Plus className="w-3 h-3" />
          </button>
        </div>
      </div>
    </section>
  );
}
