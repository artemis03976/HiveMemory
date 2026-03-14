import { Filter, ArrowUpDown } from 'lucide-react';
import type { MemoryType, SortOption } from '@/types/memory';
import { memoryTypeLabels } from '@/types/memory';

interface FilterControlsProps {
  selectedType: string | null;
  onTypeChange: (type: string | null) => void;
}

export function FilterControls({ selectedType, onTypeChange }: FilterControlsProps) {
  const memoryTypes: MemoryType[] = [
    'CODE_SNIPPET',
    'FACT',
    'URL_RESOURCE',
    'REFLECTION',
    'USER_PROFILE',
    'WORK_IN_PROGRESS',
  ];

  return (
    <div className="flex items-center gap-2">
      <Filter className="w-4 h-4 text-muted-foreground" />
      <select
        value={selectedType || ''}
        onChange={(e) => onTypeChange(e.target.value || null)}
        className="px-3 py-2 bg-muted/20 border border-white/10 rounded-lg text-sm focus:outline-none focus:ring-1 focus:ring-primary/50 cursor-pointer"
      >
        <option value="">全部类型</option>
        {memoryTypes.map((type) => (
          <option key={type} value={type}>
            {memoryTypeLabels[type]}
          </option>
        ))}
      </select>
    </div>
  );
}

interface SortControlsProps {
  sortBy: SortOption;
  onSortChange: (sort: SortOption) => void;
}

export function SortControls({ sortBy, onSortChange }: SortControlsProps) {
  const sortOptions: { value: SortOption; label: string }[] = [
    { value: 'created_desc', label: '最新创建' },
    { value: 'created_asc', label: '最早创建' },
    { value: 'accessed', label: '最近访问' },
    { value: 'count', label: '最常使用' },
    { value: 'confidence', label: '最高置信度' },
  ];

  return (
    <div className="flex items-center gap-2">
      <ArrowUpDown className="w-4 h-4 text-muted-foreground" />
      <select
        value={sortBy}
        onChange={(e) => onSortChange(e.target.value as SortOption)}
        className="px-3 py-2 bg-muted/20 border border-white/10 rounded-lg text-sm focus:outline-none focus:ring-1 focus:ring-primary/50 cursor-pointer"
      >
        {sortOptions.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}
