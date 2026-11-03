import { Search, Filter, SortDesc, LayoutGrid, List } from 'lucide-react';
import type { SearchMode, SortOption, MemoryType } from '@/types/memory';
import { memoryTypeLabels } from '@/types/memory';

interface CommandBarProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  searchMode: SearchMode;
  onSearchModeChange: (mode: SearchMode) => void;
  viewMode: 'grid' | 'list';
  onViewModeChange: (mode: 'grid' | 'list') => void;
  sortBy: SortOption;
  onSortByChange: (sort: SortOption) => void;
  selectedType: string | null;
  onSelectedTypeChange: (type: string | null) => void;
}

export default function CommandBar({
  searchQuery,
  onSearchChange,
  searchMode,
  onSearchModeChange,
  viewMode,
  onViewModeChange,
  sortBy,
  onSortByChange,
  selectedType,
  onSelectedTypeChange
}: CommandBarProps) {
  return (
    <div className="px-8 py-4 flex items-center gap-4 border-b border-white/5 bg-surface-container-low/50">
      {/* 搜索：语义检索请求后端覆盖整个记忆库，本地过滤只匹配已加载的记忆 */}
      <div className="flex bg-black/20 p-1 rounded-lg border border-white/5 shrink-0">
        {([['semantic', '语义检索'], ['exact', '本地过滤']] as [SearchMode, string][]).map(([mode, label]) => (
          <button
            key={mode}
            onClick={() => onSearchModeChange(mode)}
            className={`px-2.5 py-1 rounded-md text-xs transition-all ${searchMode === mode ? 'bg-white/10 text-primary shadow-sm' : 'text-slate-500 hover:text-slate-300'}`}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="flex-1 relative group">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 group-focus-within:text-primary transition-colors" />
        <input
          type="text"
          placeholder={searchMode === 'semantic' ? '按语义检索整个记忆库...' : '过滤已加载的记忆（支持 alias: 与 tag: 前缀）...'}
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="w-full bg-black/20 border border-white/10 rounded-xl py-2 pl-10 pr-4 text-sm text-white placeholder:text-slate-600 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all font-mono"
        />
      </div>

      {/* Filters */}
      <div className="flex items-center gap-2 relative group/filter">
        <button className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/5 text-sm text-slate-300 transition-colors">
          <Filter className="w-4 h-4" />
          {selectedType ? memoryTypeLabels[selectedType as MemoryType] : '类型筛选'}
        </button>
        {/* 类型筛选下拉框 */}
        <div className="absolute top-full mt-2 left-0 w-40 bg-surface-container-high border border-white/10 rounded-xl shadow-xl opacity-0 invisible group-hover/filter:opacity-100 group-hover/filter:visible transition-all z-50 overflow-hidden">
          <div
            onClick={() => onSelectedTypeChange(null)}
            className={`px-4 py-2 text-sm cursor-pointer hover:bg-white/5 transition-colors ${!selectedType ? 'text-primary bg-primary/10' : 'text-slate-300'}`}
          >
            全部类型
          </div>
          {Object.entries(memoryTypeLabels).map(([type, label]) => (
            <div 
              key={type}
              onClick={() => onSelectedTypeChange(type)}
              className={`px-4 py-2 text-sm cursor-pointer hover:bg-white/5 transition-colors ${selectedType === type ? 'text-primary bg-primary/10' : 'text-slate-300'}`}
            >
              {label}
            </div>
          ))}
        </div>
      </div>

      <div className="w-px h-6 bg-white/10 mx-2" />

      {/* 排序控制栏 */}
      <div className="relative group/sort">
        <button className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/5 text-sm text-slate-300 transition-colors">
          <SortDesc className="w-4 h-4" />
          <span className="capitalize">
            {{
              created_desc: '最新创建',
              created_asc: '最早创建',
              accessed: '最近访问',
              count: '访问次数最多',
              confidence: '置信度最高'
            }[sortBy]}
          </span>
        </button>
        {/* 排序下拉框 */}
        <div className="absolute top-full mt-2 right-0 w-40 bg-surface-container-high border border-white/10 rounded-xl shadow-xl opacity-0 invisible group-hover/sort:opacity-100 group-hover/sort:visible transition-all z-50 overflow-hidden">
          {(['created_desc', 'created_asc', 'accessed', 'count', 'confidence'] as SortOption[]).map((option) => (
            <div 
              key={option}
              onClick={() => onSortByChange(option)}
              className={`px-4 py-2 text-sm cursor-pointer hover:bg-white/5 transition-colors capitalize ${sortBy === option ? 'text-primary bg-primary/10' : 'text-slate-300'}`}
            >
              {{
                created_desc: '最新创建',
                created_asc: '最早创建',
                accessed: '最近访问',
                count: '访问次数最多',
                confidence: '置信度最高'
              }[option]}
            </div>
          ))}
        </div>
      </div>

      <div className="w-px h-6 bg-white/10 mx-2" />

      {/* 视图切换 */}
        <div className="flex bg-black/20 p-1 rounded-lg border border-white/5">
        <button
          onClick={() => onViewModeChange('grid')}
          className={`p-1.5 rounded-md transition-all ${viewMode === 'grid' ? 'bg-white/10 text-primary shadow-sm' : 'text-slate-500 hover:text-slate-300'}`}
        >
          <LayoutGrid className="w-4 h-4" />
        </button>
        <button
          onClick={() => onViewModeChange('list')}
          className={`p-1.5 rounded-md transition-all ${viewMode === 'list' ? 'bg-white/10 text-primary shadow-sm' : 'text-slate-500 hover:text-slate-300'}`}
        >
          <List className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}