import { useState } from 'react';
import { motion } from 'motion/react';
import { useMemories } from '@/hooks/useMemories';
import type { MemoryAtom } from '@/types/memory';
import MemoryAtomCard from './memory/MemoryAtomCard';
import MemoryAtomListItem from './memory/MemoryAtomListItem';
import MemoryHeader from './memory/MemoryHeader';
import CommandBar from './memory/CommandBar';
import MemoryDetailModal from './memory/MemoryDetailModal';

export default function MemoryGarden() {
  const [selectedAtom, setSelectedAtom] = useState<MemoryAtom | null>(null);

  const {
    memories,
    loading,
    error,
    total,
    searchQuery,
    setSearchQuery,
    viewMode,
    setViewMode,
    sortBy,
    setSortBy,
    deleteMemory,
    selectedType,
    setSelectedType
  } = useMemories();

  const handleView = (id: string) => {
    const atom = memories.find(m => m.id === id);
    if (atom) setSelectedAtom(atom);
  };
  const handleEdit = (id: string) => console.log('Edit', id);
  const handlePin = (id: string) => console.log('Pin', id);
  const handleDelete = (id: string) => {
    deleteMemory(id);
    if (selectedAtom?.id === id) {
      setSelectedAtom(null);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-background overflow-hidden">
      {/* Header & Stats */}
      <MemoryHeader 
        totalMemories={total}
        dbSize="45MB"
        warnings={0}
        onNewMemory={() => console.log('New Memory')}
      />

      {/* Command Bar */}
      <CommandBar 
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        viewMode={viewMode}
        onViewModeChange={setViewMode}
        sortBy={sortBy}
        onSortByChange={setSortBy}
        selectedType={selectedType}
        onSelectedTypeChange={setSelectedType}
      />

      {/* Atom Cards Area */}
      <div className="flex-1 overflow-y-auto p-8 scrollbar-hide">
        {loading ? (
          <div className="flex items-center justify-center h-full text-slate-500">
            <span className="animate-pulse">加载记忆中...</span>
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-full text-red-400">
            {error}
          </div>
        ) : memories.length === 0 ? (
          <div className="flex items-center justify-center h-full text-slate-500">
            未找到记忆数据。
          </div>
        ) : (
          <motion.div 
            key={`${viewMode}-${sortBy}-${selectedType}-${searchQuery}`}
            initial="hidden"
            animate="visible"
            variants={{
              hidden: { opacity: 0 },
              visible: { 
                opacity: 1,
                transition: {
                  staggerChildren: 0.05
                }
              }
            }}
            className={
              viewMode === 'grid' 
                ? "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 auto-rows-max"
                : "flex flex-col max-w-5xl mx-auto"
            }
          >
            {memories.map((atom) => (
              <motion.div
                key={atom.id}
                variants={{
                  hidden: { opacity: 0, y: 20 },
                  visible: { opacity: 1, y: 0, transition: { duration: 0.3 } }
                }}
              >
                  {viewMode === 'grid' ? (
                    <MemoryAtomCard
                      atom={atom}
                      onView={() => handleView(atom.id)}
                      onEdit={handleEdit}
                      onPin={handlePin}
                      onDelete={handleDelete}
                    />
                  ) : (
                    <MemoryAtomListItem
                      atom={atom}
                      onView={() => handleView(atom.id)}
                      onEdit={handleEdit}
                      onPin={handlePin}
                      onDelete={handleDelete}
                    />
                  )}
                </motion.div>
            ))}
          </motion.div>
        )}
      </div>

      {/* Detail Modal */}
      {selectedAtom && (
        <MemoryDetailModal
          atom={selectedAtom}
          onClose={() => setSelectedAtom(null)}
          onEdit={handleEdit}
          onPin={handlePin}
          onDelete={handleDelete}
        />
      )}
    </div>
  );
}
