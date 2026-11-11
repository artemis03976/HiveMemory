import { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import { useMemoryViewStore, useToastStore, useTopicStore } from '@/stores';
import type { LibraryTab } from '@/stores/memory/viewStore';
import { useMemories } from '@/hooks/useMemories';
import type { MemoryAtom } from '@/types/memory';
import TopicPool from './memory/TopicPool';
import MemoryAtomCard from './memory/MemoryAtomCard';
import MemoryAtomListItem from './memory/MemoryAtomListItem';
import MemoryHeader from './memory/MemoryHeader';
import CommandBar from './memory/CommandBar';
import MemoryDetailModal from './memory/MemoryDetailModal';
import CreateMemoryModal from './memory/CreateMemoryModal';
import { ConfirmDialog } from './common/ConfirmDialog';

function LibraryTabs({ tab, onChange, memoryCount, topicCount }: {
  tab: LibraryTab;
  onChange: (tab: LibraryTab) => void;
  memoryCount: number;
  topicCount: number;
}) {
  const items: [LibraryTab, string][] = [['memories', `长期记忆 · ${memoryCount}`], ['topics', `短期话题池 · ${topicCount}`]];
  return (
    <div role="tablist" aria-label="记忆种类" className="flex shrink-0 gap-0.5 rounded-[11px] border border-line bg-field p-[3px]">
      {items.map(([id, label]) => (
        <button
          key={id}
          role="tab"
          aria-selected={tab === id}
          onClick={() => onChange(id)}
          className={`h-8 whitespace-nowrap rounded-lg px-3.5 text-[13px] font-semibold transition-colors ${
            tab === id ? 'bg-surface-container-high text-fg' : 'text-fg-3 hover:text-fg'
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

export default function MemoryLibrary() {
  const { addToast } = useToastStore();
  const { topics, fetchTopics } = useTopicStore();
  const { libraryTab, setLibraryTab } = useMemoryViewStore();
  useEffect(() => { void fetchTopics(); }, [fetchTopics]);
  const [selectedAtom, setSelectedAtom] = useState<MemoryAtom | null>(null);
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);

  const {
    memories,
    loading,
    error,
    total,
    loadedCount,
    searchQuery,
    setSearchQuery,
    searchMode,
    setSearchMode,
    semanticStatus,
    semanticError,
    viewMode,
    setViewMode,
    sortBy,
    setSortBy,
    deleteMemory,
    updateMemory,
    selectedType,
    setSelectedType,
    refetch,
  } = useMemories();

  const handleView = (id: string) => {
    const atom = memories.find(m => m.id === id);
    if (atom) setSelectedAtom(atom);
  };
  const handleEdit = (id: string) => {
    const atom = memories.find(m => m.id === id);
    if (atom) setSelectedAtom(atom);
  };
  const handleModalEdit = async (id: string, patch: Parameters<typeof updateMemory>[1]) => {
    try {
      await updateMemory(id, patch);
      setSelectedAtom(prev => prev?.id === id ? { ...prev, ...patch, tags: patch.tags ?? prev.tags, alias: patch.alias ?? prev.alias } : prev);
      addToast('记忆已更新', 'success');
    } catch {
      addToast('更新失败，请重试', 'error');
    }
  };
  const confirmDelete = (id: string) => {
    setDeleteId(id);
  };
  const executeDelete = async () => {
    if (deleteId) {
      try {
        await deleteMemory(deleteId);
        addToast('记忆已删除', 'success');
      } catch {
        addToast('删除失败，请重试', 'error');
      }
      if (selectedAtom?.id === deleteId) {
        setSelectedAtom(null);
      }
      setDeleteId(null);
    }
  };

  const tabs = <LibraryTabs tab={libraryTab} onChange={setLibraryTab} memoryCount={total} topicCount={topics.length} />;

  return (
    <div className="flex-1 flex flex-col h-full bg-background overflow-hidden">
      {/* Header & Stats */}
      <MemoryHeader
        totalMemories={total}
        loadedMemories={loadedCount}
        warnings={0}
        onNewMemory={() => setShowCreateModal(true)}
      />

      {libraryTab === 'topics' ? (
        <TopicPool tabs={tabs} />
      ) : (<>
      {/* Command Bar */}
      <CommandBar
        leading={tabs}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        searchMode={searchMode}
        onSearchModeChange={setSearchMode}
        viewMode={viewMode}
        onViewModeChange={setViewMode}
        sortBy={sortBy}
        onSortByChange={setSortBy}
        selectedType={selectedType}
        onSelectedTypeChange={setSelectedType}
      />

      {/* Atom Cards Area */}
      <div className="flex-1 overflow-y-auto p-8 scrollbar-hide">
        {semanticStatus === 'done' && memories.length > 0 && (
          <p className="mb-4 text-xs text-fg-3">
            语义检索：{memories.length} 条结果，按相关度排列（排序选项对检索结果不生效）
          </p>
        )}
        {loading ? (
          <div className="flex items-center justify-center h-full text-fg-3">
            <span className="animate-pulse">加载记忆中...</span>
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-full text-magic-fire">
            {error}
          </div>
        ) : semanticStatus === 'loading' ? (
          <div className="flex items-center justify-center h-full text-fg-3">
            <span className="animate-pulse">语义检索中...</span>
          </div>
        ) : semanticStatus === 'error' ? (
          <div className="flex items-center justify-center h-full text-magic-fire">
            语义检索失败：{semanticError}
          </div>
        ) : memories.length === 0 ? (
          <div className="flex items-center justify-center h-full text-fg-3">
            {semanticStatus === 'done' ? '没有找到语义相关的记忆。' : '未找到记忆数据。'}
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
                      onDelete={confirmDelete}
                    />
                  ) : (
                    <MemoryAtomListItem
                      atom={atom}
                      onView={() => handleView(atom.id)}
                      onEdit={handleEdit}
                      onDelete={confirmDelete}
                    />
                  )}
                </motion.div>
            ))}
          </motion.div>
        )}
      </div>
      </>)}

      {/* Detail Modal */}
      {selectedAtom && (
        <MemoryDetailModal
          atom={selectedAtom}
          onClose={() => setSelectedAtom(null)}
          onEdit={handleModalEdit}
          onDelete={confirmDelete}
        />
      )}

      {/* Delete Confirmation */}
      <ConfirmDialog
        isOpen={deleteId !== null}
        title="删除记忆"
        message="确定要删除这条记忆吗？此操作不可撤销。"
        confirmText="删除"
        onConfirm={executeDelete}
        onCancel={() => setDeleteId(null)}
      />

      {/* Create Memory Modal */}
      <CreateMemoryModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onCreated={() => { refetch(); setShowCreateModal(false); }}
      />
    </div>
  );
}
