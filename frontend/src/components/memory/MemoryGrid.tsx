import { motion } from 'framer-motion';
import type { MemoryResponse } from '@/types/memory';
import { MemoryCard } from './MemoryCard';

interface MemoryGridProps {
  memories: MemoryResponse[];
  onCardClick: (memory: MemoryResponse) => void;
}

export function MemoryGrid({ memories, onCardClick }: MemoryGridProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-6">
      {memories.map((memory, index) => (
        <motion.div
          key={memory.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: index * 0.05 }}
        >
          <MemoryCard memory={memory} onClick={() => onCardClick(memory)} />
        </motion.div>
      ))}
    </div>
  );
}
