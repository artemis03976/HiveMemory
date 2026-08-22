/**
 * Topic Store - Zustand state management for topic list
 *
 * Features:
 * - Fetch active topics from backend
 * - Optimistic settle / delete with rollback on error
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { fetchTopics, settleTopic, deleteTopic, mapTopic } from '@/services/topicApi';
import type { Topic } from '@/types';
import type { TopicPoolInfo } from '@/types/chat';

interface TopicStore {
  topics: Topic[];
  isLoading: boolean;
  error: string | null;

  fetchTopics: () => Promise<void>;
  settleTopic: (topicId: string) => Promise<void>;
  deleteTopic: (topicId: string) => Promise<void>;
  setTopicsFromPool: (pool: TopicPoolInfo, activeTopicId: string) => void;
}

export const useTopicStore = create<TopicStore>()(
  devtools(
    (set, get) => ({
      topics: [],
      isLoading: false,
      error: null,

      fetchTopics: async () => {
        set({ isLoading: true, error: null });
        try {
          const topics = await fetchTopics();
          set({ topics, isLoading: false });
        } catch (e) {
          set({ isLoading: false, error: e instanceof Error ? e.message : '加载话题失败' });
        }
      },

      settleTopic: async (topicId: string) => {
        const prev = get().topics;
        set({ topics: prev.filter((t) => t.id !== topicId) });
        try {
          await settleTopic(topicId);
        } catch (e) {
          set({ topics: prev, error: e instanceof Error ? e.message : '结算失败' });
        }
      },

      deleteTopic: async (topicId: string) => {
        const prev = get().topics;
        set({ topics: prev.filter((t) => t.id !== topicId) });
        try {
          await deleteTopic(topicId);
        } catch (e) {
          set({ topics: prev, error: e instanceof Error ? e.message : '删除失败' });
        }
      },

      setTopicsFromPool: (pool: TopicPoolInfo, activeTopicId: string) => {
        const topics: Topic[] = pool.topics.map((t) => mapTopic(t, activeTopicId));
        set({ topics, isLoading: false, error: null });
      },
    }),
    { name: 'TopicStore' }
  )
);
