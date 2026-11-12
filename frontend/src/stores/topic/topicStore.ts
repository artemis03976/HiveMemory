/**
 * 短期话题池的状态：话题列表、记忆图书馆中选中的话题，以及结算与丢弃。
 * 结算与丢弃先从列表中移除，失败时恢复列表并把错误交给调用方提示。
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { fetchTopics, settleTopic, deleteTopic } from '@/services/topicApi';
import { localErrorText } from '@/transports/connectionStatus';
import type { Topic } from '@/types';

interface TopicStore {
  topics: Topic[];
  isLoading: boolean;
  error: string | null;
  /** 记忆图书馆中选中的话题；从对话页跳转过来时由跳转方设定。 */
  selectedTopicId: string | null;

  fetchTopics: () => Promise<void>;
  selectTopic: (topicId: string) => void;
  settleTopic: (topicId: string) => Promise<void>;
  deleteTopic: (topicId: string) => Promise<void>;
}

export const useTopicStore = create<TopicStore>()(
  devtools(
    (set, get) => ({
      topics: [],
      isLoading: false,
      error: null,
      selectedTopicId: null,

      fetchTopics: async () => {
        set({ isLoading: true, error: null });
        try {
          const topics = await fetchTopics();
          set({ topics, isLoading: false });
        } catch (e) {
          set({ isLoading: false, error: localErrorText(e instanceof Error ? e.message : '话题读取失败') });
        }
      },

      selectTopic: (topicId) => set({ selectedTopicId: topicId }),

      settleTopic: async (topicId) => {
        const prev = get().topics;
        set({ topics: prev.filter((t) => t.id !== topicId) });
        try {
          await settleTopic(topicId);
        } catch (e) {
          set({ topics: prev });
          throw e;
        }
      },

      deleteTopic: async (topicId) => {
        const prev = get().topics;
        set({ topics: prev.filter((t) => t.id !== topicId) });
        try {
          await deleteTopic(topicId);
        } catch (e) {
          set({ topics: prev });
          throw e;
        }
      },
    }),
    { name: 'TopicStore' }
  )
);
