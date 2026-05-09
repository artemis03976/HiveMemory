/**
 * Chat Store - Zustand state management for chat functionality
 *
 * Features:
 * - SSE connection to backend chat API
 * - Real-time message streaming with token accumulation
 * - MTP action tracking via ordered ContentBlock[]
 * - Message persistence (finalized messages only)
 * - Error handling and retry logic
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { ChatSSEClient } from '@/services/chatApi';
import { createChatSSECallbacks } from '@/stores/chatStore.callbacks';
import { applyDone, applyStreamError } from '@/stores/chatStore.updaters';
import { useTopicStore } from '@/stores/topicStore';
import { DEFAULT_AGENT_ID } from '@/constants/identity';
import type {
  Message,
  ChatConnectionState,
  ChatRequestParams,
  MemoryAtom,
} from '@/types';

// ========== Store Interface ==========

interface ChatStore {
  // State
  messages: Message[];
  connection: ChatConnectionState;
  isStreaming: boolean;
  currentTopicId: string | null;
  currentAgentId: string;
  retrievedMemories: MemoryAtom[];

  // Actions
  sendMessage: (content: string, options?: Partial<ChatRequestParams>) => Promise<void>;
  clearMessages: () => void;
  retryMessage: (messageId: string) => Promise<void>;
  setCurrentAgentId: (id: string) => void;

  // Internal
  _sseClient: ChatSSEClient | null;
  _currentStreamingMessageId: string | null;
}

// ========== Store Implementation ==========

export const useChatStore = create<ChatStore>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        messages: [],
        connection: {
          status: 'disconnected',
          error: null,
        },
        isStreaming: false,
        currentTopicId: null,
        currentAgentId: DEFAULT_AGENT_ID,
        retrievedMemories: [],
        _sseClient: null,
        _currentStreamingMessageId: null,

        // Agent Action
        setCurrentAgentId: (id: string) => set({ currentAgentId: id }),

        // Send message action
        sendMessage: async (content: string, options = {}) => {
          const state = get();

          // Prevent sending while already streaming
          if (state.isStreaming) {
            console.warn('[ChatStore] Already streaming, ignoring new message');
            return;
          }

          const currentAgentId = state.currentAgentId;

          // Add user message
          const userMessage: Message = {
            id: crypto.randomUUID(),
            role: 'user',
            content,
            contentBlocks: [{ kind: 'text', text: content }],
            timestamp: Date.now(),
          };

          set({
            messages: [...state.messages, userMessage],
            isStreaming: true,
            connection: { status: 'connecting', error: null },
            // 新一轮对话开始时先清空旧的引用记忆，避免后端无 memory_refs 事件时显示陈旧数据
            retrievedMemories: [],
          });

          // Create streaming assistant message
          const assistantMessageId = crypto.randomUUID();
          const assistantMessage: Message = {
            id: assistantMessageId,
            role: 'assistant',
            content: '',
            contentBlocks: [],
            timestamp: Date.now(),
            isStreaming: true,
            agent_id: currentAgentId, // mock assigning the current agent
          };

          set({
            messages: [...get().messages, assistantMessage],
            _currentStreamingMessageId: assistantMessageId,
          });

          // Initialize SSE client
          const client = new ChatSSEClient();
          set({ _sseClient: client });

          try {
            const callbacks = createChatSSECallbacks({
              assistantMessageId,
              updateMessages: (updater) => {
                set((s) => ({ messages: updater(s.messages) }));
              },
              setTopicInfo: (data) => {
                set({ currentTopicId: data.topic_id });
                if (data.pool) {
                  useTopicStore.getState().setTopicsFromPool(data.pool, data.topic_id);
                } else {
                  useTopicStore.getState().fetchTopics();
                }
              },
              setRetrievedMemories: (memories) => {
                set({ retrievedMemories: memories });
              },
              finalizeSuccess: (data) => {
                set((s) => ({
                  messages: applyDone(s.messages, assistantMessageId, data.final_text),
                  isStreaming: false,
                  connection: { status: 'connected', error: null },
                  _currentStreamingMessageId: null,
                }));
              },
              finalizeError: (errorMessage, errorDetail) => {
                set((s) => ({
                  messages: applyStreamError(s.messages, assistantMessageId, errorMessage, errorDetail),
                  isStreaming: false,
                  connection: { status: 'error', error: errorMessage },
                  _currentStreamingMessageId: null,
                }));
              },
            });

            await client.connect(
              {
                message: content,
                agent_id: currentAgentId,
                ...options,
              },
              callbacks
            );
          } catch (error) {
            console.error('[ChatStore] Failed to send message:', error);
          } finally {
            // Cleanup
            client.disconnect();
            set({ _sseClient: null });
          }
        },

        // Clear all messages
        clearMessages: () => {
          set({
            messages: [],
            currentTopicId: null,
            retrievedMemories: [],
          });
        },

        // Retry a failed message
        retryMessage: async (messageId: string) => {
          const state = get();
          const messageIndex = state.messages.findIndex((m) => m.id === messageId);

          if (messageIndex === -1) return;

          // Find the user message before this failed message
          const userMessage = state.messages
            .slice(0, messageIndex)
            .reverse()
            .find((m) => m.role === 'user');

          if (!userMessage) return;

          // Remove failed message and retry
          set({
            messages: state.messages.filter((m) => m.id !== messageId),
          });

          await get().sendMessage(userMessage.content);
        },
      }),
      {
        name: 'chat-store',
        version: 3,
        partialize: (state) => ({
          messages: [],
          currentTopicId: null,
          currentAgentId: state.currentAgentId,
        }),
        // 测试阶段：升级后统一清空已持久化会话，避免渲染异常导致刷新后继续崩溃
        migrate: (persisted: unknown, version: number) => {
          if (!persisted || typeof persisted !== 'object') return persisted;
          const migrated = persisted as { messages?: unknown; currentTopicId?: unknown; currentAgentId?: unknown };
          if (version < 2) {
            migrated.messages = [];
            migrated.currentTopicId = null;
            migrated.currentAgentId = DEFAULT_AGENT_ID;
          }
          if (version < 3) {
            if (migrated.currentAgentId === 'default' || !migrated.currentAgentId) {
              migrated.currentAgentId = DEFAULT_AGENT_ID;
            }
          }
          return migrated;
        },
      }
    ),
    { name: 'ChatStore' }
  )
);
