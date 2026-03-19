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
import type {
  Message,
  MtpAction,
  ContentBlock,
  ChatConnectionState,
  ChatRequestParams,
} from '@/types';

// ========== Store Interface ==========

interface ChatStore {
  // State
  messages: Message[];
  connection: ChatConnectionState;
  isStreaming: boolean;
  currentTopicId: string | null;

  // Actions
  sendMessage: (content: string, options?: Partial<ChatRequestParams>) => Promise<void>;
  clearMessages: () => void;
  retryMessage: (messageId: string) => Promise<void>;

  // Internal
  _sseClient: ChatSSEClient | null;
  _currentStreamingMessageId: string | null;
}

// ========== Helpers ==========

/** Get or create the trailing TextBlock in contentBlocks */
function appendTokenToBlocks(blocks: ContentBlock[], token: string): ContentBlock[] {
  const updated = [...blocks];
  const last = updated[updated.length - 1];
  if (last && last.kind === 'text') {
    updated[updated.length - 1] = { kind: 'text', text: last.text + token };
  } else {
    updated.push({ kind: 'text', text: token });
  }
  return updated;
}

/** Normalize stream delta to avoid duplicated prefix/overlap chunks */
function normalizeStreamDelta(currentText: string, incoming: string): string {
  if (!incoming) return '';
  if (!currentText) return incoming;
  if (currentText.endsWith(incoming)) return '';
  if (incoming.startsWith(currentText)) return incoming.slice(currentText.length);

  const maxOverlap = Math.min(currentText.length, incoming.length);
  for (let k = maxOverlap; k > 0; k--) {
    if (currentText.slice(-k) === incoming.slice(0, k)) {
      return incoming.slice(k);
    }
  }
  return incoming;
}

/** Push a new MTP block, cutting the current text segment */
function pushMtpBlock(blocks: ContentBlock[], action: MtpAction): ContentBlock[] {
  return [...blocks, { kind: 'mtp', action }];
}

/** Update the last MTP block's action status */
function updateLastMtpStatus(
  blocks: ContentBlock[],
  payload: { status: string; verb?: string; target?: string; args?: Record<string, unknown>; raw_text?: string }
): ContentBlock[] {
  const updated = [...blocks];
  for (let i = updated.length - 1; i >= 0; i--) {
    const b = updated[i];
    if (b.kind === 'mtp') {
      const verb = (payload.verb || b.action.type || 'UNKNOWN').toUpperCase();
      const command = payload.raw_text || [verb, payload.target].filter(Boolean).join(' | ') || b.action.command;
      updated[i] = {
        kind: 'mtp',
        action: {
          ...b.action,
          type: verb as any,
          command,
          params: payload.args ?? b.action.params,
          status: payload.status as any,
        },
      };
      break;
    }
  }
  return updated;
}

function normalizeVerb(verb?: string): string {
  const upper = (verb || '').toUpperCase();
  if (['RUN', 'READ', 'SEARCH', 'WRITE', 'UPDATE'].includes(upper)) {
    return upper;
  }
  return 'UNKNOWN';
}

/** Rebuild contentBlocks from final_text while preserving MTP blocks in-place */
function rebuildBlocksWithFinalText(blocks: ContentBlock[], finalText: string): ContentBlock[] {
  // If no MTP blocks exist, just return a single text block
  const hasMtp = blocks.some((b) => b.kind === 'mtp');
  if (!hasMtp) {
    return [{ kind: 'text', text: finalText }];
  }

  // Strategy: the backend's final_text is the concatenation of all text segments
  // (prefix_text from each iteration + final clean text). We split it back
  // by measuring how many text blocks we had during streaming.
  //
  // Simpler approach: keep MTP blocks in their positions, distribute final_text
  // across the text slots proportionally by their streaming lengths.
  const textSlots: { index: number; streamLen: number }[] = [];
  for (let i = 0; i < blocks.length; i++) {
    if (blocks[i].kind === 'text') {
      textSlots.push({ index: i, streamLen: (blocks[i] as any).text.length });
    }
  }

  const totalStreamLen = textSlots.reduce((s, t) => s + t.streamLen, 0);

  // If streaming accumulated nothing, put all final text in last slot
  if (totalStreamLen === 0) {
    const result: ContentBlock[] = blocks.map((b) =>
      b.kind === 'text' ? { kind: 'text', text: '' } : b
    );
    // Append final text after last mtp block
    result.push({ kind: 'text', text: finalText });
    return result;
  }

  // Distribute final_text proportionally
  const result: ContentBlock[] = [...blocks];
  let cursor = 0;
  for (let si = 0; si < textSlots.length; si++) {
    const slot = textSlots[si];
    const isLast = si === textSlots.length - 1;
    const charCount = isLast
      ? finalText.length - cursor
      : Math.round((slot.streamLen / totalStreamLen) * finalText.length);
    result[slot.index] = { kind: 'text', text: finalText.slice(cursor, cursor + charCount) };
    cursor += charCount;
  }

  return result;
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
        _sseClient: null,
        _currentStreamingMessageId: null,

        // Send message action
        sendMessage: async (content: string, options = {}) => {
          const state = get();

          // Prevent sending while already streaming
          if (state.isStreaming) {
            console.warn('[ChatStore] Already streaming, ignoring new message');
            return;
          }

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
          };

          set({
            messages: [...get().messages, assistantMessage],
            _currentStreamingMessageId: assistantMessageId,
          });

          // Initialize SSE client
          const client = new ChatSSEClient();
          set({ _sseClient: client });

          try {
            await client.connect(
              {
                message: content,
                ...options,
              },
              {
                // Token event: append to the trailing TextBlock
                onToken: (data) => {
                  set((state) => ({
                    messages: state.messages.map((msg) =>
                      msg.id === assistantMessageId
                        ? (() => {
                            const delta = normalizeStreamDelta(msg.content, data.content);
                            if (!delta) return msg;
                            return {
                              ...msg,
                              content: msg.content + delta,
                              contentBlocks: appendTokenToBlocks(msg.contentBlocks, delta),
                            };
                          })()
                        : msg
                    ),
                  }));
                },

                // MTP Start event: push a new MtpBlock (cuts the text stream)
                onMTPStart: (data) => {
                  const verb = normalizeVerb(data.verb);
                  const newAction: MtpAction = {
                    id: crypto.randomUUID(),
                    type: verb as any,
                    command: data.raw_text || [verb, data.target].filter(Boolean).join(' | ') || verb,
                    params: data.args,
                    status: 'executing',
                    timestamp: Date.now(),
                  };

                  set((state) => ({
                    messages: state.messages.map((msg) =>
                      msg.id === assistantMessageId
                        ? {
                            ...msg,
                            contentBlocks: pushMtpBlock(msg.contentBlocks, newAction),
                          }
                        : msg
                    ),
                  }));
                },

                // MTP Result event: update the last MTP block's status
                onMTPResult: (data) => {
                  set((state) => ({
                    messages: state.messages.map((msg) =>
                      msg.id === assistantMessageId
                        ? {
                            ...msg,
                            contentBlocks: updateLastMtpStatus(msg.contentBlocks, {
                              status: data.status,
                              verb: normalizeVerb(data.verb),
                              target: data.target,
                              args: data.args,
                              raw_text: data.raw_text,
                            }),
                          }
                        : msg
                    ),
                  }));
                },

                // Topic Info event: store topic ID
                onTopicInfo: (data) => {
                  set({ currentTopicId: data.topic_id });
                },

                // Done event: finalize message, rebuild text blocks from final_text
                onDone: (data) => {
                  set((state) => ({
                    messages: state.messages.map((msg) =>
                      msg.id === assistantMessageId
                        ? {
                            ...msg,
                            content: data.final_text,
                            contentBlocks: msg.contentBlocks.some((b) => b.kind === 'mtp')
                              ? msg.contentBlocks
                              : rebuildBlocksWithFinalText(msg.contentBlocks, data.final_text),
                            isStreaming: false,
                          }
                        : msg
                    ),
                    isStreaming: false,
                    connection: { status: 'connected', error: null },
                    _currentStreamingMessageId: null,
                  }));
                },

                // Error event: mark message with error
                onError: (data) => {
                  const errorMessage = data.message || '系统错误，请检查后端服务器';
                  set((state) => ({
                    messages: state.messages.map((msg) =>
                      msg.id === assistantMessageId
                        ? {
                            ...msg,
                            content: errorMessage,
                            contentBlocks: [{ kind: 'text', text: errorMessage }],
                            isStreaming: false,
                            error: errorMessage,
                          }
                        : msg
                    ),
                    isStreaming: false,
                    connection: { status: 'error', error: errorMessage },
                    _currentStreamingMessageId: null,
                  }));
                },

                // Connection Error: handle connection failures
                onConnectionError: (error) => {
                  const errorMessage = '系统错误，请检查后端服务器';
                  set((state) => ({
                    messages: state.messages.map((msg) =>
                      msg.id === assistantMessageId
                        ? {
                            ...msg,
                            content: errorMessage,
                            contentBlocks: [{ kind: 'text', text: errorMessage }],
                            isStreaming: false,
                            error: error.message,
                          }
                        : msg
                    ),
                    isStreaming: false,
                    connection: { status: 'error', error: errorMessage },
                    _currentStreamingMessageId: null,
                  }));
                },
              }
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
        version: 2,
        partialize: () => ({
          messages: [],
          currentTopicId: null,
        }),
        // 测试阶段：升级后统一清空已持久化会话，避免渲染异常导致刷新后继续崩溃
        migrate: (persisted: any, version: number) => {
          if (!persisted) return persisted as any;
          if (version < 2) {
            persisted.messages = [];
            persisted.currentTopicId = null;
          }
          return persisted as any;
        },
      }
    ),
    { name: 'ChatStore' }
  )
);
