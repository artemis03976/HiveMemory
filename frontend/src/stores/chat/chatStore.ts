/** 会话与流式交付状态。历史正文由后端持有，本地只持久化会话选择与 Agent。 */
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { ChatSSEClient, stopProcess } from '@/services/chatApi';
import * as sessionsApi from '@/services/sessionApi';
import type { SessionSummary } from '@/services/sessionApi';
import { sessionMessages } from './sessionHistory';
import { createChatSSECallbacks } from './streamHandlers';
import { applyDone, applyStreamError } from './messageReducers';
import { DEFAULT_AGENT_ID } from '@/constants/identity';
import type { Message, ChatConnectionState, ChatRequestParams, ChatRunStatus, MemoryAtom } from '@/types';

interface ChatStore {
  messages: Message[];
  connection: ChatConnectionState;
  isStreaming: boolean;
  runStatus: ChatRunStatus;
  lastRunReason: string | null;
  currentSessionId: string | null;
  currentAgentId: string;
  retrievedMemories: MemoryAtom[];
  sessions: SessionSummary[];
  sessionError: string | null;
  isLoadingHistory: boolean;
  historyReady: boolean;
  draft: string;
  setDraft: (text: string) => void;
  initializeSessions: () => Promise<void>;
  refreshSessions: () => Promise<void>;
  selectSession: (id: string) => Promise<void>;
  newSession: () => Promise<void>;
  renameSession: (id: string, title: string) => Promise<void>;
  deleteSession: (id: string) => Promise<void>;
  sendMessage: (content: string, options?: Partial<ChatRequestParams>) => Promise<void>;
  stopStreaming: () => void;
  clearMessages: () => void;
  retryMessage: (messageId: string) => Promise<void>;
  setCurrentAgentId: (id: string) => void;
  _sseClient: ChatSSEClient | null;
  _currentStreamingMessageId: string | null;
  _currentProcessId: string | null;
  _listRequest: number;
}
const errorText = (error: unknown) => error instanceof Error ? error.message : '会话操作失败';

export const useChatStore = create<ChatStore>()(devtools(persist((set, get) => ({
  messages: [], connection: { status: 'disconnected', error: null },
  isStreaming: false, runStatus: 'idle', lastRunReason: null,
  currentSessionId: null, currentAgentId: DEFAULT_AGENT_ID, retrievedMemories: [],
  sessions: [], sessionError: null, isLoadingHistory: false, historyReady: false, draft: '',
  _sseClient: null, _currentStreamingMessageId: null, _currentProcessId: null, _listRequest: 0,
  setDraft: (draft) => set({ draft }),
  setCurrentAgentId: (currentAgentId) => set({ currentAgentId }),

  refreshSessions: async () => {
    const request = get()._listRequest + 1;
    set({ _listRequest: request });
    try {
      const sessions = await sessionsApi.listSessions();
      // 多个刷新并发时，只接受最后一次请求，避免旧列表覆盖重命名或删除。
      if (get()._listRequest === request) set({ sessions });
    } catch (error) {
      if (get()._listRequest === request) set({ sessionError: errorText(error) });
    }
  },
  initializeSessions: async () => {
    const id = get().currentSessionId;
    // 先锁住历史读取，恢复完成之前不允许把空界面当成新会话发送。
    await Promise.all([get().refreshSessions(), id ? get().selectSession(id) : Promise.resolve()]);
  },
  selectSession: async (id) => {
    if (get().isStreaming || get().isLoadingHistory) return;
    set({ isLoadingHistory: true, historyReady: false, sessionError: null });
    try {
      const session = await sessionsApi.getSession(id);
      set({ historyReady: true, currentSessionId: id, messages: sessionMessages(session), retrievedMemories: [],
        runStatus: 'idle', lastRunReason: null, connection: { status: 'disconnected', error: null } });
    } catch (error) {
      if (error instanceof sessionsApi.SessionApiError && error.status === 404) {
        set({ currentSessionId: null, messages: [], retrievedMemories: [], sessionError: '会话已不存在，请新建或选择会话。' });
        await get().refreshSessions();
      } else set({ sessionError: errorText(error) });
    } finally { set({ isLoadingHistory: false }); }
  },
  newSession: async () => {
    if (get().isStreaming || get().isLoadingHistory) return;
    set({ isLoadingHistory: true, sessionError: null });
    try {
      const session = await sessionsApi.createSession();
      set({ historyReady: true, currentSessionId: session.session_id, messages: [], retrievedMemories: [], runStatus: 'idle', lastRunReason: null });
      await get().refreshSessions();
    } catch (error) { set({ sessionError: errorText(error) }); }
    finally { set({ isLoadingHistory: false }); }
  },
  renameSession: async (id, title) => {
    if (get().isStreaming || get().isLoadingHistory) return;
    set({ isLoadingHistory: true, sessionError: null });
    try { await sessionsApi.renameSession(id, title); await get().refreshSessions(); }
    catch (error) { set({ sessionError: errorText(error) }); }
    finally { set({ isLoadingHistory: false }); }
  },
  deleteSession: async (id) => {
    if (get().isStreaming || get().isLoadingHistory) return;
    set({ isLoadingHistory: true, sessionError: null });
    try {
      await sessionsApi.deleteSession(id);
      if (get().currentSessionId === id) set({ currentSessionId: null, messages: [], retrievedMemories: [], runStatus: 'idle' });
      await get().refreshSessions();
    } catch (error) { set({ sessionError: errorText(error) }); }
    finally { set({ isLoadingHistory: false }); }
  },
  stopStreaming: () => {
    const state = get();
    if (!state.isStreaming || ['cancelling', 'finalizing'].includes(state.runStatus)) return;
    set({ runStatus: 'cancelling', lastRunReason: 'user_requested' });
    if (state._currentProcessId) void stopProcess(state._currentProcessId).catch(error => {
      // stop 响应可能晚于 done，不能覆盖终态或另一轮的状态。
      if (get()._currentProcessId !== state._currentProcessId || get().runStatus !== 'cancelling') return;
      set({ connection: { status: 'error', error: errorText(error) }, runStatus: 'streaming' });
    });
  },
  sendMessage: async (content, options = {}) => {
    const state = get();
    if (state.isStreaming || state.isLoadingHistory || (state.currentSessionId && !state.historyReady)) return;
    const userId = crypto.randomUUID();
    const assistantMessageId = crypto.randomUUID();
    const client = new ChatSSEClient();
    set({
      messages: [...state.messages,
        { id: userId, role: 'user', content, timestamp: Date.now() },
        { id: assistantMessageId, role: 'assistant', content: '', contentBlocks: [],
          timestamp: Date.now(), isStreaming: true, agent_id: state.currentAgentId }],
      isStreaming: true, runStatus: 'preparing', lastRunReason: null, sessionError: null,
      connection: { status: 'connecting', error: null }, retrievedMemories: [],
      _sseClient: client, _currentStreamingMessageId: assistantMessageId,
    });
    let terminalReceived = false;
    const callbacks = createChatSSECallbacks({
      assistantMessageId,
      updateMessages: (updater) => set(s => ({ messages: updater(s.messages) })),
      setRetrievedMemories: (retrievedMemories) => set({ retrievedMemories }),
      handleCommandResult: (data) => set(s => ({ messages: applyDone(s.messages, assistantMessageId, data.message) })),
      setProcessId: (data) => {
        set({ _currentProcessId: data.process_id, currentSessionId: data.session_id, historyReady: true });
        if (get().runStatus === 'cancelling') void stopProcess(data.process_id).catch(error => {
          if (get()._currentProcessId !== data.process_id || get().runStatus !== 'cancelling') return;
          set({ runStatus: 'streaming', connection: { status: 'error', error: errorText(error) } });
        });
      },
      markStreaming: () => {
        if (get().runStatus === 'preparing') set({ runStatus: 'streaming', connection: { status: 'connected', error: null } });
      },
      setRunStatus: (data) => set({ runStatus: data.status, lastRunReason: data.reason ?? null }),
      finalizeSuccess: (data) => {
        terminalReceived = true;
        const status = data.status ?? (data.stopped ? 'cancelled' : 'completed');
        set(s => ({
          // 没有后端输出的轮次不留在消息列表，用户输入回填，便于再次发送。
          messages: data.final_text === undefined
            ? s.messages.filter(m => m.id !== userId && m.id !== assistantMessageId)
            : applyDone(s.messages, assistantMessageId, data.final_text).map(m => m.id === assistantMessageId
              ? { ...m, outcome: status, topicId: data.topic_id } : m),
          draft: data.final_text === undefined ? [content, s.draft].filter(Boolean).join('\n') : s.draft,
          isStreaming: false, runStatus: status, lastRunReason: data.reason ?? null,
          connection: { status: status === 'failed' ? 'error' : 'connected', error: status === 'failed' ? '本轮执行失败' : null },
          _currentStreamingMessageId: null, _currentProcessId: null,
        }));
        void get().refreshSessions();
      },
      finalizeError: (message, detail) => {
        terminalReceived = true;
        set(s => ({ messages: applyStreamError(s.messages, assistantMessageId, message, detail),
          isStreaming: false, runStatus: 'failed', lastRunReason: detail ?? message,
          connection: { status: 'error', error: message }, _currentStreamingMessageId: null, _currentProcessId: null }));
      },
    });
    try {
      await client.connect({ ...options, message: content, session_id: state.currentSessionId, agent_id: state.currentAgentId }, callbacks);
      if (!terminalReceived) callbacks.onConnectionError(new Error('连接提前结束，本轮未保存'));
    } catch (error) {
      if (!terminalReceived) callbacks.onConnectionError(error instanceof Error ? error : new Error(String(error)));
    } finally {
      client.disconnect();
      // 终态之后可能立即开始下一轮，旧连接不能清掉新连接的引用。
      if (get()._sseClient === client) set({ _sseClient: null });
    }
  },
  clearMessages: () => {
    if (get().isStreaming || get().isLoadingHistory) return;
    set({ messages: [], currentSessionId: null, retrievedMemories: [], runStatus: 'idle', lastRunReason: null });
  },
  retryMessage: async (id) => {
    const state = get();
    if (state.isStreaming || state.isLoadingHistory) return;
    const index = state.messages.findIndex(m => m.id === id);
    const user = state.messages.slice(0, index).reverse().find(m => m.role === 'user');
    if (index >= 0 && user) await get().sendMessage(user.content);
  },
}), {
  name: 'chat-store', version: 4,
  partialize: state => ({ currentSessionId: state.currentSessionId, currentAgentId: state.currentAgentId }),
  migrate: persisted => {
    const old = persisted as Partial<ChatStore> | null;
    return { currentSessionId: null, currentAgentId: old?.currentAgentId && old.currentAgentId !== 'default' ? old.currentAgentId : DEFAULT_AGENT_ID };
  },
}), { name: 'ChatStore' }));
