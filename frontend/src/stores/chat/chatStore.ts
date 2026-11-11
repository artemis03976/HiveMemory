/** 会话与流式交付状态。历史正文由后端持有，本地只持久化会话选择与 Agent。 */
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { ChatSSEClient, stopProcess } from '@/services/chatApi';
import * as sessionsApi from '@/services/sessionApi';
import type { SessionSummary, SessionCompactionCheckpoint } from '@/services/sessionApi';
import { sessionCheckpoint, sessionMessages } from './sessionHistory';
import { chatFailureMessage } from './failureMessage';
import { createChatSSECallbacks } from './streamHandlers';
import { applyDone, applyStreamError } from './messageReducers';
import { DEFAULT_AGENT_ID } from '@/constants/identity';
import type { Message, ChatConnectionState, ChatRequestParams, ChatRunStatus, ContextCompactionEvent, DesignatedTopic, MemoryAtom } from '@/types';
import { localErrorText } from '@/transports/connectionStatus';

interface ChatStore {
  messages: Message[];
  connection: ChatConnectionState;
  isStreaming: boolean;
  runStatus: ChatRunStatus;
  lastRunReason: string | null;
  compactionProgress: ContextCompactionEvent | null;
  compactionCheckpoint: SessionCompactionCheckpoint | null;
  currentSessionId: string | null;
  currentAgentId: string;
  retrievedMemories: MemoryAtom[];
  sessions: SessionSummary[];
  sessionError: string | null;
  isLoadingHistory: boolean;
  historyReady: boolean;
  draft: string;
  setDraft: (text: string) => void;
  /** 用户为下一轮指定的话题：发送时随请求带上，发出后清除；记忆图书馆的“在对话中引用”也设定它。 */
  designatedTopic: DesignatedTopic | null;
  setDesignatedTopic: (topic: DesignatedTopic | null) => void;
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
  _historyRevision: number;
}
const errorText = (error: unknown) => error instanceof Error ? error.message : '会话操作失败';

// 后端等交互应用完成没有时限；前端只等到能给出结论为止，超过上限交给用户稍后查看。
const RELEASE_POLL_MS = 1000;
const RELEASE_WAIT_MS = 60_000;

/**
 * 等会话不再被该进程认领：进程写入会话之后才释放认领，释放时这一轮若已保存，历史中就有它。
 * 会话已不存在也算释放，交给读取历史处理；超过上限或读取失败时返回 false，不再猜测结局。
 */
async function waitForSessionRelease(sessionId: string, processId: string): Promise<boolean> {
  const deadline = Date.now() + RELEASE_WAIT_MS;
  for (;;) {
    try {
      const session = await sessionsApi.getSession(sessionId);
      if (session.active_process_id !== processId) return true;
    } catch (error) {
      return error instanceof sessionsApi.SessionApiError && error.status === 404;
    }
    if (Date.now() >= deadline) return false;
    await new Promise(resolve => setTimeout(resolve, RELEASE_POLL_MS));
  }
}

export const useChatStore = create<ChatStore>()(devtools(persist((set, get) => ({
  messages: [], connection: { status: 'disconnected', error: null },
  isStreaming: false, runStatus: 'idle', lastRunReason: null,
  compactionProgress: null, compactionCheckpoint: null,
  currentSessionId: null, currentAgentId: DEFAULT_AGENT_ID, retrievedMemories: [],
  sessions: [], sessionError: null, isLoadingHistory: false, historyReady: false, draft: '', designatedTopic: null,
  _sseClient: null, _currentStreamingMessageId: null, _currentProcessId: null, _listRequest: 0,
  _historyRevision: 0,
  setDraft: (draft) => set({ draft }),
  setDesignatedTopic: (designatedTopic) => set({ designatedTopic }),
  setCurrentAgentId: (currentAgentId) => set({ currentAgentId }),

  refreshSessions: async () => {
    const request = get()._listRequest + 1;
    set({ _listRequest: request });
    try {
      const sessions = await sessionsApi.listSessions();
      // 多个刷新并发时，只接受最后一次请求，避免旧列表覆盖重命名或删除。
      if (get()._listRequest === request) set({ sessions });
    } catch (error) {
      if (get()._listRequest === request) set({ sessionError: localErrorText(errorText(error)) });
    }
  },
  initializeSessions: async () => {
    const id = get().currentSessionId;
    // 先锁住历史读取，恢复完成之前不允许把空界面当成新会话发送。
    await Promise.all([get().refreshSessions(), id ? get().selectSession(id) : Promise.resolve()]);
  },
  selectSession: async (id) => {
    if (get().isStreaming || get().isLoadingHistory) return;
    set({ isLoadingHistory: true, historyReady: false, sessionError: null,
      _historyRevision: get()._historyRevision + 1, compactionProgress: null });
    try {
      const session = await sessionsApi.getSession(id);
      set({ historyReady: true, currentSessionId: id, messages: sessionMessages(session), retrievedMemories: [],
        compactionCheckpoint: sessionCheckpoint(session),
        runStatus: 'idle', lastRunReason: null, connection: { status: 'disconnected', error: null } });
    } catch (error) {
      if (error instanceof sessionsApi.SessionApiError && error.status === 404) {
        set({ currentSessionId: null, messages: [], retrievedMemories: [], compactionCheckpoint: null,
          sessionError: '会话已不存在，请新建或选择会话。' });
        await get().refreshSessions();
      } else set({ sessionError: localErrorText(errorText(error)) });
    } finally { set({ isLoadingHistory: false }); }
  },
  newSession: async () => {
    if (get().isStreaming || get().isLoadingHistory) return;
    set({ isLoadingHistory: true, sessionError: null, _historyRevision: get()._historyRevision + 1 });
    try {
      const session = await sessionsApi.createSession();
      set({ historyReady: true, currentSessionId: session.session_id, messages: [], retrievedMemories: [], runStatus: 'idle', lastRunReason: null,
        compactionCheckpoint: sessionCheckpoint(session), compactionProgress: null });
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
    set({ isLoadingHistory: true, sessionError: null, _historyRevision: get()._historyRevision + 1 });
    try {
      await sessionsApi.deleteSession(id);
      if (get().currentSessionId === id) set({ currentSessionId: null, messages: [], retrievedMemories: [], runStatus: 'idle',
        compactionCheckpoint: null, compactionProgress: null });
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
    const revision = state._historyRevision + 1;
    set({
      messages: [...state.messages,
        { id: userId, role: 'user', content, timestamp: Date.now() },
        { id: assistantMessageId, role: 'assistant', content: '', contentBlocks: [],
          timestamp: Date.now(), isStreaming: true, agent_id: state.currentAgentId }],
      isStreaming: true, runStatus: 'preparing', lastRunReason: null, sessionError: null,
      compactionProgress: null, _historyRevision: revision,
      connection: { status: 'connecting', error: null }, retrievedMemories: [],
      _sseClient: client, _currentStreamingMessageId: assistantMessageId,
    });
    let terminalReceived = false;
    let compactionFinished = false;
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
      setCompactionProgress: (data) => {
        compactionFinished ||= data.status !== 'started';
        set({ compactionProgress: data });
      },
      finalizeSuccess: (data) => {
        terminalReceived = true;
        const status = data.status ?? (data.stopped ? 'cancelled' : 'completed');
        set(s => ({
          // 没有后端输出的轮次不留在消息列表，用户输入回填，便于再次发送。
          messages: data.final_text === undefined
            ? s.messages.filter(m => m.id !== userId && m.id !== assistantMessageId)
            : applyDone(s.messages, assistantMessageId, data.final_text).map(m => m.id === assistantMessageId
              ? { ...m, outcome: status, topicId: data.topic_id,
                interactionId: data.process_id ?? s._currentProcessId ?? undefined,
                failureReason: status === 'failed' ? data.failure_reason : null } : m),
          draft: data.final_text === undefined ? [content, s.draft].filter(Boolean).join('\n') : s.draft,
          isStreaming: false, runStatus: status, lastRunReason: data.failure_reason ?? data.reason ?? null,
          compactionProgress: null,
          connection: { status: status === 'failed' ? 'error' : 'connected', error: status === 'failed' ? chatFailureMessage(data.failure_reason) : null },
          _currentStreamingMessageId: null, _currentProcessId: null,
        }));
        void get().refreshSessions();
      },
      finalizeError: (message, detail) => {
        terminalReceived = true;
        set(s => ({ messages: applyStreamError(s.messages, assistantMessageId, message, detail),
          compactionProgress: null,
          isStreaming: false, runStatus: 'failed', lastRunReason: detail ?? message,
          connection: { status: 'error', error: message }, _currentStreamingMessageId: null, _currentProcessId: null }));
      },
    });
    // 连接在终态前结束：已收到 finalizing 时后端继续提交，提交结束后才写入会话、释放会话认领。
    // 等认领释放后再以服务端历史为准，历史中有这一轮才算已保存；此前结束的轮次没有保存。
    const endWithoutTerminal = async (error: Error) => {
      if (terminalReceived) return;
      const sessionId = get().currentSessionId;
      const processId = get()._currentProcessId;
      if (get().runStatus !== 'finalizing' || !sessionId || !processId) {
        callbacks.onConnectionError(error);
        return;
      }
      terminalReceived = true;
      // 等待期间保持 finalizing 与流式状态，界面不能发送下一轮或切换会话。
      set(s => ({ messages: applyDone(s.messages, assistantMessageId, null), compactionProgress: null }));
      const released = await waitForSessionRelease(sessionId, processId);
      set({ isStreaming: false, runStatus: 'idle', _currentStreamingMessageId: null, _currentProcessId: null });
      if (!released) {
        set({ connection: { status: 'error', error: '连接中断，暂时无法确认本轮是否已保存，稍后重新选择会话查看' } });
        return;
      }
      await get().selectSession(sessionId);
      const saved = get().messages.some(m => m.interactionId === processId);
      // 没有保存的轮次与无输出的轮次一样回填输入，便于再次发送。
      set(s => ({ connection: { status: 'error', error: saved ? '连接中断，本轮已保存到会话' : '连接中断，本轮未保存' },
        draft: saved ? s.draft : [content, s.draft].filter(Boolean).join('\n') }));
      void get().refreshSessions();
    };
    try {
      await client.connect({ ...options, message: content, session_id: state.currentSessionId, agent_id: state.currentAgentId }, callbacks);
      await endWithoutTerminal(new Error('连接提前结束，本轮未保存'));
      const sessionId = get().currentSessionId;
      if (compactionFinished && sessionId && get()._historyRevision === revision) {
        // done 不携带检查点；只显示会话服务真正接受的结果，晚到读取不能污染新视图。
        try {
          const session = await sessionsApi.getSession(sessionId);
          if (get()._historyRevision === revision && get().currentSessionId === sessionId) {
            set({ compactionCheckpoint: sessionCheckpoint(session) });
          }
        } catch {
          if (get()._historyRevision === revision) set({ sessionError: '压缩标记读取失败，可重新选择会话刷新。' });
        }
      }
    } catch (error) {
      await endWithoutTerminal(error instanceof Error ? error : new Error(String(error)));
    } finally {
      client.disconnect();
      // 终态之后可能立即开始下一轮，旧连接不能清掉新连接的引用。
      if (get()._sseClient === client) set({ _sseClient: null });
    }
  },
  clearMessages: () => {
    if (get().isStreaming || get().isLoadingHistory) return;
    set({ messages: [], currentSessionId: null, retrievedMemories: [], runStatus: 'idle', lastRunReason: null,
      compactionProgress: null, compactionCheckpoint: null, _historyRevision: get()._historyRevision + 1 });
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
