/**
 * Kernel Store - Zustand state management for kernel terminal.
 *
 * The store keeps the public state/actions in one place while connection
 * lifecycles and pure reducers live in focused kernel modules.
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { checkAndClaimPrimaryWindow, handleBroadcastMessage, startPrimaryHeartbeat } from '@/stores/kernel/broadcastSync';
import { BROADCAST_CHANNEL_NAME, PRIMARY_WINDOW_KEY } from '@/stores/kernel/constants';
import { appendBounded, groupLogsByTrace, toggleSpanCollapseInGroups, toggleTraceCollapseInGroups } from '@/stores/kernel/logReducers';
import { clearWebSocketReconnectTimer, disconnectWebSocket, initializeWebSocket } from '@/stores/kernel/logWebSocket';
import { initializeRuntimeEventStream } from '@/stores/kernel/runtimeEventStream';
import { filterLogs } from '@/stores/kernel/selectors';
import { markConnected, markConnecting, markDisconnected, markError } from '@/transports/state';
import { useMemoryTaskStore } from '@/stores/memory';
import type { KernelStore, LogEntry } from '@/types/kernel';

export const useKernelStore = create<KernelStore>()(
  devtools(
    persist(
      (set, get) => ({
        logs: [],
        runtimeEvents: [],
        traceGroups: new Map(),
        connection: {
          status: 'disconnected',
          error: null,
          connectedAt: null,
          reconnectAttempts: 0,
          lastPingTime: null,
        },
        runtimeEventConnection: {
          status: 'disconnected',
          error: null,
          connectedAt: null,
          lastEventId: null,
        },
        filters: {
          logLevel: null,
          loggerNamespace: '',
          searchText: '',
        },
        ui: {
          autoScroll: true,
          isPaused: false,
          maxBufferSize: 1000,
        },
        stats: {
          totalLogs: 0,
          filteredCount: 0,
          connectionUptime: 0,
          logsPerSecond: 0,
        },

        _ws: null,
        _eventSource: null,
        _broadcastChannel: null,
        _isPrimaryWindow: false,
        _reconnectTimer: null,
        _statsUpdateTimer: null,
        _manualDisconnecting: false,

        filteredLogs: () => filterLogs(get().logs, get().filters),

        connect: () => {
          const state = get();
          if (state._ws?.readyState === WebSocket.OPEN || state._ws?.readyState === WebSocket.CONNECTING) {
            console.log('[KernelStore] Already connected');
            return;
          }

          console.log('[KernelStore] Connecting...');
          set({
            connection: markConnecting(state.connection),
            _manualDisconnecting: false,
          });

          try {
            const isPrimary = checkAndClaimPrimaryWindow();
            set({ _isPrimaryWindow: isPrimary });

            console.log(`[KernelStore] Window role: ${isPrimary ? 'PRIMARY' : 'SECONDARY'}`);

            if (!state._broadcastChannel) {
              const bc = new BroadcastChannel(BROADCAST_CHANNEL_NAME);
              bc.onmessage = (event) => handleBroadcastMessage(event.data, set, get);
              set({ _broadcastChannel: bc });
              console.log('[KernelStore] BroadcastChannel initialized');
            }

            if (isPrimary) {
              initializeWebSocket(set, get);
              initializeRuntimeEventStream(set, get);
              const heartbeatTimer = startPrimaryHeartbeat(get);
              set({ _statsUpdateTimer: heartbeatTimer });
            } else {
              const bc = get()._broadcastChannel;
              if (bc) {
                bc.postMessage({ type: 'REQUEST_SYNC' });
                console.log('[KernelStore] Requested full sync from primary');
              }

              set({
                connection: {
                  ...markConnected(state.connection),
                  lastPingTime: state.connection.lastPingTime,
                },
                runtimeEventConnection: {
                  ...markConnected(state.runtimeEventConnection),
                  lastEventId: state.runtimeEventConnection.lastEventId,
                },
              });
            }
          } catch (error) {
            const message = error instanceof Error ? error.message : 'Kernel connection initialization failed';
            console.error('[KernelStore] Failed to start connection:', error);
            set({
              connection: markError(get().connection, message),
            });
          }
        },

        disconnect: () => {
          console.log('[KernelStore] Disconnecting...');
          const state = get();
          const hasWs = Boolean(state._ws);
          if (hasWs) {
            set({ _manualDisconnecting: true });
          }

          if (state._ws) {
            disconnectWebSocket();
            set({ _ws: null });
          }

          if (state._eventSource) {
            state._eventSource.close();
            set({ _eventSource: null });
          }

          if (state._broadcastChannel) {
            state._broadcastChannel.close();
            set({ _broadcastChannel: null });
          }

          if (state._reconnectTimer) {
            clearWebSocketReconnectTimer();
            set({ _reconnectTimer: null });
          }
          if (state._statsUpdateTimer) {
            clearInterval(state._statsUpdateTimer);
            set({ _statsUpdateTimer: null });
          }

          if (state._isPrimaryWindow) {
            localStorage.removeItem(PRIMARY_WINDOW_KEY);
          }

          set({
            connection: {
              ...markDisconnected(state.connection),
              lastPingTime: null,
            },
            _isPrimaryWindow: false,
            _manualDisconnecting: hasWs,
            runtimeEventConnection: {
              ...markDisconnected(state.runtimeEventConnection),
              lastEventId: state.runtimeEventConnection.lastEventId,
            },
          });
        },

        reconnect: () => {
          console.log('[KernelStore] Manual reconnect triggered');
          get().disconnect();
          setTimeout(() => get().connect(), 1000);
        },

        connectRuntimeEvents: () => {
          initializeRuntimeEventStream(set, get);
        },

        disconnectRuntimeEvents: () => {
          const state = get();
          if (state._eventSource) {
            state._eventSource.close();
          }
          set({
            _eventSource: null,
            runtimeEventConnection: {
              ...markDisconnected(state.runtimeEventConnection),
              lastEventId: state.runtimeEventConnection.lastEventId,
            },
          });
        },

        addLog: (log) => {
          const state = get();
          if (state.ui.isPaused) return;

          const newLog: LogEntry = {
            ...log,
            id: crypto.randomUUID(),
          };

          set((state) => {
            const logs = appendBounded(state.logs, [newLog], state.ui.maxBufferSize);
            return {
              logs,
              traceGroups: groupLogsByTrace(logs),
              stats: {
                ...state.stats,
                totalLogs: state.stats.totalLogs + 1,
              },
            };
          });
        },

        addLogs: (logs) => {
          const state = get();
          if (state.ui.isPaused) return;

          const newLogs: LogEntry[] = logs.map((log) => ({
            ...log,
            id: crypto.randomUUID(),
          }));

          set((state) => {
            const allLogs = appendBounded(state.logs, newLogs, state.ui.maxBufferSize);
            return {
              logs: allLogs,
              traceGroups: groupLogsByTrace(allLogs),
              stats: {
                ...state.stats,
                totalLogs: state.stats.totalLogs + newLogs.length,
              },
            };
          });
        },

        clearLogs: () => {
          set({
            logs: [],
            traceGroups: new Map(),
            stats: {
              totalLogs: 0,
              filteredCount: 0,
              connectionUptime: get().stats.connectionUptime,
              logsPerSecond: 0,
            },
          });

          const state = get();
          if (state._isPrimaryWindow && state._broadcastChannel) {
            state._broadcastChannel.postMessage({ type: 'CLEAR_LOGS' });
          }
        },

        addRuntimeEvent: (event) => {
          const state = get();
          useMemoryTaskStore.getState().applyRuntimeEvent(event);
          if (state.ui.isPaused) return;

          set((state) => {
            const runtimeEvents = appendBounded(state.runtimeEvents, [event], state.ui.maxBufferSize);
            return {
              runtimeEvents,
              runtimeEventConnection: {
                ...state.runtimeEventConnection,
                lastEventId: event.event_id,
              },
            };
          });
        },

        clearRuntimeEvents: () => {
          set({ runtimeEvents: [] });
          const state = get();
          if (state._isPrimaryWindow && state._broadcastChannel) {
            state._broadcastChannel.postMessage({ type: 'CLEAR_RUNTIME_EVENTS' });
          }
        },

        setLogLevel: (level) => {
          set((state) => ({
            filters: { ...state.filters, logLevel: level },
          }));
        },

        setLoggerNamespace: (namespace) => {
          set((state) => ({
            filters: { ...state.filters, loggerNamespace: namespace },
          }));
        },

        setSearchText: (text) => {
          set((state) => ({
            filters: { ...state.filters, searchText: text },
          }));
        },

        clearFilters: () => {
          set({
            filters: {
              logLevel: null,
              loggerNamespace: '',
              searchText: '',
            },
          });
        },

        toggleAutoScroll: () => {
          set((state) => ({
            ui: { ...state.ui, autoScroll: !state.ui.autoScroll },
          }));
        },

        togglePause: () => {
          set((state) => ({
            ui: { ...state.ui, isPaused: !state.ui.isPaused },
          }));
        },

        setMaxBufferSize: (size) => {
          set((state) => ({
            ui: { ...state.ui, maxBufferSize: size },
          }));
        },

        toggleTraceCollapse: (trace_id) => {
          set((state) => ({ traceGroups: toggleTraceCollapseInGroups(state.traceGroups, trace_id) }));
        },

        toggleSpanCollapse: (trace_id, span_name) => {
          set((state) => ({ traceGroups: toggleSpanCollapseInGroups(state.traceGroups, trace_id, span_name) }));
        },
      }),
      {
        name: 'kernel-store',
        partialize: (state) => ({
          filters: state.filters,
          ui: state.ui,
        }),
      },
    ),
    { name: 'KernelStore' },
  ),
);
