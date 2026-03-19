/**
 * Kernel Store - Zustand state management for kernel terminal
 *
 * Features:
 * - WebSocket connection to backend log streaming
 * - BroadcastChannel for cross-window synchronization
 * - Primary/secondary window election
 * - Automatic reconnection with exponential backoff
 * - Log buffering and filtering
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type {
  KernelStore,
  LogEntry,
  BroadcastMessage,
} from '@/types/kernel';

// Configuration constants
const WS_URL = import.meta.env.DEV
  ? 'ws://localhost:8769/api/v1/ws/logs'  // Custom port to avoid Windows reserved ranges
  : `ws://${window.location.host}/api/v1/ws/logs`;

const BROADCAST_CHANNEL_NAME = 'hivememory_kernel_logs';
const PRIMARY_WINDOW_KEY = 'hivememory_primary_window';
const HEARTBEAT_INTERVAL = 5000; // 5 seconds
const MAX_RECONNECT_ATTEMPTS = 10;
const RECONNECT_DELAYS = [1000, 2000, 4000, 8000, 16000, 30000]; // Exponential backoff
const PING_INTERVAL = 30000; // 30 seconds

/**
 * Check if this window should be the primary window
 * Primary window maintains the WebSocket connection
 */
function checkAndClaimPrimaryWindow(): boolean {
  const now = Date.now();
  const stored = localStorage.getItem(PRIMARY_WINDOW_KEY);

  if (!stored) {
    // No primary window exists, claim it
    localStorage.setItem(PRIMARY_WINDOW_KEY, now.toString());
    return true;
  }

  const lastHeartbeat = parseInt(stored, 10);
  if (now - lastHeartbeat > HEARTBEAT_INTERVAL * 2) {
    // Primary window is dead, take over
    localStorage.setItem(PRIMARY_WINDOW_KEY, now.toString());
    return true;
  }

  return false;
}

/**
 * Initialize WebSocket connection (primary window only)
 */
function initializeWebSocket(
  set: (partial: Partial<KernelStore>) => void,
  get: () => KernelStore
) {
  const ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    console.log('[KernelStore] WebSocket connected');
    set({
      connection: {
        status: 'connected',
        error: null,
        connectedAt: Date.now(),
        reconnectAttempts: 0,
        lastPingTime: Date.now(),
      },
    });

    // Start ping interval
    const pingTimer = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send('ping');
        set({
          connection: {
            ...get().connection,
            lastPingTime: Date.now(),
          },
        });
      }
    }, PING_INTERVAL);

    // Store ping timer for cleanup
    (ws as any)._pingTimer = pingTimer;
  };

  ws.onmessage = (event) => {
    try {
      // Handle pong response
      if (event.data === 'pong') {
        return;
      }

      const logData = JSON.parse(event.data);

      // Validate log structure
      if (!logData.timestamp || !logData.level || !logData.logger) {
        console.warn('[KernelStore] Invalid log entry:', logData);
        return;
      }

      // Add log to store
      get().addLog(logData);

      // Broadcast to other windows
      const state = get();
      if (state._isPrimaryWindow && state._broadcastChannel) {
        state._broadcastChannel.postMessage({
          type: 'NEW_LOG',
          payload: logData,
        });
      }
    } catch (error) {
      console.error('[KernelStore] Failed to parse WebSocket message:', error);
    }
  };

  ws.onerror = (error) => {
    console.error('[KernelStore] WebSocket error:', error);
    set({
      connection: {
        ...get().connection,
        status: 'error',
        error: 'WebSocket connection error',
      },
    });
  };

  ws.onclose = () => {
    console.log('[KernelStore] WebSocket closed');

    // Clear ping timer
    if ((ws as any)._pingTimer) {
      clearInterval((ws as any)._pingTimer);
    }

    const state = get();
    const attempts = state.connection.reconnectAttempts;

    if (attempts < MAX_RECONNECT_ATTEMPTS) {
      // Schedule reconnection
      const delay =
        RECONNECT_DELAYS[Math.min(attempts, RECONNECT_DELAYS.length - 1)];

      console.log(
        `[KernelStore] Reconnecting in ${delay}ms (attempt ${attempts + 1}/${MAX_RECONNECT_ATTEMPTS})`
      );

      set({
        connection: {
          ...state.connection,
          status: 'reconnecting',
          reconnectAttempts: attempts + 1,
        },
      });

      const timer = setTimeout(() => {
        if (get()._isPrimaryWindow) {
          initializeWebSocket(set, get);
        }
      }, delay);

      set({ _reconnectTimer: timer });
    } else {
      set({
        connection: {
          ...state.connection,
          status: 'error',
          error: 'Max reconnection attempts reached',
        },
      });
    }
  };

  set({ _ws: ws });
}

/**
 * Handle BroadcastChannel messages (all windows)
 */
function handleBroadcastMessage(
  message: BroadcastMessage,
  set: (partial: Partial<KernelStore>) => void,
  get: () => KernelStore
) {
  switch (message.type) {
    case 'NEW_LOG':
      get().addLog(message.payload);
      break;

    case 'BATCH_LOGS':
      get().addLogs(message.payload);
      break;

    case 'CLEAR_LOGS':
      set({ logs: [], stats: { ...get().stats, totalLogs: 0, filteredCount: 0 } });
      break;

    case 'CONNECTION_STATE':
      set({ connection: message.payload });
      break;

    case 'FILTER_UPDATE':
      set({ filters: message.payload });
      break;

    case 'UI_UPDATE':
      set({ ui: { ...get().ui, ...message.payload } });
      break;

    case 'REQUEST_SYNC':
      // Primary window responds with full state
      if (get()._isPrimaryWindow && get()._broadcastChannel) {
        const state = get();
        state._broadcastChannel!.postMessage({
          type: 'FULL_SYNC',
          payload: {
            logs: state.logs,
            filters: state.filters,
            ui: state.ui,
          },
        });
      }
      break;

    case 'FULL_SYNC':
      // Secondary window receives full state
      if (!get()._isPrimaryWindow) {
        set({
          logs: message.payload.logs,
          filters: message.payload.filters,
          ui: message.payload.ui,
        });
      }
      break;

    case 'PRIMARY_HEARTBEAT':
      // Update primary window timestamp
      localStorage.setItem(PRIMARY_WINDOW_KEY, message.timestamp.toString());
      break;
  }
}

/**
 * Start primary window heartbeat
 */
function startPrimaryHeartbeat(get: () => KernelStore) {
  const heartbeat = setInterval(() => {
    const state = get();
    if (state._isPrimaryWindow) {
      const now = Date.now();
      localStorage.setItem(PRIMARY_WINDOW_KEY, now.toString());

      // Broadcast heartbeat to other windows
      if (state._broadcastChannel) {
        state._broadcastChannel.postMessage({
          type: 'PRIMARY_HEARTBEAT',
          timestamp: now,
        });
      }
    }
  }, HEARTBEAT_INTERVAL);

  return heartbeat;
}

/**
 * Create Zustand store
 */
export const useKernelStore = create<KernelStore>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        logs: [],
        connection: {
          status: 'disconnected',
          error: null,
          connectedAt: null,
          reconnectAttempts: 0,
          lastPingTime: null,
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

        // Internal state
        _ws: null,
        _broadcastChannel: null,
        _isPrimaryWindow: false,
        _reconnectTimer: null,
        _statsUpdateTimer: null,

        // Computed selector
        filteredLogs: () => {
          const { logs, filters } = get();
          return logs.filter((log) => {
            // Level filter
            if (filters.logLevel && log.level !== filters.logLevel) {
              return false;
            }

            // Logger namespace filter
            if (
              filters.loggerNamespace &&
              !log.logger.startsWith(filters.loggerNamespace)
            ) {
              return false;
            }

            // Search text filter
            if (filters.searchText) {
              const search = filters.searchText.toLowerCase();
              return (
                log.message.toLowerCase().includes(search) ||
                log.logger.toLowerCase().includes(search)
              );
            }

            return true;
          });
        },

        // Connection actions
        connect: () => {
          const state = get();
          if (state._ws?.readyState === WebSocket.OPEN) {
            console.log('[KernelStore] Already connected');
            return;
          }

          console.log('[KernelStore] Connecting...');
          set({
            connection: {
              ...state.connection,
              status: 'connecting',
            },
          });

          // Determine if this is the primary window
          const isPrimary = checkAndClaimPrimaryWindow();
          set({ _isPrimaryWindow: isPrimary });

          console.log(
            `[KernelStore] Window role: ${isPrimary ? 'PRIMARY' : 'SECONDARY'}`
          );

          // Initialize BroadcastChannel for all windows
          if (!state._broadcastChannel) {
            const bc = new BroadcastChannel(BROADCAST_CHANNEL_NAME);
            bc.onmessage = (event) => handleBroadcastMessage(event.data, set, get);
            set({ _broadcastChannel: bc });
            console.log('[KernelStore] BroadcastChannel initialized');
          }

          if (isPrimary) {
            // Primary window: establish WebSocket
            initializeWebSocket(set, get);

            // Start heartbeat
            const heartbeatTimer = startPrimaryHeartbeat(get);
            set({ _statsUpdateTimer: heartbeatTimer });
          } else {
            // Secondary window: request full sync from primary
            const bc = get()._broadcastChannel;
            if (bc) {
              bc.postMessage({ type: 'REQUEST_SYNC' });
              console.log('[KernelStore] Requested full sync from primary');
            }

            // Set connection status to connected (via BroadcastChannel)
            set({
              connection: {
                ...state.connection,
                status: 'connected',
                connectedAt: Date.now(),
              },
            });
          }
        },

        disconnect: () => {
          console.log('[KernelStore] Disconnecting...');
          const state = get();

          // Close WebSocket
          if (state._ws) {
            state._ws.close();
            set({ _ws: null });
          }

          // Close BroadcastChannel
          if (state._broadcastChannel) {
            state._broadcastChannel.close();
            set({ _broadcastChannel: null });
          }

          // Clear timers
          if (state._reconnectTimer) {
            clearTimeout(state._reconnectTimer);
            set({ _reconnectTimer: null });
          }
          if (state._statsUpdateTimer) {
            clearInterval(state._statsUpdateTimer);
            set({ _statsUpdateTimer: null });
          }

          // Clear primary window flag
          if (state._isPrimaryWindow) {
            localStorage.removeItem(PRIMARY_WINDOW_KEY);
          }

          set({
            connection: {
              status: 'disconnected',
              error: null,
              connectedAt: null,
              reconnectAttempts: 0,
              lastPingTime: null,
            },
            _isPrimaryWindow: false,
          });
        },

        reconnect: () => {
          console.log('[KernelStore] Manual reconnect triggered');
          get().disconnect();
          setTimeout(() => get().connect(), 1000);
        },

        // Log actions
        addLog: (log) => {
          const state = get();
          if (state.ui.isPaused) return;

          const newLog: LogEntry = {
            ...log,
            id: crypto.randomUUID(),
          };

          set((state) => {
            let newLogs = [...state.logs, newLog];

            // Enforce buffer size limit
            if (newLogs.length > state.ui.maxBufferSize) {
              newLogs = newLogs.slice(-state.ui.maxBufferSize);
            }

            return {
              logs: newLogs,
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
            let allLogs = [...state.logs, ...newLogs];

            // Enforce buffer size limit
            if (allLogs.length > state.ui.maxBufferSize) {
              allLogs = allLogs.slice(-state.ui.maxBufferSize);
            }

            return {
              logs: allLogs,
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
            stats: {
              totalLogs: 0,
              filteredCount: 0,
              connectionUptime: get().stats.connectionUptime,
              logsPerSecond: 0,
            },
          });

          // Broadcast clear to other windows
          const state = get();
          if (state._isPrimaryWindow && state._broadcastChannel) {
            state._broadcastChannel.postMessage({ type: 'CLEAR_LOGS' });
          }
        },

        // Filter actions
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

        // UI actions
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
      }),
      {
        name: 'kernel-store',
        partialize: (state) => ({
          // Only persist UI preferences
          filters: state.filters,
          ui: state.ui,
        }),
      }
    ),
    { name: 'KernelStore' }
  )
);
