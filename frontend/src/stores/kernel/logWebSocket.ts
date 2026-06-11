import {
  CONNECT_TIMEOUT,
  MAX_RECONNECT_ATTEMPTS,
  PING_INTERVAL,
  RECONNECT_DELAYS,
  WS_URL,
} from '@/stores/kernel/constants';
import type { KernelStore } from '@/types/kernel';

type KernelSet = (partial: Partial<KernelStore>) => void;
type KernelGet = () => KernelStore;
type PingWebSocket = WebSocket & { _pingTimer?: ReturnType<typeof setInterval> };

export function initializeWebSocket(set: KernelSet, get: KernelGet): void {
  let ws: WebSocket;
  try {
    ws = new WebSocket(WS_URL);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'WebSocket initialization failed';
    console.error('[KernelStore] Failed to initialize WebSocket:', error);
    set({
      connection: {
        ...get().connection,
        status: 'error',
        error: message,
      },
    });
    return;
  }

  const connectTimeout = setTimeout(() => {
    if (ws.readyState === WebSocket.CONNECTING) {
      console.error('[KernelStore] WebSocket connection timeout');
      set({
        connection: {
          ...get().connection,
          status: 'error',
          error: 'WebSocket connection timeout',
        },
      });
      ws.close();
    }
  }, CONNECT_TIMEOUT);

  ws.onopen = () => {
    const managedWs = ws as PingWebSocket;
    console.log('[KernelStore] WebSocket connected');
    clearTimeout(connectTimeout);
    set({
      connection: {
        status: 'connected',
        error: null,
        connectedAt: Date.now(),
        reconnectAttempts: 0,
        lastPingTime: Date.now(),
      },
    });

    managedWs._pingTimer = setInterval(() => {
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
  };

  ws.onmessage = (event) => {
    try {
      if (event.data === 'pong') return;

      const logData = JSON.parse(event.data);
      if (!logData.timestamp || !logData.level || !logData.logger) {
        console.warn('[KernelStore] Invalid log entry:', logData);
        return;
      }

      get().addLog(logData);

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
    clearTimeout(connectTimeout);
    set({
      connection: {
        ...get().connection,
        status: 'error',
        error: 'WebSocket connection error. Check backend status or browser security policy.',
      },
    });
  };

  ws.onclose = () => {
    const managedWs = ws as PingWebSocket;
    console.log('[KernelStore] WebSocket closed');
    clearTimeout(connectTimeout);

    if (managedWs._pingTimer) {
      clearInterval(managedWs._pingTimer);
    }

    const state = get();
    if (state._manualDisconnecting) {
      set({
        connection: {
          status: 'disconnected',
          error: null,
          connectedAt: null,
          reconnectAttempts: 0,
          lastPingTime: null,
        },
        _manualDisconnecting: false,
        _ws: null,
        _reconnectTimer: null,
      });
      return;
    }

    const attempts = state.connection.reconnectAttempts;
    if (attempts < MAX_RECONNECT_ATTEMPTS) {
      const delay = RECONNECT_DELAYS[Math.min(attempts, RECONNECT_DELAYS.length - 1)];
      console.log(
        `[KernelStore] Reconnecting in ${delay}ms (attempt ${attempts + 1}/${MAX_RECONNECT_ATTEMPTS})`,
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
