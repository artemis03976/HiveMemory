import {
  CONNECT_TIMEOUT,
  MAX_RECONNECT_ATTEMPTS,
  PING_INTERVAL,
  RECONNECT_DELAYS,
  WS_URL,
} from '@/stores/kernel/constants';
import { markActivity, markConnected, markDisconnected, markError, markReconnecting } from '@/transports/state';
import { WebSocketClient } from '@/transports/websocket/websocketClient';
import type { KernelStore } from '@/types/kernel';

type KernelSet = (partial: Partial<KernelStore>) => void;
type KernelGet = () => KernelStore;
type LogPayload = Omit<KernelStore['logs'][number], 'id'>;

const websocketClient = new WebSocketClient<LogPayload>();

export function initializeWebSocket(set: KernelSet, get: KernelGet): void {
  const ws = websocketClient.connect({
    url: WS_URL,
    connectTimeoutMs: CONNECT_TIMEOUT,
    pingIntervalMs: PING_INTERVAL,
    pingMessage: 'ping',
    pongMessage: 'pong',
    retryPolicy: {
      maxAttempts: MAX_RECONNECT_ATTEMPTS,
      delays: RECONNECT_DELAYS,
    },
    getReconnectAttempts: () => get().connection.reconnectAttempts,
    shouldReconnect: () => get()._isPrimaryWindow,
    parseMessage: (data) => {
      const logData = JSON.parse(data);
      if (!logData.timestamp || !logData.level || !logData.logger) {
        console.warn('[KernelStore] Invalid log entry:', logData);
        return null;
      }
      return logData as LogPayload;
    },
    onOpen: () => {
      console.log('[KernelStore] WebSocket connected');
      const connected = markConnected(get().connection);
      set({
        connection: {
          ...connected,
          lastPingTime: connected.lastActivityAt,
        },
      });
    },
    onPong: () => {
      const active = markActivity(get().connection);
      set({
        connection: {
          ...active,
          lastPingTime: active.lastActivityAt,
        },
      });
    },
    onMessage: (logData) => {
      set({
        connection: markActivity(get().connection),
      });

      get().addLog(logData);

      const state = get();
      if (state._isPrimaryWindow && state._broadcastChannel) {
        state._broadcastChannel.postMessage({
          type: 'NEW_LOG',
          payload: logData,
        });
      }
    },
    onError: (message, error) => {
      console.error('[KernelStore] WebSocket error:', error ?? message);
      set({
        connection: markError(get().connection, message),
      });
    },
    onClosedByRequest: () => {
      console.log('[KernelStore] WebSocket closed');
      set({
        connection: {
          ...markDisconnected(get().connection),
          lastPingTime: null,
        },
        _manualDisconnecting: false,
        _ws: null,
        _reconnectTimer: null,
      });
    },
    onReconnecting: (attempt, delay) => {
      console.log(
        `[KernelStore] Reconnecting in ${delay}ms (attempt ${attempt}/${MAX_RECONNECT_ATTEMPTS})`,
      );

      set({
        connection: markReconnecting(get().connection, attempt),
      });
    },
    onReconnect: () => {
      if (get()._isPrimaryWindow) {
        initializeWebSocket(set, get);
      }
    },
    onReconnectExhausted: () => {
      set({
        connection: markError(get().connection, 'Max reconnection attempts reached'),
      });
    },
  });

  set({ _ws: ws });
}

export function disconnectWebSocket(): void {
  websocketClient.disconnect();
}

export function clearWebSocketReconnectTimer(): void {
  websocketClient.clearReconnectTimer();
}
