import {
  CONNECT_TIMEOUT,
  MAX_RECONNECT_ATTEMPTS,
  PING_INTERVAL,
  RECONNECT_DELAYS,
  WS_URL,
} from '@/stores/workspace/constants';
import { markActivity, markConnected, markDisconnected, markError, markReconnecting } from '@/transports/state';
import { WebSocketClient } from '@/transports/websocket/websocketClient';
import type { WorkspaceStore } from '@/types/workspace';

type WorkspaceSet = (partial: Partial<WorkspaceStore>) => void;
type WorkspaceGet = () => WorkspaceStore;
type LogPayload = Omit<WorkspaceStore['logs'][number], 'id'>;

const websocketClient = new WebSocketClient<LogPayload>();

export function initializeWebSocket(set: WorkspaceSet, get: WorkspaceGet): void {
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
        console.warn('[WorkspaceStore] Invalid log entry:', logData);
        return null;
      }
      return logData as LogPayload;
    },
    onOpen: () => {
      console.log('[WorkspaceStore] WebSocket connected');
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
      console.error('[WorkspaceStore] WebSocket error:', error ?? message);
      set({
        connection: markError(get().connection, message),
      });
    },
    onClosedByRequest: () => {
      console.log('[WorkspaceStore] WebSocket closed');
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
        `[WorkspaceStore] Reconnecting in ${delay}ms (attempt ${attempt}/${MAX_RECONNECT_ATTEMPTS})`,
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
