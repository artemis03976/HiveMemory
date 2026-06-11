import { getReconnectDelay, type RetryPolicy } from '@/transports/retryPolicy';

export interface WebSocketClientOptions<TMessage> {
  url: string;
  connectTimeoutMs: number;
  pingIntervalMs?: number;
  pingMessage?: string;
  pongMessage?: string;
  retryPolicy: RetryPolicy;
  getReconnectAttempts: () => number;
  shouldReconnect: () => boolean;
  parseMessage: (data: string) => TMessage | null;
  onOpen: () => void;
  onMessage: (message: TMessage) => void;
  onPong?: () => void;
  onError: (message: string, error?: Event) => void;
  onClosedByRequest: () => void;
  onReconnecting: (attempt: number, delay: number) => void;
  onReconnect: () => void;
  onReconnectExhausted: () => void;
}

type ManagedWebSocket = WebSocket & { _pingTimer?: ReturnType<typeof setInterval> };

export class WebSocketClient<TMessage> {
  private ws: ManagedWebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private manualDisconnecting = false;

  connect(options: WebSocketClientOptions<TMessage>): WebSocket | null {
    this.manualDisconnecting = false;

    let ws: ManagedWebSocket;
    try {
      ws = new WebSocket(options.url) as ManagedWebSocket;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'WebSocket initialization failed';
      options.onError(message);
      return null;
    }

    const connectTimeout = setTimeout(() => {
      if (ws.readyState === WebSocket.CONNECTING) {
        options.onError('WebSocket connection timeout');
        ws.close();
      }
    }, options.connectTimeoutMs);

    ws.onopen = () => {
      clearTimeout(connectTimeout);
      options.onOpen();

      const pingIntervalMs = options.pingIntervalMs;
      const pingMessage = options.pingMessage;
      if (pingIntervalMs && pingMessage) {
        ws._pingTimer = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(pingMessage);
          }
        }, pingIntervalMs);
      }
    };

    ws.onmessage = (event) => {
      try {
        if (typeof event.data !== 'string') return;
        if (options.pongMessage && event.data === options.pongMessage) {
          options.onPong?.();
          return;
        }

        const parsed = options.parseMessage(event.data);
        if (parsed !== null) {
          options.onMessage(parsed);
        }
      } catch (error) {
        console.error('[WebSocketClient] Failed to parse message:', error);
      }
    };

    ws.onerror = (error) => {
      clearTimeout(connectTimeout);
      options.onError('WebSocket connection error. Check backend status or browser security policy.', error);
    };

    ws.onclose = () => {
      clearTimeout(connectTimeout);
      this.clearPing(ws);

      if (this.manualDisconnecting) {
        this.ws = null;
        this.manualDisconnecting = false;
        options.onClosedByRequest();
        return;
      }

      if (!options.shouldReconnect()) {
        this.ws = null;
        options.onClosedByRequest();
        return;
      }

      const attempt = options.getReconnectAttempts() + 1;
      const delay = getReconnectDelay(attempt - 1, options.retryPolicy);
      if (delay === null) {
        this.ws = null;
        options.onReconnectExhausted();
        return;
      }

      options.onReconnecting(attempt, delay);
      this.reconnectTimer = setTimeout(() => {
        options.onReconnect();
      }, delay);
    };

    this.ws = ws;
    return ws;
  }

  disconnect(): void {
    this.manualDisconnecting = true;
    this.clearReconnectTimer();
    if (this.ws) {
      this.ws.close();
      return;
    }
    this.manualDisconnecting = false;
  }

  clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  get current(): WebSocket | null {
    return this.ws;
  }

  private clearPing(ws: ManagedWebSocket): void {
    if (ws._pingTimer) {
      clearInterval(ws._pingTimer);
      ws._pingTimer = undefined;
    }
  }

}
