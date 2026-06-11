export type TransportStatus =
  | 'idle'
  | 'connecting'
  | 'connected'
  | 'reconnecting'
  | 'disconnecting'
  | 'disconnected'
  | 'disabled'
  | 'error';

export interface TransportState {
  status: TransportStatus;
  error: string | null;
  connectedAt: number | null;
  disconnectedAt: number | null;
  reconnectAttempts: number;
  lastActivityAt: number | null;
}

export function createInitialTransportState(
  status: TransportStatus = 'disconnected',
): TransportState {
  return {
    status,
    error: null,
    connectedAt: null,
    disconnectedAt: null,
    reconnectAttempts: 0,
    lastActivityAt: null,
  };
}

export function markConnecting<T extends Partial<TransportState>>(state: T): T & { status: 'connecting'; error: null } {
  return {
    ...state,
    status: 'connecting',
    error: null,
  };
}

export function markConnected<T extends Partial<TransportState>>(
  state: T,
  now = Date.now(),
): T & {
  status: 'connected';
  error: null;
  connectedAt: number;
  disconnectedAt: null;
  reconnectAttempts: 0;
  lastActivityAt: number;
} {
  return {
    ...state,
    status: 'connected',
    error: null,
    connectedAt: now,
    disconnectedAt: null,
    reconnectAttempts: 0,
    lastActivityAt: now,
  };
}

export function markReconnecting<T extends Partial<TransportState>>(
  state: T,
  reconnectAttempts: number,
  now = Date.now(),
): T & {
  status: 'reconnecting';
  error: null;
  reconnectAttempts: number;
  lastActivityAt: number;
} {
  return {
    ...state,
    status: 'reconnecting',
    error: null,
    reconnectAttempts,
    lastActivityAt: now,
  };
}

export function markDisconnecting<T extends Partial<TransportState>>(
  state: T,
): T & { status: 'disconnecting' } {
  return {
    ...state,
    status: 'disconnecting',
  };
}

export function markDisconnected<T extends Partial<TransportState>>(
  state: T,
  now = Date.now(),
): T & {
  status: 'disconnected';
  error: null;
  connectedAt: null;
  disconnectedAt: number;
  reconnectAttempts: 0;
} {
  return {
    ...state,
    status: 'disconnected',
    error: null,
    connectedAt: null,
    disconnectedAt: now,
    reconnectAttempts: 0,
  };
}

export function markDisabled<T extends Partial<TransportState>>(
  state: T,
  reason: string,
  now = Date.now(),
): T & {
  status: 'disabled';
  error: string;
  connectedAt: null;
  disconnectedAt: number;
} {
  return {
    ...state,
    status: 'disabled',
    error: reason,
    connectedAt: null,
    disconnectedAt: now,
  };
}

export function markError<T extends Partial<TransportState>>(
  state: T,
  error: string,
  now = Date.now(),
): T & {
  status: 'error';
  error: string;
  disconnectedAt: number;
} {
  return {
    ...state,
    status: 'error',
    error,
    disconnectedAt: now,
  };
}

export function markActivity<T extends Partial<TransportState>>(
  state: T,
  now = Date.now(),
): T & { lastActivityAt: number } {
  return {
    ...state,
    lastActivityAt: now,
  };
}

export function resetReconnect<T extends Partial<TransportState>>(state: T): T & { reconnectAttempts: 0 } {
  return {
    ...state,
    reconnectAttempts: 0,
  };
}
