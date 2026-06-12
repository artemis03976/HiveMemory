const BACKEND_ORIGIN = import.meta.env.VITE_BACKEND_ORIGIN || window.location.origin;

export const API_ORIGIN = new URL(BACKEND_ORIGIN, window.location.origin);
const WS_PROTOCOL = API_ORIGIN.protocol === 'https:' ? 'wss' : 'ws';

export const WS_URL = `${WS_PROTOCOL}://${API_ORIGIN.host}/api/v1/ws/logs`;
export const RUNTIME_EVENTS_URL = `${API_ORIGIN.origin}/api/v1/runtime-events/stream`;
export const RUNTIME_EVENTS_STATUS_URL = `${API_ORIGIN.origin}/api/v1/runtime-events/status`;

export const BROADCAST_CHANNEL_NAME = 'hivememory_kernel_logs';
export const PRIMARY_WINDOW_KEY = 'hivememory_primary_window';
export const HEARTBEAT_INTERVAL = 5000;
export const MAX_RECONNECT_ATTEMPTS = 10;
export const RECONNECT_DELAYS = [1000, 2000, 4000, 8000, 16000, 30000];
export const PING_INTERVAL = 30000;
export const CONNECT_TIMEOUT = 10000;
