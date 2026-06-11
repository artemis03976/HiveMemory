import { HEARTBEAT_INTERVAL, PRIMARY_WINDOW_KEY } from '@/stores/kernel/constants';
import type { BroadcastMessage, KernelStore } from '@/types/kernel';

type KernelSet = (partial: Partial<KernelStore>) => void;
type KernelGet = () => KernelStore;

export function checkAndClaimPrimaryWindow(): boolean {
  const now = Date.now();
  const stored = localStorage.getItem(PRIMARY_WINDOW_KEY);

  if (!stored) {
    localStorage.setItem(PRIMARY_WINDOW_KEY, now.toString());
    return true;
  }

  const lastHeartbeat = parseInt(stored, 10);
  if (now - lastHeartbeat > HEARTBEAT_INTERVAL * 2) {
    localStorage.setItem(PRIMARY_WINDOW_KEY, now.toString());
    return true;
  }

  return false;
}

export function handleBroadcastMessage(message: BroadcastMessage, set: KernelSet, get: KernelGet): void {
  switch (message.type) {
    case 'NEW_LOG':
      get().addLog(message.payload);
      break;

    case 'NEW_RUNTIME_EVENT':
      get().addRuntimeEvent(message.payload);
      break;

    case 'BATCH_LOGS':
      get().addLogs(message.payload);
      break;

    case 'CLEAR_LOGS':
      set({ logs: [], stats: { ...get().stats, totalLogs: 0, filteredCount: 0 } });
      break;

    case 'CLEAR_RUNTIME_EVENTS':
      set({ runtimeEvents: [] });
      break;

    case 'CONNECTION_STATE':
      set({ connection: message.payload });
      break;

    case 'RUNTIME_EVENT_CONNECTION_STATE':
      set({ runtimeEventConnection: message.payload });
      break;

    case 'FILTER_UPDATE':
      set({ filters: message.payload });
      break;

    case 'UI_UPDATE':
      set({ ui: { ...get().ui, ...message.payload } });
      break;

    case 'REQUEST_SYNC': {
      const state = get();
      if (state._isPrimaryWindow && state._broadcastChannel) {
        state._broadcastChannel.postMessage({
          type: 'FULL_SYNC',
          payload: {
            logs: state.logs,
            runtimeEvents: state.runtimeEvents,
            filters: state.filters,
            ui: state.ui,
          },
        });
      }
      break;
    }

    case 'FULL_SYNC':
      if (!get()._isPrimaryWindow) {
        set({
          logs: message.payload.logs,
          runtimeEvents: message.payload.runtimeEvents,
          filters: message.payload.filters,
          ui: message.payload.ui,
        });
      }
      break;

    case 'PRIMARY_HEARTBEAT':
      localStorage.setItem(PRIMARY_WINDOW_KEY, message.timestamp.toString());
      break;
  }
}

export function startPrimaryHeartbeat(get: KernelGet): ReturnType<typeof setInterval> {
  return setInterval(() => {
    const state = get();
    if (!state._isPrimaryWindow) return;

    const now = Date.now();
    localStorage.setItem(PRIMARY_WINDOW_KEY, now.toString());

    if (state._broadcastChannel) {
      state._broadcastChannel.postMessage({
        type: 'PRIMARY_HEARTBEAT',
        timestamp: now,
      });
    }
  }, HEARTBEAT_INTERVAL);
}
