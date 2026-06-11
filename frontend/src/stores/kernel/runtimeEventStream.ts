import { RUNTIME_EVENTS_STATUS_URL, RUNTIME_EVENTS_URL } from '@/stores/kernel/constants';
import type { KernelStore, RuntimeEvent } from '@/types/kernel';

type KernelSet = (partial: Partial<KernelStore>) => void;
type KernelGet = () => KernelStore;

export function initializeRuntimeEventStream(set: KernelSet, get: KernelGet): void {
  const state = get();
  if (
    state._eventSource &&
    state.runtimeEventConnection.status !== 'disabled' &&
    state.runtimeEventConnection.status !== 'error'
  ) {
    return;
  }

  const lastEventId = state.runtimeEventConnection.lastEventId;
  const url = new URL(RUNTIME_EVENTS_URL);
  if (lastEventId) {
    url.searchParams.set('last_event_id', lastEventId);
  }

  set({
    runtimeEventConnection: {
      ...state.runtimeEventConnection,
      status: 'connecting',
      error: null,
    },
  });

  void fetch(RUNTIME_EVENTS_STATUS_URL)
    .then((response) => (response.ok ? response.json() : { enabled: false }))
    .then((status: { enabled?: boolean }) => {
      if (!status.enabled) {
        set({
          runtimeEventConnection: {
            ...get().runtimeEventConnection,
            status: 'disabled',
            error: 'RuntimeEvent stream disabled',
          },
        });
        return false;
      }
      return true;
    })
    .then((enabled) => {
      if (!enabled) return;
      openRuntimeEventSource(url, set, get);
    })
    .catch((error) => {
      const message = error instanceof Error ? error.message : 'RuntimeEvent status check failed';
      set({
        runtimeEventConnection: {
          ...get().runtimeEventConnection,
          status: 'error',
          error: message,
        },
      });
    });
}

function openRuntimeEventSource(url: URL, set: KernelSet, get: KernelGet): void {
  try {
    const eventSource = new EventSource(url.toString());

    eventSource.onopen = () => {
      set({
        runtimeEventConnection: {
          ...get().runtimeEventConnection,
          status: 'connected',
          connectedAt: Date.now(),
          error: null,
        },
      });
    };

    eventSource.addEventListener('runtime_event', (message) => {
      try {
        const runtimeEvent = JSON.parse(message.data) as RuntimeEvent;
        get().addRuntimeEvent(runtimeEvent);

        const current = get();
        if (current._isPrimaryWindow && current._broadcastChannel) {
          current._broadcastChannel.postMessage({
            type: 'NEW_RUNTIME_EVENT',
            payload: runtimeEvent,
          });
        }
      } catch (error) {
        console.error('[KernelStore] Failed to parse RuntimeEvent:', error);
      }
    });

    eventSource.onerror = () => {
      const currentSource = get()._eventSource;
      const status = currentSource?.readyState === EventSource.CLOSED ? 'disabled' : 'error';
      set({
        runtimeEventConnection: {
          ...get().runtimeEventConnection,
          status,
          error: status === 'disabled' ? 'RuntimeEvent stream disabled' : 'RuntimeEvent stream connection error',
        },
      });
    };

    set({ _eventSource: eventSource });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'RuntimeEvent stream initialization failed';
    set({
      runtimeEventConnection: {
        ...get().runtimeEventConnection,
        status: 'error',
        error: message,
      },
    });
  }
}
