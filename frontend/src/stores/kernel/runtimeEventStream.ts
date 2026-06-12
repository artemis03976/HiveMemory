import { RUNTIME_EVENTS_STATUS_URL, RUNTIME_EVENTS_URL } from '@/stores/kernel/constants';
import { EventSourceClient } from '@/transports/sse/eventSourceClient';
import { markConnected, markConnecting, markDisabled, markError } from '@/transports/state';
import type { KernelStore, RuntimeEvent } from '@/types/kernel';

type KernelSet = (partial: Partial<KernelStore>) => void;
type KernelGet = () => KernelStore;

const runtimeEventClient = new EventSourceClient<RuntimeEvent>();

export function initializeRuntimeEventStream(set: KernelSet, get: KernelGet): void {
  const state = get();
  if (
    state._eventSource &&
    state.runtimeEventConnection.status !== 'disabled' &&
    state.runtimeEventConnection.status !== 'error'
  ) {
    return;
  }

  set({
    runtimeEventConnection: markConnecting(state.runtimeEventConnection),
  });

  void fetch(RUNTIME_EVENTS_STATUS_URL)
    .then((response) => (response.ok ? response.json() : { enabled: false }))
    .then((status: { enabled?: boolean }) => {
      if (!status.enabled) {
        set({
          runtimeEventConnection: markDisabled(get().runtimeEventConnection, 'RuntimeEvent stream disabled'),
        });
        return false;
      }
      return true;
    })
    .then((enabled) => {
      if (!enabled) return;
      openRuntimeEventSource(state.runtimeEventConnection.lastEventId, set, get);
    })
    .catch((error) => {
      const message = error instanceof Error ? error.message : 'RuntimeEvent status check failed';
      set({
        runtimeEventConnection: markError(get().runtimeEventConnection, message),
      });
    });
}

function openRuntimeEventSource(lastEventId: string | null, set: KernelSet, get: KernelGet): void {
  try {
    const eventSource = runtimeEventClient.connect({
      url: RUNTIME_EVENTS_URL,
      lastEventId,
      eventName: 'runtime_event',
      parseEvent: (data) => JSON.parse(data) as RuntimeEvent,
      onOpen: () => {
        set({
          runtimeEventConnection: markConnected(get().runtimeEventConnection),
        });
      },
      onEvent: (runtimeEvent) => {
        get().addRuntimeEvent(runtimeEvent);

        const current = get();
        if (current._isPrimaryWindow && current._broadcastChannel) {
          current._broadcastChannel.postMessage({
            type: 'NEW_RUNTIME_EVENT',
            payload: runtimeEvent,
          });
        }
      },
      onStatus: (status, error) => {
        if (status === 'disabled') {
          set({
            runtimeEventConnection: markDisabled(get().runtimeEventConnection, error ?? 'RuntimeEvent stream disabled'),
          });
          return;
        }
        if (status === 'error') {
          set({
            runtimeEventConnection: markError(get().runtimeEventConnection, error ?? 'RuntimeEvent stream connection error'),
          });
        }
      },
    });

    set({ _eventSource: eventSource });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'RuntimeEvent stream initialization failed';
    set({
      runtimeEventConnection: markError(get().runtimeEventConnection, message),
    });
  }
}
