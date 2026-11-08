import { apiUrl } from '@/transports/backendUrl';
import { EventSourceClient } from '@/transports/sse/eventSourceClient';
import { markConnected, markConnecting, markDisabled, markError } from '@/transports/state';
import type { WorkspaceStore, RuntimeEvent } from '@/types/workspace';

type WorkspaceSet = (partial: Partial<WorkspaceStore>) => void;
type WorkspaceGet = () => WorkspaceStore;

const runtimeEventClient = new EventSourceClient<RuntimeEvent>();

export function initializeRuntimeEventStream(set: WorkspaceSet, get: WorkspaceGet): void {
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

  void fetch(apiUrl('/api/v1/runtime-events/status'))
    .then((response) => (response.ok ? response.json() : { enabled: false }))
    .then((status: { enabled?: boolean; buffer_size?: number }) => {
      if (!status.enabled) {
        set({
          runtimeEventConnection: markDisabled(get().runtimeEventConnection, 'RuntimeEvent stream disabled'),
        });
        return;
      }
      openRuntimeEventSource(state.runtimeEventConnection.lastEventId, status.buffer_size ?? null, set, get);
    })
    .catch((error) => {
      const message = error instanceof Error ? error.message : 'RuntimeEvent status check failed';
      set({
        runtimeEventConnection: markError(get().runtimeEventConnection, message),
      });
    });
}

/**
 * 没有回放点的新连接请求回放整个后端缓冲（`replay_last` 取缓冲大小），刷新页面后仍能看到
 * 缓冲中的历史；带回放点的续传由后端回放其后的全部事件。
 */
function openRuntimeEventSource(
  lastEventId: string | null,
  bufferSize: number | null,
  set: WorkspaceSet,
  get: WorkspaceGet,
): void {
  const streamPath = bufferSize
    ? `/api/v1/runtime-events/stream?replay_last=${bufferSize}`
    : '/api/v1/runtime-events/stream';
  try {
    const eventSource = runtimeEventClient.connect({
      url: apiUrl(streamPath),
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
