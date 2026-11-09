import { ApiError, readApiError } from '@/services/apiError';
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

  void fetchRuntimeEventStatus()
    .then((status) => {
      if (!status.enabled) {
        set({
          runtimeEventConnection: markDisabled(get().runtimeEventConnection, 'RuntimeEvent stream disabled'),
        });
        return;
      }
      openRuntimeEventSource(state.runtimeEventConnection.lastEventId, status.buffer_size ?? null, set, get);
    })
    .catch((error) => {
      set({
        runtimeEventConnection: markError(get().runtimeEventConnection, statusErrorMessage(error)),
      });
    });
}

/** 状态接口被后端拒绝时如实报错，不能当成事件流未启用。 */
async function fetchRuntimeEventStatus(): Promise<{ enabled?: boolean; buffer_size?: number }> {
  const response = await fetch(apiUrl('/api/v1/runtime-events/status'));
  if (!response.ok) throw await readApiError(response, '事件流状态读取失败');
  return response.json();
}

function statusErrorMessage(error: unknown): string {
  return error instanceof ApiError ? error.message : '事件流连接失败';
}

/**
 * 浏览器在 EventSource 收到非 200 响应后不再重连，也不交出状态码：事件流未启用（503）
 * 与管理凭据失效（401）看起来一样。以状态接口再问一次，区分这两种情况；状态接口说已启用
 * 时按连接失败处理，由用户重试，不自动重开。
 */
function explainClosedStream(set: WorkspaceSet, get: WorkspaceGet): void {
  void fetchRuntimeEventStatus()
    .then((status) => {
      set({
        runtimeEventConnection: status.enabled
          ? markError(get().runtimeEventConnection, '事件流连接失败')
          : markDisabled(get().runtimeEventConnection, 'RuntimeEvent stream disabled'),
      });
    })
    .catch((error) => {
      set({
        runtimeEventConnection: markError(get().runtimeEventConnection, statusErrorMessage(error)),
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
      onStatus: (status) => {
        if (status === 'disabled') {
          explainClosedStream(set, get);
          return;
        }
        if (status === 'error') {
          // 浏览器仍在自动重连；原因拿不到，界面只说连接失败。
          set({
            runtimeEventConnection: markError(get().runtimeEventConnection, '事件流连接失败'),
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
