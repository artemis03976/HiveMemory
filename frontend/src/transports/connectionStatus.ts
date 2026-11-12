/**
 * 与后端的连接状况，供页面顶部的状态条使用；各页面不再各自重复连接类的原因。
 *
 * - 连不上后端：请求在网络层失败时置位，之后任一请求得到后端的响应即清除；
 * - 管理凭据失效：后端以凭据缺失或不一致拒绝时置位，刷新页面才能取得新的凭据，因此不随后续响应清除。
 */
export interface ConnectionState {
  unreachable: boolean;
  credentialInvalid: boolean;
}

export type ConnectionEvent = 'network-failed' | 'responded' | 'credential-invalid';

export function nextConnectionState(state: ConnectionState, event: ConnectionEvent): ConnectionState {
  switch (event) {
    case 'network-failed':
      return state.unreachable ? state : { ...state, unreachable: true };
    case 'responded':
      return state.unreachable ? { ...state, unreachable: false } : state;
    case 'credential-invalid':
      return state.credentialInvalid ? state : { ...state, credentialInvalid: true };
  }
}

let current: ConnectionState = { unreachable: false, credentialInvalid: false };
const listeners = new Set<() => void>();

export function reportConnection(event: ConnectionEvent): void {
  const next = nextConnectionState(current, event);
  if (next === current) return;
  current = next;
  listeners.forEach((listener) => listener());
}

export function getConnectionState(): ConnectionState {
  return current;
}

/**
 * 页面局部读取失败时显示的文字。连不上后端或管理凭据失效时，顶部的状态条已经写明原因与办法，
 * 局部只说数据暂不可用；其他失败照常显示各自的原因。在捕获错误时调用：连接状况在请求抛出之前已经更新。
 */
export function localErrorText(reason: string): string {
  return current.unreachable || current.credentialInvalid ? '数据暂不可用' : reason;
}

export function subscribeConnection(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** 访问后端的 fetch：照常返回响应或抛出错误，同时把网络层失败与成功响应报告给连接状况。 */
export async function backendFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  try {
    const response = await fetch(input, init);
    reportConnection('responded');
    return response;
  } catch (error) {
    // 取消不是连接问题；fetch 在网络层失败时抛出 TypeError。
    if (error instanceof TypeError) reportConnection('network-failed');
    throw error;
  }
}
