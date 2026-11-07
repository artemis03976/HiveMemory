/**
 * 后端地址：HTTP、SSE 与 WebSocket 的 URL 都从这里取，三者始终连接同一个后端。
 *
 * 构建或开发服务器设置 `VITE_BACKEND_ORIGIN` 时直连这个 origin；不设置时（默认）连接页面
 * 自身的 origin，即同源部署的后端或 Vite 代理。
 */
const BACKEND_ORIGIN: string | undefined = import.meta.env.VITE_BACKEND_ORIGIN || undefined;

/** HTTP 与 SSE 请求的 URL；未设置后端 origin 时保持相对路径，由页面 origin 承接。 */
export function apiUrl(path: string): string {
  return BACKEND_ORIGIN ? new URL(path, BACKEND_ORIGIN).toString() : path;
}

/** WebSocket 的 URL；浏览器的 WebSocket 只接受绝对地址，未设置后端 origin 时取页面 origin。 */
export function websocketUrl(path: string): string {
  const url = new URL(path, BACKEND_ORIGIN ?? window.location.origin);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.toString();
}
