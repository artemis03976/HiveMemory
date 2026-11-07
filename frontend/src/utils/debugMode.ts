/**
 * 前端的 debug 模式：只在构建或开发服务器显式设置 `VITE_DEBUG=true` 时开启。
 *
 * 开启后，记忆、Agent、设置与记忆任务的读取失败时才换用 mock 数据，便于脱离后端调界面；
 * 关闭时（默认）读取失败如实显示错误，不能把假数据当作真实结果展示。
 */
export function isDebugMode(): boolean {
  return import.meta.env.VITE_DEBUG === 'true';
}
