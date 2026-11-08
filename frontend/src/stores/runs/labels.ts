/** 运行与时间线共用的中文标签；未列出的取值照原样显示。 */
export const GENERATION_SOURCE_LABEL: Record<string, string> = {
  WRITE: '写入',
  UPDATE: '更新',
  SETTLE: '结算',
  MERGE: '合并',
};

export const SETTLE_REASON_LABEL: Record<string, string> = {
  idle_timeout: '空闲超时',
  manual_settle: '手动结算',
  shutdown: '系统关闭',
  lru_eviction: 'LRU 回收',
};

/** 长标识只显示前 8 位，完整值在详情与原始事件中。 */
export function shortId(id: string | null | undefined): string {
  return id ? id.slice(0, 8) : '—';
}
