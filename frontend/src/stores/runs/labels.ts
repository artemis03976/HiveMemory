/** 运行与时间线共用的中文标签；未列出的取值照原样显示。 */
export const GENERATION_SOURCE_LABEL: Record<string, string> = {
  WRITE: '写入',
  UPDATE: '更新',
  SETTLE: '结算',
  MERGE: '合并',
};

export const SETTLE_REASON_LABEL: Record<string, string> = {
  idle_timeout: '空闲超时',
  volume_threshold: '按量',
  manual_settle: '手动结算',
  shutdown: '系统关闭',
  lru_eviction: 'LRU 回收',
};

/**
 * `topic.settled` 的名称：空闲或按量时只沉淀、话题留在池中；退役时话题离开话题池，有新内容时先沉淀。
 * 没有 `retired` 的事件来自区分二者之前，那时每次结算都结束话题，仍叫“话题结算”。
 */
export function topicSettledLabel(data: Record<string, unknown> | undefined): string {
  const retired = data?.retired;
  if (typeof retired !== 'boolean') return '话题结算';
  if (!retired) return '话题沉淀';
  return data?.generation_submitted === false ? '话题退役（没有新内容）' : '话题沉淀并退役';
}

/** 长标识只显示前 8 位，完整值在详情与原始事件中。 */
export function shortId(id: string | null | undefined): string {
  return id ? id.slice(0, 8) : '—';
}
