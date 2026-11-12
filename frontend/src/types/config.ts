/**
 * 主配置的 JSON 对象。字段结构以后端的配置模型为准，前端不重复定义：
 * 读写都按字段路径进行（`readPath` 与设置页的字段表），字段表中的每条路径由后端测试核对。
 */
export type HiveMemoryConfig = Record<string, unknown>;

/** 主配置接口的读取与保存结果：期望配置，以及立即生效与等待重启的字段路径。 */
export interface ConfigResponse {
  config: HiveMemoryConfig;
  immediate_fields: string[];
  pending_restart: string[];
}

export interface ValidationError {
  field: string;
  message: string;
  severity: 'error' | 'warning';
}
