/** Agent 图纸的模型参数与提示词语言。 */
export interface AgentProfileConfig {
  model_name: string;
  temperature: number;
  /** 为空时沿用模型定义的默认值。 */
  top_p: number | null;
  language: string;
}

export type AgentStatus = 'Active' | 'Inactive';

export interface AgentData {
  id: string;          // 后端 UUID
  alias: string;       // index.alias — snake_case 标识符
  name: string;        // index.title — 显示名
  summary: string;     // index.summary — 一句话简介
  tags: string[];      // index.tags — 语义标签
  systemPrompt: string; // payload.content — 人格/系统指令
  model: string;       // config.model_name
  status: AgentStatus;
  config: AgentProfileConfig;
  /** 创建与最近保存的时间（ISO 字符串）；尚未保存的草稿没有。 */
  createdAt?: string;
  updatedAt?: string;
}
