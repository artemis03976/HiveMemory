/** Agent 图纸的模型参数与提示词语言。 */
export interface AgentProfileConfig {
  model_name: string;
  temperature: number;
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
}
