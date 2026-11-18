/** Agent Profile 的模型参数。提示词语言由全局语言设置决定，不属于单个 Agent。 */
export interface AgentProfileConfig {
  model: string;
  temperature: number;
  /** 为空时沿用模型定义的默认值。 */
  top_p: number | null;
}

export type AgentStatus = 'Active' | 'Inactive';

export interface AgentData {
  agentId: string;      // agent_id — Profile 的文件名，创建后不可改
  name: string;         // title — 用户给它起的名字
  summary: string;      // summary — 一句话描述它做什么
  systemPrompt: string; // persona — 人格/系统指令
  model: string;        // config.model
  status: AgentStatus;
  config: AgentProfileConfig;
  /** 最近保存的时间（ISO 字符串，即 Profile 文件的修改时间）；尚未保存的草稿没有。 */
  updatedAt?: string;
}
