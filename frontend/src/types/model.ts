/**
 * 模型注册表类型定义
 *
 * 对应后端 src/hivememory/server/routers/models.py 的请求/响应模型。
 * 注册表是可用 LLM 模型的单一数据源，前端通过 /api/v1/models CRUD 管理。
 */

/** 注册表中的单条模型记录（GET 响应，api_key 已脱敏） */
export interface RegisteredModel {
  id: string;
  display_name: string;
  litellm_model: string;
  /** 提供商标识，用于按 provider 凭证表解析 api_key/api_base */
  provider: string;
  /** 脱敏后的 API 密钥，如 'sk-...abcd'；未设置为 null */
  api_key_masked: string | null;
  api_base: string | null;
  temperature: number;
  max_tokens: number;
  top_p: number;
  is_default: boolean;
}

/** 创建模型的请求体 */
export interface ModelCreatePayload {
  id: string;
  display_name: string;
  litellm_model: string;
  /** 提供商标识，留空则由后端从 litellm_model 前缀自动推导 */
  provider?: string;
  /** 明文 API 密钥，留空则由后端从 provider 凭证或环境变量读取 */
  api_key?: string | null;
  api_base?: string | null;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  is_default?: boolean;
}

/**
 * 更新模型的请求体 — 全部可选，只发送需要修改的字段。
 * api_key 传空字符串 "" 表示清除密钥（改为从 provider 凭证/环境变量读取）。
 */
export interface ModelUpdatePayload {
  display_name?: string;
  litellm_model?: string;
  provider?: string;
  api_key?: string | null;
  api_base?: string | null;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  is_default?: boolean;
}
