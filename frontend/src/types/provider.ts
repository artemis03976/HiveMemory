/** 提供商凭证的 API 响应体 */
export interface RegisteredProvider {
  name: string;
  /** 脱敏后的 API 密钥，如 "sk-...abcd"；未设置则为 null */
  api_key_masked: string | null;
  api_base: string | null;
  /** true 表示来自环境变量（只读，不可通过 UI 删除） */
  is_from_env: boolean;
}

/** 创建或更新提供商的请求体 */
export interface ProviderUpsertPayload {
  /** API 密钥；传 null 表示保留现有值（更新时留空） */
  api_key?: string | null;
  api_base?: string | null;
}
