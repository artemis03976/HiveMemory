/**
 * 用户导向身份选择类型定义
 *
 * 与计划文档 v0.6.2-identity-projection-cleanup.md 中的 IdentitySelection
 * 传输形状对应；后端将其一次性冻结为唯一的 IdentityScope。
 *
 * @module types/identity
 */

/**
 * 一次用户导向的身份选择。
 *
 * - `user_id + workspace_id` 是所有请求的基础选择；
 * - `agent_id` 仅 Chat / Agent run 等由具体 Agent 执行的操作提供，
 *   保留值 `system` 由后端注入，前端不得选择；
 * - `session_id` 仅 Chat / 被动会话需要时提供。
 */
export interface IdentitySelection {
  user_id: string;
  workspace_id: string;
  agent_id?: string;
  session_id?: string | null;
}
