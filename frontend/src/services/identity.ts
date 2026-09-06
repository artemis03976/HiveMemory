/**
 * 用户导向身份选择上下文
 *
 * 前端身份收敛（v0.6.2）的唯一事实源：
 * - 维护 `user_id + workspace_id` 基础选择，所有身份相关请求统一携带；
 * - Chat / Agent run 等操作在选择上附加具体 `agent_id`（见调用方）；
 * - 响应中的 `user_id` 是 owner 展示投影，不得反推为下一次 actor 选择。
 *
 * W0 阶段 Workspace 创建/切换未产品化，选择固定为默认用户与公共默认
 * Workspace；后续接入登录或 Workspace 切换时只需替换本模块的状态来源，
 * 不改动各 API client。
 *
 * @module services/identity
 */

import { DEFAULT_USER_ID, DEFAULT_WORKSPACE_ID } from '@/constants/identity';
import type { IdentitySelection } from '@/types';

let currentSelection: IdentitySelection = {
  user_id: DEFAULT_USER_ID,
  workspace_id: DEFAULT_WORKSPACE_ID,
};

/**
 * 读取当前用户导向身份选择。
 */
export function getIdentitySelection(): IdentitySelection {
  return { ...currentSelection };
}

/**
 * 更新用户导向身份选择。
 *
 * 当前阶段没有 Workspace 切换 UI，调用方应保持基础选择稳定；
 * `agent_id` 属于具体操作的参数，不应写入全局基础选择。
 */
export function setIdentitySelection(selection: Partial<IdentitySelection>): void {
  currentSelection = {
    ...currentSelection,
    ...selection,
  };
}

/**
 * 构造统一请求上下文请求头（x-user-id / x-workspace-id）。
 *
 * 后端 server 边界以该请求头为用户导向基础选择，body/query 携带同一
 * 字段且不一致时会被显式拒绝（HTTP 409）。
 */
export function identityHeaders(): Record<string, string> {
  const { user_id, workspace_id } = getIdentitySelection();
  return {
    'x-user-id': user_id,
    'x-workspace-id': workspace_id,
  };
}
