/**
 * 统一身份标识常量
 *
 * 必须与后端 src/hivememory/core/constants.py 保持一致
 *
 * 身份选择约定（v0.6.2 收敛）：
 * - 前端维护并传递 `user_id + workspace_id` 基础选择；
 *   只有 Chat、Agent run 等由具体 Agent 执行的操作才附加 `agent_id`。
 * - 保留值 `system`（非 Agent action 的来源标记）由后端注入，
 *   前端不得把它作为可选择的 Agent。
 *
 * @module constants/identity
 */

/**
 * 默认用户 ID - 用于未登录/匿名场景
 */
export const DEFAULT_USER_ID = 'default';

/**
 * 公共默认 Workspace ID - 当前阶段唯一开放的产品入口
 *
 * 必须与后端 MAIN_WORKSPACE_ID 保持一致
 */
export const DEFAULT_WORKSPACE_ID = 'main_workspace';

/**
 * 默认 Agent ID - 全能人偶，拥有完整权限
 */
export const DEFAULT_AGENT_ID = 'omni_doll';

/**
 * 测试用户 ID - 仅用于单元测试和集成测试
 */
export const TEST_USER_ID = 'test_user';

/**
 * 测试 Agent ID - 仅用于单元测试
 */
export const TEST_AGENT_ID = 'test_agent';
