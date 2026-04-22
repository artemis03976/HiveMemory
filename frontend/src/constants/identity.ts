/**
 * 统一身份标识常量
 *
 * 必须与后端 src/hivememory/core/constants.py 保持一致
 *
 * @module constants/identity
 */

/**
 * 默认用户 ID - 用于未登录/匿名场景
 */
export const DEFAULT_USER_ID = 'default';

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
