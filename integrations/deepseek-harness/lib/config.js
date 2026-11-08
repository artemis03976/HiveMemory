/** 插件配置：服务地址、身份声明、接入凭据与状态目录，全部来自环境变量。 */

import { homedir } from 'node:os'
import { join } from 'node:path'

// 插件在 HiveMemory 接入登记中的 principal；回流与记忆工具共用同一份接入登记与凭据。
export const PRINCIPAL_ID = 'harness:dsh'

// 导入轮次与写入时轮次信息中的来源标识。
export const SOURCE = 'dsh'

const DEFAULT_URL = 'http://localhost:8000'
const DEFAULT_AGENT_ID = 'dsh'
const DEFAULT_TIMEOUT_MS = 60_000

/**
 * 读取服务地址、身份声明与接入凭据。
 *
 * 环境变量为 `HIVEMEMORY_URL`、`HIVEMEMORY_USER_ID`、`HIVEMEMORY_AGENT_ID` 与
 * `HIVEMEMORY_TOKEN`；前三个与 Claude Code 连接器同名，凭据变量不同名，两个
 * harness 共用一套用户环境变量时各读各的凭据。凭据去掉首尾空白，复制时带入的
 * 换行不会让核对失败。`userId` 为空时不发送用户请求头，由服务端取默认用户；
 * `token` 为空时不发送，服务端以凭据无效拒绝。状态目录保存水位、拒绝记录与
 * 日志，优先使用 `HIVEMEMORY_CONNECTOR_STATE_DIR`，否则使用用户目录下的固定位置。
 */
export function configFromEnv(env = process.env) {
  return {
    url: (env.HIVEMEMORY_URL || DEFAULT_URL).replace(/\/+$/, ''),
    agentId: env.HIVEMEMORY_AGENT_ID || DEFAULT_AGENT_ID,
    userId: env.HIVEMEMORY_USER_ID || null,
    token: (env.HIVEMEMORY_TOKEN || '').trim() || null,
    stateDir: env.HIVEMEMORY_CONNECTOR_STATE_DIR || join(homedir(), '.hivememory', 'dsh-connector'),
    timeoutMs: DEFAULT_TIMEOUT_MS,
  }
}
