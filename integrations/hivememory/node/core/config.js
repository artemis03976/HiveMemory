/** 连接器配置：服务地址、身份声明、接入凭据与状态目录，按 harness 取值，其余来自环境变量。 */

import { homedir } from 'node:os'
import { join } from 'node:path'

// Windows 原生运行时 HiveMemory 后端的地址。
const DEFAULT_URL = 'http://127.0.0.1:8769'
const DEFAULT_TIMEOUT_MS = 60_000

/**
 * 以一个 harness 的固定取值与环境变量构造配置。
 *
 * `harness` 给出 principal、来源标识、默认 Agent、凭据变量名、默认状态目录名与
 * MCP 初始化时声明的客户端名。环境变量为 `HIVEMEMORY_URL`、`HIVEMEMORY_USER_ID`、
 * `HIVEMEMORY_AGENT_ID`、`HIVEMEMORY_CONNECTOR_STATE_DIR` 与该 harness 的凭据变量。
 * 凭据变量名带上 harness：多个 harness 共用一套用户环境变量时各读各的凭据，不会
 * 拿别的 harness 的凭据冒充本连接器的 principal；服务端核对这份凭据的变量也是这个
 * 名字。凭据去掉首尾空白，复制时带入的换行不会让核对失败。`userId` 为空时不发送
 * 用户请求头，由服务端取默认用户；`token` 为空时不发送，服务端以凭据无效拒绝。
 */
export function configFromEnv(harness, env = process.env) {
  return {
    principalId: harness.principalId,
    source: harness.source,
    clientName: harness.clientName,
    url: (env.HIVEMEMORY_URL || DEFAULT_URL).replace(/\/+$/, ''),
    agentId: env.HIVEMEMORY_AGENT_ID || harness.defaultAgentId,
    userId: env.HIVEMEMORY_USER_ID || null,
    token: (env[harness.tokenEnv] || '').trim() || null,
    stateDir: env.HIVEMEMORY_CONNECTOR_STATE_DIR || join(homedir(), '.hivememory', harness.stateDirName),
    timeoutMs: DEFAULT_TIMEOUT_MS,
  }
}
