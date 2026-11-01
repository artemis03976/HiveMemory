/**
 * HiveMemory 的 HTTP 客户端：MCP 端点上的工具调用，以及向 Import Bus 提交一轮。
 *
 * 每个请求以 `X-Principal-ID` 声明插件的 principal、以 `Authorization: Bearer`
 * 携带接入凭据，由服务端的统一认证网关核对。MCP 端点是无状态、JSON 响应的
 * streamable HTTP：每个请求独立处理，不需要先初始化会话。
 *
 * 提交结果的分类决定水位：确认（服务端在这一轮应用完成后才确认）后前进；
 * 可重试（网络错误、超时、503 等服务暂不可用）时保留水位、停止处理这个会话，
 * 等下次唤醒；413 与 422 是这一轮内容本身的问题，重发不会改变结果，写入拒绝
 * 记录后前进；其余 4xx（凭据未配置或不一致、身份声明不合法、未获准入、地址或
 * 版本不对）是插件配置的问题，与具体轮次无关，前进只会让后续轮次全部丢失，
 * 因此同样保留水位、停止处理，修正配置后补发；500 等服务器内部错误（含这一轮
 * 在应用时失败）由同步循环有限次重试。
 */

import { PRINCIPAL_ID } from './config.js'

const MCP_PATH = '/mcp'
const IMPORT_PATH = '/api/v1/import/turns'

// MCP 初始化时声明的协议版本；服务端不支持时以它支持的版本应答。
const MCP_PROTOCOL_VERSION = '2025-06-18'

// 服务暂不可用：关闭中、Gateway 暂不可用、提交队列拥塞，或前置代理的同类状态。
const RETRYABLE_STATUSES = new Set([429, 502, 503, 504])

// 只因这一轮的内容被拒绝的状态码：请求体过大、请求校验失败。
const TURN_REJECTED_STATUSES = new Set([413, 422])

export class HiveMemoryClient {
  constructor(config, fetchImpl = fetch) {
    this.config = config
    this.fetch = fetchImpl
    this.nextId = 1
  }

  /** 初始化结果中的服务说明，供系统提示词使用；没有说明时为空串。 */
  async instructions(signal) {
    const result = await this.mcp('initialize', {
      protocolVersion: MCP_PROTOCOL_VERSION,
      capabilities: {},
      clientInfo: { name: 'hivememory-dsh-connector', version: '1' },
    }, signal)
    return typeof result.instructions === 'string' ? result.instructions : ''
  }

  /** 服务端声明的工具：名字、说明与参数 schema。 */
  async listTools(signal) {
    const result = await this.mcp('tools/list', {}, signal)
    return Array.isArray(result.tools) ? result.tools : []
  }

  /** 调用一个工具，返回 MCP 工具结果（`content` 与 `isError`）。 */
  async callTool(name, args, signal) {
    return this.mcp('tools/call', { name, arguments: args }, signal)
  }

  /**
   * 发送一个 JSON-RPC 请求并返回 `result`。
   *
   * 网络错误、超时、非 2xx 状态与 JSON-RPC 错误都以 `Error` 抛出，说明写明
   * HiveMemory 一侧的原因；调用方的取消原样抛出。
   */
  async mcp(method, params, signal) {
    const body = { jsonrpc: '2.0', id: this.nextId++, method, params }
    const headers = {
      ...this.identityHeaders(),
      Accept: 'application/json, text/event-stream',
      'X-Agent-ID': this.config.agentId,
    }
    let response
    try {
      response = await this.post(MCP_PATH, body, headers, signal)
    } catch (error) {
      if (signal?.aborted) throw error
      throw new Error(`HiveMemory 暂时无法访问（${this.config.url}）：${error.message}`)
    }
    const { status, payload } = response
    if (status < 200 || status >= 300) {
      throw new Error(`HiveMemory MCP 请求失败，HTTP ${status}`)
    }
    if (payload.error !== undefined) {
      throw new Error(`HiveMemory MCP 请求失败：${payload.error.message ?? JSON.stringify(payload.error)}`)
    }
    return payload.result ?? {}
  }

  /**
   * 提交一轮并分类结果：`{ kind, status, importStatus, error, reason }`。
   *
   * `kind` 为 `confirmed`、`retryable`、`misconfigured`、`server_error` 或
   * `rejected`；网络错误与超时时 `status` 为空。调用方的取消原样抛出。
   */
  async submit(body, signal) {
    let response
    try {
      response = await this.post(IMPORT_PATH, body, this.identityHeaders(), signal)
    } catch (error) {
      if (signal?.aborted) throw error
      // 连接被拒绝、超时与 DNS 失败都表示服务暂时不可达。
      return { kind: 'retryable', status: null }
    }
    const { status, payload } = response
    if (status >= 200 && status < 300) {
      return { kind: 'confirmed', status, importStatus: payload.status ?? null }
    }
    const details = { status, error: payload.error ?? null, reason: payload.reason ?? null }
    if (RETRYABLE_STATUSES.has(status)) return { kind: 'retryable', ...details }
    if (status >= 500) return { kind: 'server_error', ...details }
    if (TURN_REJECTED_STATUSES.has(status)) return { kind: 'rejected', ...details }
    return { kind: 'misconfigured', ...details }
  }

  identityHeaders() {
    const headers = { 'Content-Type': 'application/json', 'X-Principal-ID': PRINCIPAL_ID }
    if (this.config.token) headers.Authorization = `Bearer ${this.config.token}`
    if (this.config.userId) headers['X-User-ID'] = this.config.userId
    return headers
  }

  /** 以 JSON 发送 POST；响应体不是 JSON 对象时按空对象处理。 */
  async post(path, body, headers, signal) {
    const timeout = AbortSignal.timeout(this.config.timeoutMs)
    const response = await this.fetch(this.config.url + path, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: signal === undefined ? timeout : AbortSignal.any([signal, timeout]),
    })
    let payload = {}
    try {
      const parsed = JSON.parse(await response.text())
      if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) payload = parsed
    } catch {
      // 代理返回的错误页等非 JSON 响应只按状态码分类。
    }
    return { status: response.status, payload }
  }
}
