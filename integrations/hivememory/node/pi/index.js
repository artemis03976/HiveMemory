/**
 * HiveMemory 的 pi 扩展：把 pi 会话中已经结束的一轮导入 HiveMemory 的 Import
 * Bus，并让 pi 中的 Agent 经扩展注册的工具检索、读取、写入与更新 HiveMemory 的
 * 记忆。
 *
 * - 记忆工具：会话开始后在后台从 HiveMemory 的 MCP 端点取得服务说明与工具列表，
 *   注册四个记忆工具；每次开始运行时最多等一小段时间让它完成，第一轮的请求因此
 *   就带着这些工具。服务说明作为系统提示词的一节。HiveMemory 不可用时工具不
 *   出现，pi 照常工作，下一次开始运行时再试。
 * - 一轮：从开始运行（`before_agent_start`）到 `agent_settled`，运行中追加的消息
 *   与排队的后续消息都并入这一轮。开始运行时生成轮次标识、记下当时的叶子条目；
 *   结束时先记入待提交，再写结束标记，然后在后台提交。
 * - 回流：会话开始时补发此前未确认的轮次；会话关闭时最多等一小段时间让进行中的
 *   提交完成，`pi -p` 因此在这一轮得到确认之后才退出。任何失败只进日志。
 *
 * 不保存会话的运行（`--no-session`）没有会话文件，不导入。配置全部来自环境
 * 变量，状态与日志写入状态目录；扩展只用 Node 标准库，不依赖 pi 的包。
 */

import { randomUUID } from 'node:crypto'

import { HiveMemoryClient } from '../core/client.js'
import { configFromEnv } from '../core/config.js'
import { ConnectorLog } from '../core/log.js'
import { StateStore } from '../core/state.js'
import { SessionSync } from './sync.js'
import { memoryToolDefinitions } from './tools.js'
import { branchSince, inFlightTurnInfo, lastStopReason, MARKER_TYPE, outcomeOf, SOURCE } from './turns.js'

export const HARNESS = {
  principalId: 'harness:pi',
  source: SOURCE,
  defaultAgentId: 'pi',
  tokenEnv: 'HIVEMEMORY_PI_TOKEN',
  stateDirName: 'pi-connector',
  clientName: 'hivememory-pi-connector',
}

// 开始运行时等待记忆工具注册的上限：HiveMemory 不可达时连接很快失败，响应慢时
// 不让每一轮都多等太久。
const CONNECT_WAIT_MS = 3_000

// 会话关闭时等待进行中提交的上限，超时的留到下次启动补发。
const SHUTDOWN_WAIT_MS = 5_000

// 收集为这一轮用户消息的输入来源：用户输入与 RPC 客户端；其他扩展发来的不收集。
const USER_INPUT_SOURCES = new Set(['interactive', 'rpc'])

export default function hivememoryExtension(pi) {
  install(pi, process.env, fetch)
}

/** 在 pi 上注册扩展的事件处理与工具；环境变量与 `fetch` 由调用方给出。 */
export function install(pi, env, fetchImpl) {
  const config = configFromEnv(HARNESS, env)
  const log = new ConnectorLog(config.stateDir)
  const client = new HiveMemoryClient(config, fetchImpl)
  const abort = new AbortController()
  const sync = new SessionSync({
    store: new StateStore(config.stateDir),
    client,
    log,
    agentId: config.agentId,
    signal: abort.signal,
  })

  // 进行中的一轮：轮次标识、开始时的叶子条目与收到的原始输入。
  let current = null
  // 空闲时收到、尚未开始运行的用户输入。
  let idleInput = null

  let instructions = ''
  let registered = false
  let connecting = null

  function turnInfo(ctx) {
    if (current === null) throw new Error('HiveMemory 写入与更新只能在一轮进行中调用')
    const sessionManager = ctx.sessionManager
    return inFlightTurnInfo(sessionManager.getEntries(), sessionManager.getSessionId(), current)
  }

  function connect() {
    if (registered) return Promise.resolve()
    if (connecting !== null) return connecting
    connecting = (async () => {
      const [text, serverTools] = await Promise.all([
        client.instructions(abort.signal),
        client.listTools(abort.signal),
      ])
      abort.signal.throwIfAborted()
      const definitions = memoryToolDefinitions(serverTools, { client, turnInfo })
      if (definitions.length === 0) throw new Error('server does not offer the four memory tools')
      for (const definition of definitions) pi.registerTool(definition)
      instructions = text
      registered = true
      log.write(`connected url=${config.url} tools=${definitions.map(tool => tool.name).join(',')}`)
    })()
      .catch(error => {
        if (!abort.signal.aborted) log.write(`connect failed url=${config.url} error=${error.message}`)
      })
      .finally(() => {
        connecting = null
      })
    return connecting
  }

  pi.on('session_start', async () => {
    void connect()
    void sync.recover().catch(error => log.write(`recover failed error=${error.message}`))
  })

  pi.on('input', async event => {
    const text = USER_INPUT_SOURCES.has(event.source) ? event.text : null
    if (current !== null) {
      if (text !== null) current.inputs.push(text)
    } else {
      idleInput = text
    }
  })

  pi.on('before_agent_start', async (event, ctx) => {
    await waitAtMost(connect(), CONNECT_WAIT_MS)
    if (current === null) {
      const inputs = event.messageType === 'user' && idleInput !== null ? [idleInput] : []
      current = { turn: randomUUID(), from: ctx.sessionManager.getLeafId() ?? null, inputs }
      idleInput = null
    }
    if (instructions) event.systemPromptOptions.sections.hivememory = instructions
  })

  pi.on('agent_settled', async (event, ctx) => {
    const run = current
    current = null
    if (run === null) return
    const sessionManager = ctx.sessionManager
    const file = sessionManager.getSessionFile()
    if (!file) return
    const sessionId = sessionManager.getSessionId()
    try {
      const entries = branchSince(sessionManager.getEntries(), sessionManager.getLeafId(), run.from)
      const outcome = outcomeOf(event.aborted, lastStopReason(entries))
      sync.noteEnded(sessionId, file, run.turn)
      pi.appendEntry(MARKER_TYPE, { turn: run.turn, session: sessionId, from: run.from, outcome, inputs: run.inputs })
      void sync.request(sessionId)
    } catch (error) {
      log.write(`turn end failed session=${sessionId} turn=${run.turn} error=${error.message}`)
    }
  })

  pi.on('session_shutdown', async () => {
    await sync.drain(SHUTDOWN_WAIT_MS)
    abort.abort()
  })
}

async function waitAtMost(promise, timeoutMs) {
  let timer
  const timeout = new Promise(resolve => {
    timer = setTimeout(resolve, timeoutMs)
  })
  await Promise.race([promise, timeout])
  clearTimeout(timer)
}
