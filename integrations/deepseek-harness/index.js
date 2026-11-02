/**
 * HiveMemory 的 dsh 插件：把 dsh 会话中已经结束的轮次导入 HiveMemory 的 Import
 * Bus，并让 dsh 中的 Agent 经原生工具检索、读取、写入与更新 HiveMemory 的记忆。
 *
 * - 记忆工具：插件加载时与每次创建 Agent 时（直到成功一次），从 HiveMemory 的
 *   MCP 端点取得工具列表与服务说明，注册四个记忆工具，服务说明作为系统提示词
 *   的一节。HiveMemory 不可用时工具不出现，dsh 照常工作。
 * - 回流：一轮结束与 Agent 创建或恢复时同步该会话，插件加载时补发此前未确认
 *   的轮次；网络提交在后台进行，不阻塞 dsh，任何失败只进日志。子 Agent 的
 *   会话不导入，它交回的消息随父会话的一轮导入。
 *
 * 配置全部来自环境变量，状态与日志写入状态目录；插件只用 Node 标准库，不依赖
 * dsh 的包，经 `ctx` 取得 dsh 的服务。
 */

import { HiveMemoryClient } from './lib/client.js'
import { configFromEnv } from './lib/config.js'
import { ConnectorLog } from './lib/log.js'
import { StateStore } from './lib/state.js'
import { SessionSync } from './lib/sync.js'
import { currentTurnInfo, memoryToolDefinitions } from './lib/tools.js'
import { isSubagentSession } from './lib/turns.js'

export const name = 'hivememory-connector'

export const inject = ['tools', 'sessions', 'sessionQuery', 'systemPrompt']

export function apply(ctx) {
  const config = configFromEnv()
  const log = new ConnectorLog(config.stateDir)
  const client = new HiveMemoryClient(config)
  const abort = new AbortController()
  ctx.effect(() => () => abort.abort())

  const sync = new SessionSync({
    store: new StateStore(config.stateDir),
    client,
    log,
    agentId: config.agentId,
    signal: abort.signal,
    sessions: {
      isLive: sessionId => ctx.sessions.get(sessionId) !== undefined,
      flush: sessionId => ctx.sessions.flush(ctx.sessions.get(sessionId)),
      read: sessionId => ctx.sessionQuery.readSession(sessionId),
    },
  })

  let instructions = ''
  ctx.systemPrompt.section({
    name: 'hivememory',
    order: ctx.systemPrompt.getSectionOrder('MCP_SERVERS'),
    interpolate: false,
    text: () => instructions,
  })

  let connecting = null
  let connected = false
  function connect() {
    if (connected || connecting !== null) return
    connecting = (async () => {
      const [text, serverTools] = await Promise.all([
        client.instructions(abort.signal),
        client.listTools(abort.signal),
      ])
      abort.signal.throwIfAborted()
      const definitions = memoryToolDefinitions(serverTools, {
        client,
        turnInfoFor: session => currentTurnInfo(session, {
          readSession: sessionId => ctx.sessionQuery.readSession(sessionId),
        }),
      })
      registerTools(definitions)
      instructions = text
      connected = true
      log.write(`connected url=${config.url} tools=${definitions.map(tool => tool.name).join(',')}`)
    })()
      .catch((error) => {
        if (!abort.signal.aborted) log.write(`connect failed url=${config.url} error=${error.message}`)
      })
      .finally(() => {
        connecting = null
      })
  }

  /** 全部注册或全部不注册：模型不会只看到一部分记忆工具。 */
  function registerTools(definitions) {
    const disposers = []
    try {
      for (const definition of definitions) disposers.push(ctx.tools.register(definition))
    } catch (error) {
      for (const dispose of disposers) dispose()
      throw error
    }
  }

  /** 首次见到会话时登记；插件首次运行之前创建的会话不回填。 */
  function track(session) {
    sync.register(session.id, { createdAt: session.header.createdAt })
  }

  // 两个监听都在 dsh 的执行路径上同步运行：只做本地的快速工作，失败只进日志。
  // 子 Agent 的会话既不登记也不同步。
  ctx.on('agent/created', ({ agent }) => {
    try {
      connect()
      if (isSubagentSession(agent.session.header)) return
      track(agent.session)
      void sync.request(agent.session.id)
    } catch (error) {
      log.write(`agent/created failed session=${agent.session.id} error=${error.message}`)
    }
  })

  ctx.on('session/event', (session, event) => {
    if (event.type !== 'turn/end' || isSubagentSession(session.header)) return
    try {
      track(session)
      sync.noteEnded(session.id, event.data.turn)
      void sync.request(session.id)
    } catch (error) {
      log.write(`turn/end failed session=${session.id} error=${error.message}`)
    }
  })

  connect()
  sync.recover().catch((error) => {
    log.write(`recover failed error=${error.message}`)
  })
}
