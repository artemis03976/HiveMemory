/**
 * dsh 会话记录的轮次切分与映射：把会话事件投影为 Import Bus 的导入轮次与在途轮次。
 *
 * 读取的是 `sessionQuery.readSession` 返回的逻辑事件流（会话头、fork 继承的
 * 事件数与完整事件），不读取会话文件。依赖的结构：
 *
 * - 一轮从 `turn/start` 到同一轮号的 `turn/end`，轮号在会话内唯一且递增（恢复与
 *   fork 都接着已有的最大轮号编号）。外部会话标识取会话标识，外部轮次标识取
 *   轮号。没有 `turn/end` 的一轮仍在进行。
 * - fork 出的会话以继承的前缀开头，序号小于 `inheritedEventCount` 的事件属于
 *   来源会话，不读取；继承前缀中未结束的一轮由 fork 补上的收尾结束，它的
 *   `turn/start` 在前缀中，因此不形成轮次。
 * - 子 Agent 的会话（会话头的来源为 `subagent`）不导入：子 Agent 是临时的
 *   执行线程，它的过程不进入记忆生成，要保留的内容由它自己写入记忆；它交回
 *   父 Agent 的消息随父会话的一轮导入。
 * - `user/message` 中来源为 `user` 的是人发出的消息（一轮的提示与运行中追加的
 *   消息），按顺序以空行拼接为这一轮的用户消息。来源为 `subagent-settled` 的
 *   是子 Agent 结算后 dsh 发出的通知，带它的结局与最后的输出，是子 Agent 交回
 *   的最终结果，按出现位置写为 `subagent_message` 事件。子 Agent 运行中用
 *   `send_message` 发来的消息（来源 `agent-message`）不读取：结算通知已经带有
 *   它最后的输出，两者都读会让同一份报告重复进入记忆生成。其他来源是 harness
 *   注入的上下文（技能正文、目录说明、压缩摘要、后台任务通知等），同样不读取。
 *   一轮可能没有人发出的消息（例如由子 Agent 的结算通知唤醒），用户消息因此
 *   为空。
 * - `assistant/message` 的内容块：`text` → 助手消息，`reasoning` → 思考（没有
 *   正文的跳过），`tool-call` 块不读取，调用以独立的 `tool/call` 事件为准；
 *   `model_used` 取最后一条助手消息的模型。没有形成助手消息的尝试
 *   （`assistant/attempt`）不读取。
 * - `tool/call` → 工具调用（调用标识、工具名与模型给出的参数）；`tool/result`
 *   按 `toolCallId` 补全到同一轮的调用上，作为它的结局（`isError` 为真时状态为
 *   `error`，否则为 `success`），找不到调用的结果跳过。
 * - 本插件的四个记忆工具在这里归一化为 HiveMemory 能力调用，只写动词与作用
 *   对象，不上传工具名：检索为 `SEARCH`，查询留在参数中；读取为 `READ`，参数
 *   `aliases` 移出参数、作为作用对象；写入为 `WRITE`，没有作用对象；更新为
 *   `UPDATE`，参数 `alias` 移出参数、作为唯一的作用对象。检索与读取取回的是
 *   已有记忆，写入与更新的内容已经由显式写入保存，HiveMemory 生成时都不会把
 *   它们再提取一遍。插件注册的工具名就是 HiveMemory MCP 工具的名字，dsh 工具
 *   注册表中同名的工具只能有一个；工具参数改名时，识别表与 HiveMemory 的
 *   MCP server 同时修改。其他工具写工具名与完整参数。
 * - 不认识的事件类型与内容块一律跳过，不报错。
 */

import { SOURCE } from './config.js'

// 最近若干轮供 Gateway 理解上文；数量与服务端截取的最近轮数一致。
export const RECENT_TURNS = 3

// 最近轮次只供 Gateway 理解上文，每段正文截断到这个长度（按字符计），避免
// 一轮过大的内容拖着后续轮次一起超过服务端的请求体上限。
const RECENT_TEXT_LIMIT = 2000

// 写入与更新工具中由插件填写的轮次信息参数，与 HiveMemory MCP server 的参数名一致。
export const TURN_PARAM = 'turn'

// 子 Agent 结算后 dsh 发给父 Agent 的通知的来源，通知带子 Agent 最后的输出。
const SUBAGENT_SETTLED_SOURCE = 'subagent-settled'

/**
 * 本插件的记忆工具对应的 HiveMemory 能力：动词与作用对象所在的参数。
 * `single` 表示作用对象参数是单个 alias，否则是 alias 列表。
 */
export const MEMORY_TOOLS = {
  memory_search: { verb: 'SEARCH' },
  memory_read: { verb: 'READ', targetParam: 'aliases', single: false },
  memory_write: { verb: 'WRITE', needsTurn: true },
  memory_update: { verb: 'UPDATE', targetParam: 'alias', single: true, needsTurn: true },
}

/**
 * dsh 轮次结束原因到 HiveMemory 结局。完成之外的原因都表示这一轮的材料不完整：
 * 被取消或崩溃后收尾的一轮为 `cancelled`，出错、被阻止或输出达到上限的一轮为
 * `failed`。fork 收尾只出现在继承前缀中未结束的一轮上，该轮不形成轮次，列出
 * 只为覆盖全部原因。插件扩展的未知原因按 `failed`。
 */
const OUTCOMES = {
  completed: 'completed',
  aborted: 'cancelled',
  interrupted: 'cancelled',
  forked: 'cancelled',
  error: 'failed',
  blocked: 'failed',
  'max-tokens': 'failed',
}

/** dsh 为被委托的子 Agent 创建的会话：会话头的来源为 `subagent`，`parentSession` 是直接父会话。 */
export function isSubagentSession(header) {
  return header?.origin === 'subagent'
}

/**
 * 切分一份会话记录中本会话自己的轮次，按先后排列。
 *
 * 每一轮为 `{ number, closed, reason, userMessage, events, modelUsed }`：
 * `events` 是用户消息之后的导入事件，序号从 1 起递增；`reason` 是
 * `turn/end` 的结束原因，仍在进行的一轮为 `null`。
 */
export function sessionTurns(snapshot) {
  const turns = []
  let current = null
  for (const event of snapshot.events) {
    if (event.seq < snapshot.inheritedEventCount) continue
    const data = event.data
    if (event.type === 'turn/start') {
      // dsh 的轮次不嵌套：恢复与冷读都会为崩溃时未结束的一轮补上收尾。
      current = new TurnBuilder(data.turn)
    } else if (current === null) {
      continue
    } else if (event.type === 'turn/end') {
      if (data.turn !== current.number) continue
      turns.push(current.build(data.reason))
      current = null
    } else if (event.type === 'user/message') {
      current.addUserMessage(data)
    } else if (event.type === 'assistant/message') {
      current.addAssistantMessage(data.message)
    } else if (event.type === 'tool/call') {
      current.addToolCall(data)
    } else if (event.type === 'tool/result') {
      current.addToolResult(data.message)
    }
  }
  if (current !== null) turns.push(current.build(null))
  return turns
}

/**
 * 一轮的来源、外部标识、用户消息与之前几轮。
 *
 * 导入请求与写入时补上的轮次信息共用这一份：服务端由来源、外部会话与外部
 * 轮次标识推导出同一个交互标识，并以同样的输入为这一轮路由。之前几轮取
 * 该轮之前最近的已结束轮次。
 */
export function turnInfo(sessionId, turns, index) {
  const turn = turns[index]
  const previous = turns.slice(0, index).filter(earlier => earlier.closed).slice(-RECENT_TURNS)
  return {
    source: SOURCE,
    external_conversation_id: sessionId,
    external_turn_id: String(turn.number),
    user_message: turn.userMessage,
    recent_turns: previous.map(earlier => ({
      user_message: truncate(earlier.userMessage),
      assistant_text: truncate(assistantText(earlier)),
      // 服务端据此查出上一轮实际所在的话题，话题路由失败时续接。
      external_turn_id: String(earlier.number),
    })),
  }
}

/** 一轮已结束交互的导入请求体。 */
export function importBody(sessionId, turns, index, agentId) {
  const turn = turns[index]
  return {
    ...turnInfo(sessionId, turns, index),
    agent_id: agentId,
    outcome: OUTCOMES[turn.reason.kind] ?? 'failed',
    events: turn.events,
    model_used: turn.modelUsed,
  }
}

class TurnBuilder {
  constructor(number) {
    this.number = number
    this.userTexts = []
    this.events = []
    this.modelUsed = null
  }

  addUserMessage(message) {
    const source = message.source?.kind
    const text = contentText(message.content)
    if (!text.trim()) return
    if (source === 'user') this.userTexts.push(text)
    else if (source === SUBAGENT_SETTLED_SOURCE) this.addEvent('subagent_message', text)
  }

  addAssistantMessage(message) {
    const model = message.source?.model
    if (typeof model === 'string' && model) this.modelUsed = model
    for (const block of message.content ?? []) {
      if (block.type === 'text' && block.text?.trim()) {
        this.addEvent('assistant_message', block.text)
      } else if (block.type === 'reasoning' && block.text?.trim()) {
        this.addEvent('thought', block.text)
      }
    }
  }

  addToolCall(call) {
    const args = parseArguments(call.arguments)
    const memoryTool = MEMORY_TOOLS[call.name]
    if (memoryTool === undefined) {
      this.addEvent('tool_call', '', { action_id: call.callId, tool_name: call.name, tool_args: args })
      return
    }
    // 轮次信息由插件在调用 HiveMemory 时补上，不属于这次调用的内容。
    delete args[TURN_PARAM]
    let targets
    if (memoryTool.targetParam !== undefined) {
      targets = memoryTargets(memoryTool, args[memoryTool.targetParam])
      if (targets === null) {
        // 模型传错了作用对象，HiveMemory 已以工具错误拒绝这次调用。按普通工具写出：
        // 服务端要求读取与更新带作用对象，否则会拒绝整轮。
        this.addEvent('tool_call', '', { action_id: call.callId, tool_name: call.name, tool_args: args })
        return
      }
      delete args[memoryTool.targetParam]
    }
    this.addEvent('tool_call', '', {
      action_id: call.callId,
      verb: memoryTool.verb,
      targets,
      tool_args: Object.keys(args).length > 0 ? args : undefined,
    })
  }

  addToolResult(message) {
    const call = this.events.find(event => event.kind === 'tool_call' && event.action_id === message.toolCallId)
    if (call === undefined || call.status !== undefined) return
    call.status = message.isError ? 'error' : 'success'
    call.result = contentText(message.content)
  }

  addEvent(kind, content, fields = {}) {
    const event = { kind, sequence: this.events.length + 1, content }
    for (const [key, value] of Object.entries(fields)) {
      if (value !== undefined) event[key] = value
    }
    this.events.push(event)
  }

  build(reason) {
    return {
      number: this.number,
      closed: reason !== null,
      reason,
      userMessage: this.userTexts.join('\n\n'),
      events: this.events,
      modelUsed: this.modelUsed,
    }
  }
}

/** 一轮的回复正文：按顺序以空行拼接全部助手正文，只作为后续轮次的最近上文。 */
function assistantText(turn) {
  return turn.events
    .filter(event => event.kind === 'assistant_message')
    .map(event => event.content)
    .join('\n\n')
}

/** 用户消息与工具结果的正文：文本原样，图片与文件写占位。 */
function contentText(content) {
  const parts = []
  for (const block of content ?? []) {
    if (block.type === 'text') parts.push(block.text)
    else if (block.type === 'image') parts.push('[image]')
    else if (block.type === 'file') parts.push('[file]')
  }
  return parts.filter(Boolean).join('\n')
}

/** 模型给出的参数是 JSON 字符串；不是对象时整体放在 `input` 下。 */
function parseArguments(raw) {
  let value
  try {
    value = JSON.parse(raw)
  } catch {
    return { input: raw }
  }
  return isPlainObject(value) ? value : { input: value }
}

/** 参数中的作用对象；形状不对时为 `null`。 */
function memoryTargets(memoryTool, value) {
  if (memoryTool.single) return typeof value === 'string' && value ? [value] : null
  const valid = Array.isArray(value) && value.length > 0 && value.every(item => typeof item === 'string')
  return valid ? value : null
}

function isPlainObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** 按字符截断，不切开代理对。 */
function truncate(text) {
  const characters = Array.from(text)
  return characters.length <= RECENT_TEXT_LIMIT ? text : characters.slice(0, RECENT_TEXT_LIMIT).join('')
}
