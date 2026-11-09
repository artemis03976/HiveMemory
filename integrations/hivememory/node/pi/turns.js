/**
 * pi 会话记录的轮次投影：结束标记、一轮的条目、导入请求与写入时的轮次信息。
 *
 * pi 的会话记录是一棵树：条目以 `id` 与 `parentId` 相连，`/tree` 回到较早的
 * 条目继续时，离开的分支仍留在同一个文件中；fork 与 clone 把当前路径上的条目
 * （连同条目标识）复制进一个新会话。pi 没有与 HiveMemory 一轮对应的持久标识，
 * 扩展为每一轮写一条结束标记（`custom` 条目，不进入模型上下文），数据为：
 *
 * - `turn`：轮次标识，开始运行时生成，写入时补上的轮次信息与导入都用它；
 * - `session`：写下标记的会话，fork 与 clone 复制来的标记与新会话不同，不导入；
 * - `from`：开始运行时的叶子条目，这一轮的条目都追加在它之后；
 * - `outcome`：这一轮的结局；
 * - `inputs`：这一轮收到的用户原始输入（技能与模板展开之前），按顺序。
 *
 * 一轮的条目是从结束标记沿 `parentId` 回溯、直到 `from` 之前的那一段，与其他
 * 分支无关。用户消息取原始输入而不取会话记录中的用户消息：技能与模板展开后的
 * 正文不是用户的新内容。条目的映射：
 *
 * - 助手消息的 `text` 块写为助手消息，有正文的 `thinking` 块写为思考，
 *   `toolCall` 块写为工具调用；`model_used` 取最后一条助手消息的模型；
 * - 工具结果消息按 `toolCallId` 补全到同一轮的调用上，作为它的结局（`isError`
 *   为真时状态为 `error`，否则为 `success`），找不到调用的结果跳过；
 * - 用户消息、系统消息、其他扩展注入的消息、压缩与分支摘要、上下文编辑与
 *   模型切换等条目不读取。
 */

import { RECENT_TURNS, toolCallFields, truncate } from '../core/memory-tools.js'

// 导入轮次与写入时轮次信息中的来源标识。
export const SOURCE = 'pi'

// 结束标记的 `customType`。
export const MARKER_TYPE = 'hivememory-turn'

/**
 * 一轮的结局：被中断为 `cancelled`；否则按最后一条助手消息的结束原因，出错与
 * 输出达到上限（此时 pi 已经做过恢复）为 `failed`，没有助手消息时同样为
 * `failed`，其余为 `completed`。
 */
export function outcomeOf(aborted, stopReason) {
  if (aborted || stopReason === 'aborted') return 'cancelled'
  if (stopReason === undefined || stopReason === 'error' || stopReason === 'length') return 'failed'
  return 'completed'
}

/** 从 `leafId` 回溯到 `fromId`（不含）的条目，按先后排列；`fromId` 为空时回溯到根。 */
export function branchSince(entries, leafId, fromId) {
  const byId = indexEntries(entries)
  return chain(byId, leafId, fromId)
}

/** 一段条目中最后一条助手消息的结束原因；没有助手消息时为 `undefined`。 */
export function lastStopReason(entries) {
  const assistant = entries.findLast(entry => entry.type === 'message' && entry.message?.role === 'assistant')
  return assistant?.message.stopReason
}

/**
 * 本会话写下的结束标记对应的轮次，按标记在记录中的先后排列。
 *
 * 每一轮为 `{ turnId, from, outcome, userMessage, events, modelUsed }`。
 */
export function sessionTurns(entries, sessionId) {
  const byId = indexEntries(entries)
  return entries
    .filter(entry => isOwnMarker(entry, sessionId))
    .map(marker => projectTurn(byId, marker))
}

/** 已结束一轮的导入请求体。 */
export function importBody(entries, sessionId, turn, agentId) {
  const byId = indexEntries(entries)
  return {
    ...turnInfo(byId, sessionId, turn.turnId, turn.userMessage, turn.from),
    agent_id: agentId,
    outcome: turn.outcome,
    events: turn.events,
    model_used: turn.modelUsed,
  }
}

/**
 * 进行中一轮的轮次信息，写入与更新时交给 HiveMemory。
 *
 * 与这一轮结束后导入时的同名字段由同一个函数从同样的条目构造，服务端因此推导出
 * 同一个交互标识。运行中追加的输入会使导入时的用户消息更长，交互标识不含用户
 * 消息，不受影响。
 */
export function inFlightTurnInfo(entries, sessionId, run) {
  return turnInfo(indexEntries(entries), sessionId, run.turn, run.inputs.join('\n\n'), run.from)
}

/**
 * 一轮的来源、外部标识、用户消息与之前几轮。之前几轮取当前分支上、这一轮开始
 * 之前最近的本会话结束标记对应的轮次。
 */
function turnInfo(byId, sessionId, turnId, userMessage, fromId) {
  const previous = []
  let id = fromId
  while (id && previous.length < RECENT_TURNS) {
    const entry = byId.get(id)
    if (entry === undefined) break
    if (isOwnMarker(entry, sessionId)) previous.unshift(projectTurn(byId, entry))
    id = entry.parentId
  }
  return {
    source: SOURCE,
    external_conversation_id: sessionId,
    external_turn_id: turnId,
    user_message: userMessage,
    recent_turns: previous.map(earlier => ({
      user_message: truncate(earlier.userMessage),
      assistant_text: truncate(assistantText(earlier)),
      // 服务端据此查出上一轮实际所在的话题，话题路由失败时续接。
      external_turn_id: earlier.turnId,
    })),
  }
}

function projectTurn(byId, marker) {
  const data = marker.data
  const turn = new TurnBuilder()
  for (const entry of chain(byId, marker.parentId, data.from)) {
    if (entry.type !== 'message') continue
    if (entry.message.role === 'assistant') turn.addAssistantMessage(entry.message)
    else if (entry.message.role === 'toolResult') turn.addToolResult(entry.message)
  }
  return {
    turnId: data.turn,
    from: data.from ?? null,
    outcome: data.outcome,
    userMessage: (data.inputs ?? []).join('\n\n'),
    events: turn.events,
    modelUsed: turn.modelUsed,
  }
}

class TurnBuilder {
  constructor() {
    this.events = []
    this.modelUsed = null
  }

  addAssistantMessage(message) {
    if (typeof message.model === 'string' && message.model) this.modelUsed = message.model
    for (const block of message.content ?? []) {
      if (block.type === 'text' && block.text?.trim()) {
        this.addEvent('assistant_message', block.text)
      } else if (block.type === 'thinking' && block.thinking?.trim()) {
        this.addEvent('thought', block.thinking)
      } else if (block.type === 'toolCall') {
        this.addEvent('tool_call', '', toolCallFields(block.id, block.name, block.arguments))
      }
    }
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
}

function isOwnMarker(entry, sessionId) {
  return entry.type === 'custom' && entry.customType === MARKER_TYPE && entry.data?.session === sessionId
}

function indexEntries(entries) {
  return new Map(entries.map(entry => [entry.id, entry]))
}

function chain(byId, leafId, fromId) {
  const entries = []
  let id = leafId
  while (id && id !== fromId) {
    const entry = byId.get(id)
    if (entry === undefined) break
    entries.push(entry)
    id = entry.parentId
  }
  return entries.reverse()
}

/** 一轮的回复正文：按顺序以空行拼接全部助手正文，只作为后续轮次的最近上文。 */
function assistantText(turn) {
  return turn.events
    .filter(event => event.kind === 'assistant_message')
    .map(event => event.content)
    .join('\n\n')
}

/** 工具结果的正文：文本原样，图片写占位。 */
function contentText(content) {
  if (typeof content === 'string') return content
  const parts = []
  for (const block of content ?? []) {
    if (block.type === 'text') parts.push(block.text)
    else if (block.type === 'image') parts.push('[image]')
  }
  return parts.filter(Boolean).join('\n')
}
