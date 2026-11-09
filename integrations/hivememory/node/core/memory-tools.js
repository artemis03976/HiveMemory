/**
 * 记忆工具的回声规则与最近轮次的截断，与 harness 无关。
 *
 * 连接器自己注册的四个记忆工具在回流时归一化为 HiveMemory 能力调用，只写动词
 * 与作用对象，不上传工具名：检索为 `SEARCH`，查询留在参数中；读取为 `READ`，
 * 参数 `aliases` 移出参数、作为作用对象；写入为 `WRITE`，没有作用对象；更新为
 * `UPDATE`，参数 `alias` 移出参数、作为唯一的作用对象。检索与读取取回的是已有
 * 记忆，写入与更新的内容已经由显式写入保存，HiveMemory 生成时都不会把它们再
 * 提取一遍。工具名就是 HiveMemory MCP 工具的名字；工具参数改名时，这张表与
 * HiveMemory 的 MCP server 同时修改。其他工具写工具名与完整参数。
 */

// 最近若干轮供 Gateway 理解上文；数量与服务端截取的最近轮数一致。
export const RECENT_TURNS = 3

// 最近轮次只供 Gateway 理解上文，每段正文截断到这个长度（按字符计），避免
// 一轮过大的内容拖着后续轮次一起超过服务端的请求体上限。
const RECENT_TEXT_LIMIT = 2000

// 写入与更新工具中由连接器填写的轮次信息参数，与 HiveMemory MCP server 的参数名一致。
export const TURN_PARAM = 'turn'

/**
 * 记忆工具对应的 HiveMemory 能力：动词与作用对象所在的参数。
 * `single` 表示作用对象参数是单个 alias，否则是 alias 列表。
 */
export const MEMORY_TOOLS = {
  memory_search: { verb: 'SEARCH' },
  memory_read: { verb: 'READ', targetParam: 'aliases', single: false },
  memory_write: { verb: 'WRITE', needsTurn: true },
  memory_update: { verb: 'UPDATE', targetParam: 'alias', single: true, needsTurn: true },
}

/**
 * 一次工具调用写入导入事件的字段（不含事件种类、序号与结局）。
 *
 * `args` 是模型给出的参数：对象原样复制，JSON 字符串先解析，其他值整体放在
 * `input` 下。本连接器的记忆工具去掉 `turn`（它由连接器补上，不属于这次调用的
 * 内容）并写动词与作用对象；作用对象形状不对时 HiveMemory 已以工具错误拒绝这次
 * 调用，按普通工具写出，因为服务端要求读取与更新带作用对象，否则会拒绝整轮。
 */
export function toolCallFields(callId, name, args) {
  const params = argumentObject(args)
  const memoryTool = MEMORY_TOOLS[name]
  if (memoryTool === undefined) return { action_id: callId, tool_name: name, tool_args: params }
  delete params[TURN_PARAM]
  let targets
  if (memoryTool.targetParam !== undefined) {
    targets = memoryTargets(memoryTool, params[memoryTool.targetParam])
    if (targets === null) return { action_id: callId, tool_name: name, tool_args: params }
    delete params[memoryTool.targetParam]
  }
  return {
    action_id: callId,
    verb: memoryTool.verb,
    targets,
    tool_args: Object.keys(params).length > 0 ? params : undefined,
  }
}

/** 按字符截断，不切开代理对。 */
export function truncate(text) {
  const characters = Array.from(text)
  return characters.length <= RECENT_TEXT_LIMIT ? text : characters.slice(0, RECENT_TEXT_LIMIT).join('')
}

function argumentObject(raw) {
  let value = raw
  if (typeof raw === 'string') {
    try {
      value = JSON.parse(raw)
    } catch {
      return { input: raw }
    }
  }
  return isPlainObject(value) ? { ...value } : { input: value }
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
