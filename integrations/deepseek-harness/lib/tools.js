/**
 * 记忆工具：把 HiveMemory MCP 端点声明的检索、读取、写入与更新注册为 dsh 原生工具。
 *
 * 工具名、说明与参数 schema 取自服务端的工具列表，插件不另写一份；模型看到的
 * 工具名就是 HiveMemory 的工具名。每次调用经 MCP 端点转交 HiveMemory，结果的
 * 正文原样交给模型，HiveMemory 的工具错误作为 dsh 的工具错误交给模型。
 *
 * 写入与更新必须带上这一轮的信息，HiveMemory 据此在写入时为这一轮落位。dsh
 * 不允许在工具执行前改写模型给出的参数，插件因此在自己的工具中，以调用方
 * Agent 的会话构造轮次信息再交给 HiveMemory：参数 `turn` 从交给模型的 schema
 * 中去掉，模型无法填写，也无法绕过；子 Agent 中的调用以子 Agent 自己的会话与
 * 当前一轮落位。
 */

import { MEMORY_TOOLS, sessionTurns, TURN_PARAM, turnInfo } from './turns.js'

/**
 * 服务端工具列表中本插件认识的工具，转为 dsh 工具定义。
 *
 * `turnInfoFor(session)` 构造调用方会话当前一轮的轮次信息。
 */
export function memoryToolDefinitions(serverTools, { client, turnInfoFor }) {
  return serverTools
    .filter(tool => Object.hasOwn(MEMORY_TOOLS, tool.name))
    .map(tool => toolDefinition(tool, { client, turnInfoFor }))
}

/**
 * 调用方会话当前一轮的轮次信息，与这一轮结束后导入时的同名字段相同。
 *
 * 工具在一轮之中执行，调用方会话实时记录的最后一轮就是当前一轮，它还没有
 * 结束；之前几轮与导入时一样取该轮之前最近的已结束轮次，服务端因此推导出
 * 同一个交互标识。最后一轮已经结束（或会话还没有轮次）时抛出错误，由 dsh
 * 作为工具错误交给模型。
 */
export async function currentTurnInfo(session, { readSession }) {
  const turns = sessionTurns(await readSession(session.id))
  const index = turns.length - 1
  if (index === -1 || turns[index].closed) {
    throw new Error('当前没有进行中的一轮，无法确定写入属于哪一轮')
  }
  return turnInfo(session.id, turns, index)
}

function toolDefinition(tool, { client, turnInfoFor }) {
  const needsTurn = MEMORY_TOOLS[tool.name].needsTurn === true
  return {
    name: tool.name,
    description: tool.description ?? '',
    parameters: withoutTurn(tool.inputSchema),
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    async execute(args, exec) {
      const request = typeof args === 'object' && args !== null && !Array.isArray(args) ? { ...args } : {}
      delete request[TURN_PARAM]
      if (needsTurn) request[TURN_PARAM] = await turnInfoFor(exec.agent.session)
      const result = await client.callTool(tool.name, request, exec.signal)
      const text = resultText(result)
      if (result.isError === true) throw new Error(text)
      return text
    },
  }
}

/** 去掉轮次信息参数；`$defs` 只承载轮次信息的模型，一并去掉。 */
function withoutTurn(schema) {
  const copy = structuredClone(schema ?? { type: 'object', properties: {} })
  if (copy.properties !== undefined) delete copy.properties[TURN_PARAM]
  if (Array.isArray(copy.required)) copy.required = copy.required.filter(name => name !== TURN_PARAM)
  delete copy.$defs
  return copy
}

function resultText(result) {
  return (result.content ?? [])
    .filter(block => block.type === 'text')
    .map(block => block.text)
    .join('\n')
}
