/**
 * 记忆工具：把 HiveMemory MCP 端点声明的检索、读取、写入与更新注册为 pi 工具。
 *
 * 工具名、说明与参数 schema 取自服务端的工具列表，扩展不另写一份；模型看到的
 * 工具名就是 HiveMemory 的工具名。每次调用经 MCP 端点转交 HiveMemory，结果的
 * 正文原样交给模型，HiveMemory 的工具错误作为 pi 的工具错误交给模型。
 *
 * 写入与更新必须带上这一轮的信息，HiveMemory 据此在写入时为这一轮确定话题。
 * 扩展在自己的工具执行时，以当前一轮构造轮次信息再交给 HiveMemory：参数 `turn`
 * 从交给模型的 schema 中去掉，模型填写的也会被覆盖。扩展没有加载时这些工具
 * 不存在，模型也就无法绕过它自己填写轮次信息。
 *
 * 工具只交给模型直接调用（`exposure: "model-only"`），codemode 与其他工具不能
 * 调用它们：经其他工具发起的内层调用不进入会话记录，回流时无法给这些调用写
 * 回声标记，HiveMemory 生成时会把取回的记忆再提取一遍。
 */

import { MEMORY_TOOLS, TURN_PARAM } from '../core/memory-tools.js'

/**
 * 服务端工具列表中的四个记忆工具，转为 pi 工具定义；缺少任何一个时返回空列表，
 * 模型不会只看到一部分记忆工具。
 *
 * `turnInfo(ctx)` 构造调用方所在一轮的轮次信息，不在一轮之中时抛出错误。
 */
export function memoryToolDefinitions(serverTools, { client, turnInfo }) {
  const known = serverTools.filter(tool => Object.hasOwn(MEMORY_TOOLS, tool.name))
  if (known.length !== Object.keys(MEMORY_TOOLS).length) return []
  return known.map(tool => toolDefinition(tool, { client, turnInfo }))
}

function toolDefinition(tool, { client, turnInfo }) {
  const memoryTool = MEMORY_TOOLS[tool.name]
  return {
    name: tool.name,
    label: tool.name,
    description: tool.description ?? '',
    parameters: withoutTurn(tool.inputSchema),
    exposure: 'model-only',
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      const args = { ...params }
      delete args[TURN_PARAM]
      if (memoryTool.needsTurn) args[TURN_PARAM] = turnInfo(ctx)
      const result = await client.callTool(tool.name, args, signal)
      return { content: toolContent(result.content), details: undefined, isError: result.isError === true }
    },
  }
}

/** 交给模型的参数 schema：去掉由扩展填写的轮次信息。 */
function withoutTurn(schema) {
  const copy = structuredClone(schema ?? { type: 'object', properties: {} })
  if (copy.properties) delete copy.properties[TURN_PARAM]
  if (Array.isArray(copy.required)) copy.required = copy.required.filter(name => name !== TURN_PARAM)
  return copy
}

/** MCP 工具结果的正文：文本原样，其他内容写占位。 */
function toolContent(content) {
  const blocks = (content ?? []).map(block => (
    block.type === 'text' ? { type: 'text', text: block.text } : { type: 'text', text: `[${block.type}]` }
  ))
  return blocks.length > 0 ? blocks : [{ type: 'text', text: '' }]
}
