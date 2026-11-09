import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'

import { install } from '../pi/index.js'
import { MARKER_TYPE } from '../pi/turns.js'
import { PiSession } from './pi-session.js'

const TURN_SCHEMA = { anyOf: [{ type: 'object' }, { type: 'null' }], default: null }
const SERVER_TOOLS = [
  { name: 'memory_search', inputSchema: { type: 'object', properties: { query: { type: 'string' } } } },
  { name: 'memory_read', inputSchema: { type: 'object', properties: { aliases: { type: 'array' } } } },
  { name: 'memory_write', inputSchema: { type: 'object', properties: { content: { type: 'string' }, turn: TURN_SCHEMA } } },
  { name: 'memory_update', inputSchema: { type: 'object', properties: { alias: { type: 'string' }, turn: TURN_SCHEMA } } },
]

/** HiveMemory 的替身：MCP 端点与导入端点，记录工具调用与导入的轮次。 */
function fakeHiveMemory({ reachable = true } = {}) {
  const server = { toolCalls: [], imports: [] }
  server.fetch = async (url, init) => {
    if (!reachable) throw new TypeError('fetch failed')
    const body = JSON.parse(init.body)
    if (url.endsWith('/api/v1/import/turns')) {
      server.imports.push(body)
      return Response.json({ status: 'accepted' })
    }
    let result = {}
    if (body.method === 'initialize') result = { instructions: 'how to use HiveMemory' }
    if (body.method === 'tools/list') result = { tools: SERVER_TOOLS }
    if (body.method === 'tools/call') {
      server.toolCalls.push(body.params)
      result = { content: [{ type: 'text', text: 'draft_x' }] }
    }
    return Response.json({ jsonrpc: '2.0', id: body.id, result })
  }
  return server
}

/** pi 扩展 API 的替身：登记事件处理与工具，按 pi 的方式把条目追加到会话中。 */
class FakePi {
  constructor(session) {
    this.session = session
    this.handlers = {}
    this.tools = {}
    this.sessionManagerOverrides = {}
  }

  on(event, handler) {
    ;(this.handlers[event] ??= []).push(handler)
  }

  registerTool(definition) {
    this.tools[definition.name] = definition
  }

  appendEntry(customType, data) {
    this.session.append({ type: 'custom', customType, data })
  }

  get ctx() {
    return { sessionManager: { ...this.session.sessionManager, ...this.sessionManagerOverrides } }
  }

  async emit(event, payload = {}) {
    for (const handler of this.handlers[event] ?? []) await handler(payload, this.ctx)
  }

  /** 开始运行：pi 先给 `input`，再给 `before_agent_start`，之后写入用户消息。 */
  async start(text, { source = 'interactive' } = {}) {
    await this.emit('input', { text, source })
    const event = { messageType: 'user', prompt: text, systemPromptOptions: { sections: {} } }
    await this.emit('before_agent_start', event)
    this.session.user(text)
    return event
  }

  async callTool(name, params, callId = 'c1') {
    this.session.assistant([{ type: 'toolCall', id: callId, name, arguments: params }], { stopReason: 'toolUse' })
    const result = await this.tools[name].execute(callId, params, undefined, undefined, this.ctx)
    this.session.toolResult(callId, result.content[0].text)
  }
}

function setup(server = fakeHiveMemory()) {
  const session = new PiSession()
  const pi = new FakePi(session)
  const env = { HIVEMEMORY_URL: 'http://hm.test', HIVEMEMORY_CONNECTOR_STATE_DIR: mkdtempSync(join(tmpdir(), 'pi-ext-')) }
  install(pi, env, server.fetch)
  return { session, pi, server }
}

const markers = session => session.entries.filter(entry => entry.customType === MARKER_TYPE).map(entry => entry.data)

test('a turn with a memory write is marked, imported, and carries the same identity as the write', async () => {
  const { session, pi, server } = setup()
  await pi.emit('session_start', { reason: 'startup' })

  const event = await pi.start('remember this')
  assert.deepEqual(Object.keys(pi.tools).sort(), ['memory_read', 'memory_search', 'memory_update', 'memory_write'])
  assert.equal(event.systemPromptOptions.sections.hivememory, 'how to use HiveMemory')
  await pi.callTool('memory_write', { content: 'note', turn: { external_turn_id: 'forged' } })
  session.assistant([{ type: 'text', text: 'saved' }])
  await pi.emit('agent_settled', { aborted: false })
  await pi.emit('session_shutdown', { reason: 'quit' })

  const [marker] = markers(session)
  assert.equal(marker.outcome, 'completed')
  assert.deepEqual(marker.inputs, ['remember this'])
  const writeTurn = server.toolCalls[0].arguments.turn
  assert.equal(writeTurn.external_turn_id, marker.turn)
  const [imported] = server.imports
  assert.equal(imported.external_turn_id, marker.turn)
  assert.equal(imported.external_conversation_id, writeTurn.external_conversation_id)
  assert.equal(imported.user_message, 'remember this')
  assert.deepEqual(imported.events.find(e => e.kind === 'tool_call').verb, 'WRITE')
})

test('steering and follow-up inputs join the running turn; an aborted run is cancelled', async () => {
  const { session, pi, server } = setup()
  await pi.emit('session_start', { reason: 'startup' })

  await pi.start('first')
  await pi.emit('input', { text: 'steer', source: 'rpc', streamingBehavior: 'steer' })
  await pi.emit('input', { text: 'from another extension', source: 'extension', streamingBehavior: 'followUp' })
  session.assistant([], { stopReason: 'aborted' })
  await pi.emit('agent_settled', { aborted: true })
  await pi.emit('session_shutdown', { reason: 'quit' })

  assert.deepEqual(markers(session)[0].inputs, ['first', 'steer'])
  assert.equal(server.imports[0].outcome, 'cancelled')
  assert.equal(server.imports[0].user_message, 'first\n\nsteer')
})

test('each run gets its own turn and a failed run is imported as failed', async () => {
  const { session, pi, server } = setup()
  await pi.emit('session_start', { reason: 'startup' })

  await pi.start('one')
  session.assistant([{ type: 'text', text: 'ok' }])
  await pi.emit('agent_settled', { aborted: false })
  await pi.start('two')
  session.assistant([], { stopReason: 'error' })
  await pi.emit('agent_settled', { aborted: false })
  await pi.emit('session_shutdown', { reason: 'quit' })

  const [first, second] = markers(session)
  assert.notEqual(first.turn, second.turn)
  assert.deepEqual(server.imports.map(body => body.outcome), ['completed', 'failed'])
  assert.deepEqual(server.imports[1].recent_turns.map(turn => turn.external_turn_id), [first.turn])
})

test('pi keeps working without HiveMemory: no tools, and the turn waits for the next start', async () => {
  const down = fakeHiveMemory({ reachable: false })
  const { session, pi } = setup(down)
  await pi.emit('session_start', { reason: 'startup' })

  await pi.start('offline')
  session.assistant([{ type: 'text', text: 'ok' }])
  await pi.emit('agent_settled', { aborted: false })
  await pi.emit('session_shutdown', { reason: 'quit' })

  assert.deepEqual(pi.tools, {})
  assert.equal(markers(session).length, 1)
})

test('a run without a session file writes no marker', async () => {
  const { session, pi, server } = setup()
  pi.sessionManagerOverrides = { getSessionFile: () => undefined }
  await pi.emit('session_start', { reason: 'startup' })

  await pi.start('ephemeral')
  session.assistant([{ type: 'text', text: 'ok' }])
  await pi.emit('agent_settled', { aborted: false })
  await pi.emit('session_shutdown', { reason: 'quit' })

  assert.equal(markers(session).length, 0)
  assert.deepEqual(server.imports, [])
})
