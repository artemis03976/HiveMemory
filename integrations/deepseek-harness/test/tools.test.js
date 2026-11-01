import assert from 'node:assert/strict'
import { test } from 'node:test'

import { currentTurnInfo, memoryToolDefinitions } from '../lib/tools.js'
import { importBody, sessionTurns } from '../lib/turns.js'
import { SessionLog } from './session-log.js'

const TURN_SCHEMA = { anyOf: [{ $ref: '#/$defs/InFlightTurn' }, { type: 'null' }], default: null }

const SERVER_TOOLS = [
  { name: 'memory_search', description: 'search', inputSchema: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] } },
  { name: 'memory_read', description: 'read', inputSchema: { type: 'object', properties: { aliases: { type: 'array' } }, required: ['aliases'] } },
  {
    name: 'memory_write',
    description: 'write',
    inputSchema: {
      type: 'object',
      properties: { content: { type: 'string' }, turn: TURN_SCHEMA },
      required: ['content', 'turn'],
      $defs: { InFlightTurn: { type: 'object' } },
    },
  },
  { name: 'memory_update', description: 'update', inputSchema: { type: 'object', properties: { alias: { type: 'string' }, turn: TURN_SCHEMA }, required: ['alias'] } },
  { name: 'something_else', description: 'other', inputSchema: { type: 'object' } },
]

/** 记录调用并按给定结果应答的 MCP 客户端。 */
function fakeClient(result = { content: [{ type: 'text', text: 'ok' }] }) {
  return {
    calls: [],
    async callTool(name, args) {
      this.calls.push({ name, args })
      return result
    },
  }
}

function definitions(client, turnInfoFor = async () => ({ external_turn_id: '7' })) {
  return Object.fromEntries(memoryToolDefinitions(SERVER_TOOLS, { client, turnInfoFor }).map(d => [d.name, d]))
}

const exec = { agent: { session: { id: 'session-1' } }, signal: new AbortController().signal }

test('only the four memory tools are registered, with the server description', () => {
  const tools = definitions(fakeClient())

  assert.deepEqual(Object.keys(tools), ['memory_search', 'memory_read', 'memory_write', 'memory_update'])
  assert.equal(tools.memory_write.description, 'write')
})

test('the turn parameter is hidden from the model', () => {
  const { memory_write: write } = definitions(fakeClient())

  assert.deepEqual(write.parameters, { type: 'object', properties: { content: { type: 'string' } }, required: ['content'] })
  assert.equal(SERVER_TOOLS[2].inputSchema.properties.turn, TURN_SCHEMA, 'server schema left untouched')
})

test('writes carry the trusted turn and override one the model made up', async () => {
  const client = fakeClient()
  const seen = []
  const tools = definitions(client, async (session) => {
    seen.push(session.id)
    return { external_turn_id: '7' }
  })

  await tools.memory_write.execute({ content: 'fact', turn: { forged: true } }, exec)

  assert.deepEqual(seen, ['session-1'])
  assert.deepEqual(client.calls, [{ name: 'memory_write', args: { content: 'fact', turn: { external_turn_id: '7' } } }])
})

test('searches go out without turn information', async () => {
  const client = fakeClient()

  await definitions(client).memory_search.execute({ query: 'q' }, exec)

  assert.deepEqual(client.calls, [{ name: 'memory_search', args: { query: 'q' } }])
})

test('the tool returns the result text and raises HiveMemory errors to the model', async () => {
  const ok = await definitions(fakeClient()).memory_search.execute({ query: 'q' }, exec)
  const failing = definitions(fakeClient({ content: [{ type: 'text', text: '{"error":"workspace.operation_denied"}' }], isError: true }))

  assert.equal(ok, 'ok')
  await assert.rejects(failing.memory_read.execute({ aliases: ['a'] }, exec), /operation_denied/)
})

test('the turn info at write time identifies the same turn the import later carries', async () => {
  const log = new SessionLog({ id: 'session-1' }).turn(1, 'earlier').startTurn(2).user('write this down')

  const info = await currentTurnInfo({ id: 'session-1' }, { readSession: async () => log.snapshot() })
  log.assistant([{ type: 'text', text: 'saved' }]).endTurn(2)
  const imported = importBody('session-1', sessionTurns(log.snapshot()), 1, 'dsh')

  for (const field of ['source', 'external_conversation_id', 'external_turn_id', 'user_message', 'recent_turns']) {
    assert.deepEqual(info[field], imported[field], field)
  }
})

test('writing outside a running turn is refused', async () => {
  const ended = new SessionLog().turn(1, 'finished')

  for (const log of [ended, new SessionLog()]) {
    const pending = currentTurnInfo({ id: 'session-1' }, { readSession: async () => log.snapshot() })
    await assert.rejects(pending, /没有进行中的一轮/)
  }
})
