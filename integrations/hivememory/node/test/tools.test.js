import assert from 'node:assert/strict'
import { test } from 'node:test'

import { memoryToolDefinitions } from '../pi/tools.js'

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

function definitions(client, turnInfo = () => ({ external_turn_id: 't7' })) {
  return Object.fromEntries(memoryToolDefinitions(SERVER_TOOLS, { client, turnInfo }).map(d => [d.name, d]))
}

test('the four memory tools are offered to the model only, with the server description', () => {
  const tools = definitions(fakeClient())

  assert.deepEqual(Object.keys(tools), ['memory_search', 'memory_read', 'memory_write', 'memory_update'])
  assert.equal(tools.memory_write.description, 'write')
  for (const tool of Object.values(tools)) assert.equal(tool.exposure, 'model-only')
})

test('no tool is offered when the server lacks one of the four', () => {
  const partial = SERVER_TOOLS.filter(tool => tool.name !== 'memory_update')

  assert.deepEqual(memoryToolDefinitions(partial, { client: fakeClient(), turnInfo: () => ({}) }), [])
})

test('the turn parameter is hidden from the model', () => {
  const tools = definitions(fakeClient())

  assert.equal('turn' in tools.memory_write.parameters.properties, false)
  assert.deepEqual(tools.memory_write.parameters.required, ['content'])
  assert.equal('turn' in SERVER_TOOLS[2].inputSchema.properties, true)
})

test('writes carry the trusted turn and override one the model made up', async () => {
  const client = fakeClient()
  const ctx = { sessionManager: {} }
  let seen = null
  const tools = definitions(client, c => {
    seen = c
    return { external_turn_id: 't7' }
  })

  await tools.memory_write.execute('c1', { content: 'note', turn: { external_turn_id: 'forged' } }, undefined, undefined, ctx)

  assert.deepEqual(client.calls, [{ name: 'memory_write', args: { content: 'note', turn: { external_turn_id: 't7' } } }])
  assert.equal(seen, ctx)
})

test('searches go out without turn information', async () => {
  const client = fakeClient()

  await definitions(client).memory_search.execute('c1', { query: 'q', turn: { x: 1 } }, undefined, undefined, {})

  assert.deepEqual(client.calls, [{ name: 'memory_search', args: { query: 'q' } }])
})

test('the result text goes back to the model and HiveMemory errors are marked as errors', async () => {
  const ok = await definitions(fakeClient()).memory_search.execute('c1', { query: 'q' }, undefined, undefined, {})
  const failed = await definitions(fakeClient({ content: [{ type: 'text', text: 'bad alias' }], isError: true }))
    .memory_read.execute('c1', { aliases: ['a'] }, undefined, undefined, {})

  assert.deepEqual(ok.content, [{ type: 'text', text: 'ok' }])
  assert.equal(ok.isError, false)
  assert.deepEqual(failed.content, [{ type: 'text', text: 'bad alias' }])
  assert.equal(failed.isError, true)
})

test('writing outside a running turn is refused before reaching HiveMemory', async () => {
  const client = fakeClient()
  const tools = definitions(client, () => {
    throw new Error('not in a turn')
  })

  await assert.rejects(tools.memory_write.execute('c1', { content: 'x' }, undefined, undefined, {}), /not in a turn/)
  assert.deepEqual(client.calls, [])
})
