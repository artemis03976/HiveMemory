import assert from 'node:assert/strict'
import { test } from 'node:test'

import { HiveMemoryClient } from '../core/client.js'
import { configFromEnv } from '../core/config.js'
import { HARNESS } from '../pi/index.js'

const CONFIG = {
  principalId: 'harness:pi',
  clientName: 'hivememory-pi-connector',
  url: 'http://hm.test',
  agentId: 'pi',
  userId: 'u1',
  token: 'secret',
  timeoutMs: 1000,
}

/** 记录请求并按给定状态与响应体应答的 fetch；`status` 为 `null` 时模拟网络错误。 */
function fakeFetch(status, payload = {}) {
  const requests = []
  const fetchImpl = async (url, init) => {
    requests.push({ url, headers: init.headers, body: JSON.parse(init.body) })
    if (status === null) throw new TypeError('fetch failed')
    return new Response(typeof payload === 'string' ? payload : JSON.stringify(payload), { status })
  }
  return { fetchImpl, requests }
}

test('pi reads its own credential variable, agent and state directory', () => {
  const config = configFromEnv(HARNESS, {
    HIVEMEMORY_PI_TOKEN: ' pi-secret\n',
    HIVEMEMORY_CODEX_TOKEN: 'codex-secret',
    HIVEMEMORY_URL: 'http://127.0.0.1:9000/',
  })

  assert.equal(config.token, 'pi-secret')
  assert.equal(config.principalId, 'harness:pi')
  assert.equal(config.agentId, 'pi')
  assert.equal(config.url, 'http://127.0.0.1:9000')
  assert.match(config.stateDir, /pi-connector$/)
  assert.equal(configFromEnv(HARNESS, { HIVEMEMORY_CODEX_TOKEN: 'codex-secret' }).token, null)
})

test('submit sends the principal, credential and user with the turn', async () => {
  const { fetchImpl, requests } = fakeFetch(200, { status: 'accepted' })

  const result = await new HiveMemoryClient(CONFIG, fetchImpl).submit({ external_turn_id: 't1' })

  assert.deepEqual(result, { kind: 'confirmed', status: 200, importStatus: 'accepted' })
  assert.equal(requests[0].url, 'http://hm.test/api/v1/import/turns')
  assert.equal(requests[0].headers['X-Principal-ID'], 'harness:pi')
  assert.equal(requests[0].headers.Authorization, 'Bearer secret')
  assert.equal(requests[0].headers['X-User-ID'], 'u1')
})

test('submit results are classified by status', async () => {
  const cases = [
    [null, 'retryable'],
    [429, 'retryable'],
    [503, 'retryable'],
    [413, 'rejected'],
    [422, 'rejected'],
    [401, 'misconfigured'],
    [403, 'misconfigured'],
    [404, 'misconfigured'],
    [500, 'server_error'],
  ]
  for (const [status, kind] of cases) {
    const { fetchImpl } = fakeFetch(status, { error: 'e', reason: 'r' })
    const result = await new HiveMemoryClient(CONFIG, fetchImpl).submit({})
    assert.equal(result.kind, kind, `status ${status}`)
  }
})

test('submit propagates the caller cancellation instead of classifying it', async () => {
  const controller = new AbortController()
  controller.abort()
  const fetchImpl = async (_url, init) => {
    init.signal.throwIfAborted()
  }

  await assert.rejects(new HiveMemoryClient(CONFIG, fetchImpl).submit({}, controller.signal), { name: 'AbortError' })
})

test('tool calls go to the MCP endpoint with the agent and the client name', async () => {
  const { fetchImpl, requests } = fakeFetch(200, { jsonrpc: '2.0', id: 1, result: { instructions: 'use it' } })

  const instructions = await new HiveMemoryClient(CONFIG, fetchImpl).instructions()

  assert.equal(instructions, 'use it')
  assert.equal(requests[0].url, 'http://hm.test/mcp')
  assert.equal(requests[0].headers['X-Agent-ID'], 'pi')
  assert.equal(requests[0].body.params.clientInfo.name, 'hivememory-pi-connector')
})

test('MCP failures are raised with a readable reason', async () => {
  const unreachable = fakeFetch(null)
  const failing = fakeFetch(200, { jsonrpc: '2.0', id: 1, error: { message: 'bad request' } })

  await assert.rejects(new HiveMemoryClient(CONFIG, unreachable.fetchImpl).listTools(), /暂时无法访问/)
  await assert.rejects(new HiveMemoryClient(CONFIG, failing.fetchImpl).listTools(), /bad request/)
})
