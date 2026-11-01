import assert from 'node:assert/strict'
import { test } from 'node:test'

import { HiveMemoryClient } from '../lib/client.js'

const CONFIG = { url: 'http://hm.test', agentId: 'dsh', userId: 'u1', token: 'secret', timeoutMs: 1000 }

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

test('submit sends the principal, credential and user with the turn', async () => {
  const { fetchImpl, requests } = fakeFetch(200, { status: 'accepted' })

  const result = await new HiveMemoryClient(CONFIG, fetchImpl).submit({ external_turn_id: '1' })

  assert.deepEqual(result, { kind: 'confirmed', status: 200, importStatus: 'accepted' })
  assert.equal(requests[0].url, 'http://hm.test/api/v1/import/turns')
  assert.equal(requests[0].headers['X-Principal-ID'], 'harness:dsh')
  assert.equal(requests[0].headers.Authorization, 'Bearer secret')
  assert.equal(requests[0].headers['X-User-ID'], 'u1')
  assert.deepEqual(requests[0].body, { external_turn_id: '1' })
})

test('submit omits credential and user headers that are not configured', async () => {
  const { fetchImpl, requests } = fakeFetch(200)

  await new HiveMemoryClient({ ...CONFIG, userId: null, token: null }, fetchImpl).submit({})

  assert.equal('Authorization' in requests[0].headers, false)
  assert.equal('X-User-ID' in requests[0].headers, false)
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

test('error code and reason come from the response body, non-JSON bodies are tolerated', async () => {
  const withBody = fakeFetch(503, { error: 'import_bus.unavailable', reason: 'gateway_unavailable' })
  const htmlPage = fakeFetch(502, '<html>bad gateway</html>')

  const detailed = await new HiveMemoryClient(CONFIG, withBody.fetchImpl).submit({})
  const bare = await new HiveMemoryClient(CONFIG, htmlPage.fetchImpl).submit({})

  assert.deepEqual([detailed.error, detailed.reason], ['import_bus.unavailable', 'gateway_unavailable'])
  assert.deepEqual([bare.kind, bare.error], ['retryable', null])
})

test('submit propagates the caller cancellation instead of classifying it', async () => {
  const controller = new AbortController()
  const fetchImpl = async () => {
    controller.abort()
    throw new DOMException('aborted', 'AbortError')
  }

  await assert.rejects(new HiveMemoryClient(CONFIG, fetchImpl).submit({}, controller.signal), { name: 'AbortError' })
})

test('tool calls go to the MCP endpoint with the agent and return the result', async () => {
  const { fetchImpl, requests } = fakeFetch(200, {
    jsonrpc: '2.0',
    id: 1,
    result: { content: [{ type: 'text', text: 'hit' }] },
  })

  const result = await new HiveMemoryClient(CONFIG, fetchImpl).callTool('memory_search', { query: 'q' })

  assert.deepEqual(result, { content: [{ type: 'text', text: 'hit' }] })
  assert.equal(requests[0].url, 'http://hm.test/mcp')
  assert.equal(requests[0].headers['X-Agent-ID'], 'dsh')
  assert.equal(requests[0].headers['X-Principal-ID'], 'harness:dsh')
  assert.deepEqual(requests[0].body.params, { name: 'memory_search', arguments: { query: 'q' } })
})

test('MCP failures are raised with a readable reason', async () => {
  const rpcError = fakeFetch(200, { jsonrpc: '2.0', id: 1, error: { code: -32601, message: 'unsupported' } })
  const httpError = fakeFetch(421)
  const unreachable = fakeFetch(null)

  for (const [fake, pattern] of [[rpcError, /unsupported/], [httpError, /HTTP 421/], [unreachable, /暂时无法访问/]]) {
    await assert.rejects(new HiveMemoryClient(CONFIG, fake.fetchImpl).listTools(), pattern)
  }
})
