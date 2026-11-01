import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'

import { StateStore } from '../lib/state.js'
import { SessionSync } from '../lib/sync.js'
import { SessionLog } from './session-log.js'

const FIRST_RUN = 1_000

/** 按顺序给出提交结果的客户端；结果用完之后一律确认。 */
class FakeClient {
  constructor(results = []) {
    this.results = results
    this.submitted = []
  }

  async submit(body) {
    this.submitted.push(body)
    this.onSubmit?.()
    return this.results.shift() ?? { kind: 'confirmed', status: 200, importStatus: 'accepted' }
  }
}

/** 会话服务的替身：`logs` 中的会话可以读取，`live` 中的会话在本进程中存活。 */
class FakeSessions {
  constructor({ logs = {}, live = [] } = {}) {
    this.logs = logs
    this.live = new Set(live)
    this.flushed = []
  }

  isLive(sessionId) {
    return this.live.has(sessionId)
  }

  async flush(sessionId) {
    this.flushed.push(sessionId)
  }

  async read(sessionId) {
    const log = this.logs[sessionId]
    if (log === undefined) throw Object.assign(new Error('not found'), { code: 'SESSION_QUERY_SESSION_NOT_FOUND' })
    return log.snapshot()
  }
}

function setup({ logs = {}, live = Object.keys(logs), results = [] } = {}) {
  const dir = mkdtempSync(join(tmpdir(), 'hivememory-dsh-'))
  const store = new StateStore(dir)
  const client = new FakeClient(results)
  const sessions = new FakeSessions({ logs, live })
  const log = { lines: [], write(line) { this.lines.push(line) } }
  const sync = new SessionSync({
    store, client, log, sessions, agentId: 'dsh', signal: new AbortController().signal, now: FIRST_RUN,
  })
  return { dir, store, client, sessions, sync }
}

const submittedTurns = client => client.submitted.map(body => body.external_turn_id)

test('closed turns are submitted in order and the watermark follows confirmations', async () => {
  const log = new SessionLog().turn(1, 'a').turn(2, 'b').startTurn(3).user('running')
  const { store, client, sync } = setup({ logs: { 'session-1': log } })
  sync.register('session-1', { createdAt: FIRST_RUN + 1 })

  await sync.request('session-1')

  assert.deepEqual(submittedTurns(client), ['1', '2'])
  assert.equal(client.submitted[0].agent_id, 'dsh')
  assert.equal(store.load('session-1').lastTurn, 2)
})

test('a live session is flushed before it is read', async () => {
  const { sessions, sync } = setup({ logs: { 'session-1': new SessionLog().turn(1, 'a') } })
  sync.register('session-1', { createdAt: FIRST_RUN + 1 })

  await sync.request('session-1')

  assert.deepEqual(sessions.flushed, ['session-1'])
})

test('a retryable failure keeps the watermark and the next sync resends from that turn', async () => {
  const log = new SessionLog().turn(1, 'a').turn(2, 'b')
  const { store, client, sync } = setup({ logs: { 'session-1': log }, results: [{ kind: 'retryable', status: 503 }] })
  sync.register('session-1', { createdAt: FIRST_RUN + 1 })

  await sync.request('session-1')
  const watermark = store.load('session-1').lastTurn
  await sync.request('session-1')

  assert.equal(watermark, 0)
  assert.deepEqual(submittedTurns(client), ['1', '1', '2'])
  assert.equal(store.load('session-1').lastTurn, 2)
})

test('a misconfigured plugin stops without advancing', async () => {
  const { store, client, sync } = setup({
    logs: { 'session-1': new SessionLog().turn(1, 'a').turn(2, 'b') },
    results: [{ kind: 'misconfigured', status: 403 }],
  })
  sync.register('session-1', { createdAt: FIRST_RUN + 1 })

  await sync.request('session-1')

  assert.deepEqual(submittedTurns(client), ['1'])
  assert.equal(store.load('session-1').lastTurn, 0)
})

test('a rejected turn is recorded and skipped', async () => {
  const { dir, store, client, sync } = setup({
    logs: { 'session-1': new SessionLog().turn(1, 'a').turn(2, 'b') },
    results: [{ kind: 'rejected', status: 422, error: 'request.validation_failed', reason: null }],
  })
  sync.register('session-1', { createdAt: FIRST_RUN + 1 })

  await sync.request('session-1')

  const [rejection] = readFileSync(join(dir, 'rejections.jsonl'), 'utf8').trim().split('\n').map(line => JSON.parse(line))
  assert.deepEqual([rejection.session_id, rejection.turn, rejection.status], ['session-1', 1, 422])
  assert.deepEqual(submittedTurns(client), ['1', '2'])
  assert.equal(store.load('session-1').lastTurn, 2)
})

test('the third consecutive server error rejects the turn instead of blocking the session', async () => {
  const serverError = { kind: 'server_error', status: 500 }
  const { store, client, sync } = setup({
    logs: { 'session-1': new SessionLog().turn(1, 'a').turn(2, 'b') },
    results: [serverError, serverError, serverError],
  })
  sync.register('session-1', { createdAt: FIRST_RUN + 1 })

  await sync.request('session-1')
  await sync.request('session-1')
  const beforeThird = store.load('session-1').lastTurn
  await sync.request('session-1')

  assert.equal(beforeThird, 0)
  assert.deepEqual(submittedTurns(client), ['1', '1', '1', '2'])
  assert.deepEqual(store.load('session-1').failures, {})
  assert.equal(store.load('session-1').lastTurn, 2)
})

test('a session created before the first run is not backfilled', async () => {
  const log = new SessionLog().turn(1, 'before install').startTurn(2).user('after install')
  const { store, client, sync } = setup({ logs: { 'session-1': log } })
  sync.register('session-1', { createdAt: FIRST_RUN - 1 })

  await sync.request('session-1')
  log.assistant([{ type: 'text', text: 'done' }]).endTurn(2)
  await sync.request('session-1')

  assert.equal(store.load('session-1').origin, 'existing')
  assert.deepEqual(submittedTurns(client), ['2'])
})

test('a session that is not live here only submits turns seen ending', async () => {
  // The persisted copy shows turn 2 closed, as a cold read balances a crash tail
  // or an open turn still running in another dsh process.
  const log = new SessionLog().turn(1, 'a').turn(2, 'maybe still running')
  const { client, sync } = setup({ logs: { 'session-1': log }, live: [] })
  sync.register('session-1', { createdAt: FIRST_RUN + 1 })
  sync.noteEnded('session-1', 1)

  await sync.request('session-1')

  assert.deepEqual(submittedTurns(client), ['1'])
})

test('recover syncs only sessions with ended turns that are not confirmed', async () => {
  const { client, sync } = setup({
    logs: {
      pending: new SessionLog({ id: 'pending' }).turn(1, 'a'),
      settled: new SessionLog({ id: 'settled' }).turn(1, 'a'),
    },
    live: [],
  })
  sync.register('pending', { createdAt: FIRST_RUN + 1 })
  sync.register('settled', { createdAt: FIRST_RUN + 1 })
  sync.noteEnded('pending', 1)

  await sync.recover()

  assert.deepEqual(client.submitted.map(body => body.external_conversation_id), ['pending'])
})

test('a deleted session forgets its state', async () => {
  const { store, sync } = setup({ logs: {} })
  sync.register('gone', { createdAt: FIRST_RUN + 1 })

  await sync.request('gone')

  assert.equal(store.load('gone'), null)
})

test('a turn that ends while a sync is submitting is picked up by another pass', async () => {
  const log = new SessionLog().turn(1, 'a')
  const { client, sync } = setup({ logs: { 'session-1': log } })
  sync.register('session-1', { createdAt: FIRST_RUN + 1 })
  let second = null
  client.onSubmit = () => {
    if (second !== null) return
    log.turn(2, 'ended while syncing')
    second = sync.request('session-1')
  }

  await sync.request('session-1')
  await second

  assert.deepEqual(submittedTurns(client), ['1', '2'])
})
