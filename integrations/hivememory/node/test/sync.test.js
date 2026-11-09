import assert from 'node:assert/strict'
import { appendFileSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'

import { StateStore } from '../core/state.js'
import { readSessionFile, SessionSync } from '../pi/sync.js'
import { PiSession } from './pi-session.js'

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

function setup(results = []) {
  const stateDir = mkdtempSync(join(tmpdir(), 'pi-connector-'))
  const store = new StateStore(stateDir)
  const client = new FakeClient(results)
  const sync = new SessionSync({
    store,
    client,
    log: { write() {} },
    agentId: 'pi',
    signal: new AbortController().signal,
  })
  return { stateDir, store, client, sync, session: new PiSession() }
}

/** 一轮结束：先记入待提交，再写结束标记，与扩展的顺序相同。 */
function endTurn(sync, session, turn) {
  const from = session.leafId
  session.user(turn)
  session.assistant([{ type: 'text', text: `reply ${turn}` }])
  sync.noteEnded(session.sessionId, session.file, turn)
  session.marker({ turn, from, outcome: 'completed', inputs: [turn] })
}

const submittedTurns = client => client.submitted.map(body => body.external_turn_id)

test('ended turns are submitted in marker order and the state goes away once all are confirmed', async () => {
  const { store, client, sync, session } = setup()
  endTurn(sync, session, 't1')
  endTurn(sync, session, 't2')

  await sync.request(session.sessionId)

  assert.deepEqual(submittedTurns(client), ['t1', 't2'])
  assert.equal(store.load(session.sessionId), null)
})

test('only turns noted as ended are submitted, so earlier marked turns are not backfilled', async () => {
  const { client, sync, session } = setup()
  session.marker({ turn: 'before-install', from: null, outcome: 'completed', inputs: [] })
  endTurn(sync, session, 't1')

  await sync.request(session.sessionId)

  assert.deepEqual(submittedTurns(client), ['t1'])
})

test('a retryable failure keeps the turn pending and stops the session until the next trigger', async () => {
  const { store, client, sync, session } = setup([{ kind: 'retryable', status: null }])
  endTurn(sync, session, 't1')
  endTurn(sync, session, 't2')

  await sync.request(session.sessionId)
  assert.deepEqual(submittedTurns(client), ['t1'])
  assert.deepEqual(store.load(session.sessionId).pending, ['t1', 't2'])

  await sync.request(session.sessionId)
  assert.deepEqual(submittedTurns(client), ['t1', 't1', 't2'])
})

test('a misconfigured connector stops without dropping the turn', async () => {
  const { store, sync, session } = setup([{ kind: 'misconfigured', status: 401, error: 'auth.invalid' }])
  endTurn(sync, session, 't1')

  await sync.request(session.sessionId)

  assert.deepEqual(store.load(session.sessionId).pending, ['t1'])
})

test('a rejected turn is recorded and the next turn still goes out', async () => {
  const { stateDir, client, sync, session } = setup([{ kind: 'rejected', status: 422, error: 'request.invalid', reason: 'bad' }])
  endTurn(sync, session, 't1')
  endTurn(sync, session, 't2')

  await sync.request(session.sessionId)

  assert.deepEqual(submittedTurns(client), ['t1', 't2'])
  const [record] = readFileSync(join(stateDir, 'rejections.jsonl'), 'utf8').trim().split('\n').map(JSON.parse)
  assert.equal(record.turn, 't1')
  assert.equal(record.status, 422)
})

test('the third consecutive server error rejects the turn instead of blocking the session', async () => {
  const serverError = { kind: 'server_error', status: 500 }
  const { store, client, sync, session } = setup([serverError, serverError, serverError])
  endTurn(sync, session, 't1')

  await sync.request(session.sessionId)
  await sync.request(session.sessionId)
  assert.deepEqual(store.load(session.sessionId).pending, ['t1'])
  await sync.request(session.sessionId)

  assert.equal(client.submitted.length, 3)
  assert.equal(store.load(session.sessionId), null)
})

test('a pending turn whose marker never got written is dropped', async () => {
  const { store, client, sync, session } = setup()
  sync.noteEnded(session.sessionId, session.file, 'lost')

  await sync.request(session.sessionId)

  assert.deepEqual(client.submitted, [])
  assert.equal(store.load(session.sessionId), null)
})

test('recover syncs only sessions with pending turns', async () => {
  const { client, sync, session } = setup([{ kind: 'retryable', status: null }])
  endTurn(sync, session, 't1')
  await sync.request(session.sessionId)

  await sync.recover()

  assert.deepEqual(submittedTurns(client), ['t1', 't1'])
})

test('a deleted session file forgets its state', async () => {
  const { store, client, sync, session } = setup()
  endTurn(sync, session, 't1')
  rmSync(session.file)

  await sync.request(session.sessionId)

  assert.deepEqual(client.submitted, [])
  assert.equal(store.load(session.sessionId), null)
})

test('a turn that ends while a sync is submitting is picked up by another pass', async () => {
  const { client, sync, session } = setup()
  endTurn(sync, session, 't1')
  client.onSubmit = () => {
    client.onSubmit = null
    endTurn(sync, session, 't2')
    void sync.request(session.sessionId)
  }

  await sync.request(session.sessionId)

  assert.deepEqual(submittedTurns(client), ['t1', 't2'])
})

test('drain waits for a running sync but not beyond its limit', async () => {
  const { sync, session } = setup()
  endTurn(sync, session, 't1')
  let release
  sync.client.submit = () => new Promise(resolve => {
    release = () => resolve({ kind: 'confirmed', status: 200 })
  })
  void sync.request(session.sessionId)

  const started = Date.now()
  await sync.drain(50)
  assert.ok(Date.now() - started < 1000)
  release()
})

test('the session file reader skips the header, unparsable lines and an unfinished last line', () => {
  const { session } = setup()
  session.user('hello')
  appendFileSync(session.file, 'not json\n{"type":"message","id":"partial"')

  const entries = readSessionFile(session.file)

  assert.deepEqual(entries.map(entry => entry.id), session.entries.map(entry => entry.id))
})
