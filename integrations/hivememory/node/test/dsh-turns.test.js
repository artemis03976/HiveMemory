import assert from 'node:assert/strict'
import { test } from 'node:test'

import { importBody, sessionTurns, turnInfo } from '../dsh/turns.js'
import { SessionLog } from './dsh-session-log.js'

function body(log, index = 0) {
  return importBody(log.id, sessionTurns(log.snapshot()), index, 'dsh')
}

test('a turn maps reasoning, text, tool calls with their results and the model', () => {
  const log = new SessionLog()
    .startTurn(1)
    .user('list files')
    .assistant([
      { type: 'reasoning', text: 'look first' },
      { type: 'tool-call', id: 'c1', name: 'bash', arguments: '{"command":"ls"}' },
    ], 'deepseek-pro')
    .toolCall('c1', 'bash', { command: 'ls' })
    .toolResult('c1', 'a.txt')
    .assistant([{ type: 'text', text: 'one file' }], 'deepseek-flash')
    .endTurn(1)

  const imported = body(log)

  assert.equal(imported.user_message, 'list files')
  assert.equal(imported.model_used, 'deepseek-flash')
  assert.deepEqual(imported.events, [
    { kind: 'thought', sequence: 1, content: 'look first' },
    {
      kind: 'tool_call',
      sequence: 2,
      content: '',
      action_id: 'c1',
      tool_name: 'bash',
      tool_args: { command: 'ls' },
      status: 'success',
      result: 'a.txt',
    },
    { kind: 'assistant_message', sequence: 3, content: 'one file' },
  ])
})

test('failed results become error outcomes and calls without a result have no outcome', () => {
  const log = new SessionLog()
    .startTurn(1)
    .user('run')
    .toolCall('c1', 'bash', { command: 'false' })
    .toolResult('c1', 'exit 1', true)
    .toolCall('c2', 'bash', { command: 'sleep 9' })
    .toolResult('missing', 'stray result')
    .endTurn(1, { kind: 'aborted', reason: { kind: 'user' } })

  const [failed, unanswered] = body(log).events

  assert.deepEqual([failed.status, failed.result], ['error', 'exit 1'])
  assert.equal('status' in unanswered, false)
  assert.equal('result' in unanswered, false)
})

test('the four memory tools become capability calls without the turn parameter', () => {
  const log = new SessionLog()
    .startTurn(1)
    .user('remember this')
    .toolCall('s', 'memory_search', { query: 'dsh' })
    .toolCall('r', 'memory_read', { aliases: ['mem_a', 'mem_b'] })
    .toolCall('w', 'memory_write', { content: 'fact', title: 't', turn: { forged: true } })
    .toolCall('u', 'memory_update', { alias: 'mem_a', instruction: 'fix', turn: { forged: true } })
    .endTurn(1)

  const calls = body(log).events.map(({ kind, sequence, content, ...call }) => call)

  assert.deepEqual(calls, [
    { action_id: 's', verb: 'SEARCH', tool_args: { query: 'dsh' } },
    { action_id: 'r', verb: 'READ', targets: ['mem_a', 'mem_b'] },
    { action_id: 'w', verb: 'WRITE', tool_args: { content: 'fact', title: 't' } },
    { action_id: 'u', verb: 'UPDATE', targets: ['mem_a'], tool_args: { instruction: 'fix' } },
  ])
})

test('memory calls with malformed targets are written as ordinary tools', () => {
  const log = new SessionLog()
    .startTurn(1)
    .user('read')
    .toolCall('r', 'memory_read', { aliases: [] })
    .toolCall('u', 'memory_update', { alias: '', instruction: 'fix' })
    .endTurn(1)

  const calls = body(log).events.map(event => [event.tool_name, event.verb])

  assert.deepEqual(calls, [['memory_read', undefined], ['memory_update', undefined]])
})

test('arguments that are not a JSON object are kept under input', () => {
  const log = new SessionLog()
    .startTurn(1)
    .user('call')
    .toolCall('a', 'tool', '"bare"')
    .toolCall('b', 'tool', '{not json')
    .endTurn(1)

  const args = body(log).events.map(event => event.tool_args)

  assert.deepEqual(args, [{ input: 'bare' }, { input: '{not json' }])
})

test('the user message joins human messages in order and skips injected context', () => {
  const log = new SessionLog()
    .startTurn(1)
    .user('first ask')
    .user('skill body', { kind: 'skill' })
    .add('user/message', {
      role: 'user',
      content: [{ type: 'text', text: 'and this' }, { type: 'image', attachment: {} }],
      source: { kind: 'user' },
    })
    .endTurn(1)

  assert.equal(body(log).user_message, 'first ask\n\nand this\n[image]')
})

test('subagent settlement notices become subagent events in place, not user text', () => {
  const log = new SessionLog()
    .startTurn(1)
    .user('research this repo')
    .toolCall('c1', 'subagent', { prompt: 'dig into storage' })
    .toolResult('c1', 'started subagent child-1')
    .user('Agent child-1 sent a message: report', { kind: 'agent-message', form: 'relay', senderSessionId: 'child-1' })
    .add('user/message', {
      role: 'user',
      content: [
        { type: 'text', text: 'Background subagent child-1 finished.' },
        { type: 'text', text: 'Its closing message:' },
        { type: 'text', text: 'report' },
      ],
      source: { kind: 'subagent-settled', form: 'notice', senderSessionId: 'child-1' },
    })
    .user('background job done', { kind: 'tool-jobs', form: 'notice' })
    .assistant([{ type: 'text', text: 'summary' }])
    .endTurn(1)

  const imported = body(log)

  assert.equal(imported.user_message, 'research this repo')
  assert.deepEqual(imported.events.map(event => [event.kind, event.content]), [
    ['tool_call', ''],
    ['subagent_message', 'Background subagent child-1 finished.\nIts closing message:\nreport'],
    ['assistant_message', 'summary'],
  ])
})

test('a turn woken by a subagent settlement notice has an empty user message', () => {
  const log = new SessionLog()
    .startTurn(1)
    .user('Background subagent child-1 finished.', { kind: 'subagent-settled', form: 'notice' })
    .assistant([{ type: 'text', text: 'noted' }])
    .endTurn(1)

  const imported = body(log)

  assert.equal(imported.user_message, '')
  assert.deepEqual(imported.events[0], {
    kind: 'subagent_message',
    sequence: 1,
    content: 'Background subagent child-1 finished.',
  })
})

test('turn end reasons map to completed, cancelled or failed', () => {
  const reasons = [
    [{ kind: 'completed' }, 'completed'],
    [{ kind: 'aborted', reason: { kind: 'user' } }, 'cancelled'],
    [{ kind: 'interrupted' }, 'cancelled'],
    [{ kind: 'error', error: { message: 'boom', code: 'UNKNOWN' } }, 'failed'],
    [{ kind: 'blocked' }, 'failed'],
    [{ kind: 'max-tokens' }, 'failed'],
    [{ kind: 'plugin-defined' }, 'failed'],
  ]
  for (const [reason, outcome] of reasons) {
    const log = new SessionLog().startTurn(1).user('x').endTurn(1, reason)
    assert.equal(body(log).outcome, outcome, reason.kind)
  }
})

test('an unfinished last turn is reported open', () => {
  const log = new SessionLog().turn(1, 'done').startTurn(2).user('still running')

  const turns = sessionTurns(log.snapshot())

  assert.deepEqual(turns.map(turn => [turn.number, turn.closed]), [[1, true], [2, false]])
})

test('a fork child reads only its own turns', () => {
  const parent = new SessionLog().turn(1, 'parent ask').startTurn(2).user('cut here')
  const child = new SessionLog({ id: 'child', inheritedEventCount: parent.events.length })
  child.events = structuredClone(parent.events)
  child.add('session/end-seed', { inherited: true }).endTurn(2, { kind: 'forked' }).turn(3, 'child ask')

  const turns = sessionTurns(child.snapshot())

  assert.deepEqual(turns.map(turn => [turn.number, turn.userMessage]), [[3, 'child ask']])
})

test('identity and recent turns come from the session and the closed turns before it', () => {
  const log = new SessionLog({ id: 'sess-9' })
    .turn(1, 'one').turn(2, 'two').turn(3, 'three').turn(4, 'four')
    .startTurn(5).user('five')

  const info = turnInfo(log.id, sessionTurns(log.snapshot()), 4)

  assert.deepEqual(info, {
    source: 'dsh',
    external_conversation_id: 'sess-9',
    external_turn_id: '5',
    user_message: 'five',
    recent_turns: [
      { user_message: 'two', assistant_text: 'reply 2', external_turn_id: '2' },
      { user_message: 'three', assistant_text: 'reply 3', external_turn_id: '3' },
      { user_message: 'four', assistant_text: 'reply 4', external_turn_id: '4' },
    ],
  })
})

test('recent turn texts are truncated by characters without splitting surrogate pairs', () => {
  const long = '😀'.repeat(2001)
  const log = new SessionLog().turn(1, long).startTurn(2).user('next')

  const [recent] = turnInfo(log.id, sessionTurns(log.snapshot()), 1).recent_turns

  assert.equal(Array.from(recent.user_message).length, 2000)
  assert.equal(recent.user_message, '😀'.repeat(2000))
})
