import assert from 'node:assert/strict'
import { test } from 'node:test'

import { importBody, inFlightTurnInfo, outcomeOf, sessionTurns } from '../pi/turns.js'
import { PiSession } from './pi-session.js'

const FORGED_TURN = { source: 'forged', external_conversation_id: 'x', external_turn_id: 'x', user_message: 'x' }

/** 一轮完整的运行：开始时的叶子、用户消息（技能展开后的正文）、助手与工具结果、结束标记。 */
function runTurn(session, turn, { inputs = [`input ${turn}`], outcome = 'completed', body } = {}) {
  const from = session.leafId
  session.append({ type: 'message', message: { role: 'system', content: '', sections: {} } })
  session.user(`<skill name="x">expanded</skill> ${turn}`)
  body?.(session)
  session.assistant([{ type: 'text', text: `reply ${turn}` }])
  session.marker({ turn, from, outcome, inputs })
  return from
}

test('a turn maps text, thinking, tool calls with their results and the model', () => {
  const session = new PiSession()
  runTurn(session, 't1', {
    body: s => {
      s.assistant([
        { type: 'thinking', thinking: 'plan it', thinkingSignature: 'sig' },
        { type: 'thinking', thinking: '  ' },
        { type: 'text', text: 'calling' },
        { type: 'toolCall', id: 'c1', name: 'bash', arguments: { command: 'ls' } },
        { type: 'toolCall', id: 'c2', name: 'read', arguments: { path: 'a' } },
      ], { stopReason: 'toolUse', model: 'model-a' })
      s.toolResult('c1', 'files')
      s.toolResult('c2', 'missing', { isError: true })
      s.toolResult('unknown', 'skipped')
    },
  })

  const [turn] = sessionTurns(session.entries, session.sessionId)

  assert.deepEqual(turn.events, [
    { kind: 'thought', sequence: 1, content: 'plan it' },
    { kind: 'assistant_message', sequence: 2, content: 'calling' },
    { kind: 'tool_call', sequence: 3, content: '', action_id: 'c1', tool_name: 'bash', tool_args: { command: 'ls' }, status: 'success', result: 'files' },
    { kind: 'tool_call', sequence: 4, content: '', action_id: 'c2', tool_name: 'read', tool_args: { path: 'a' }, status: 'error', result: 'missing' },
    { kind: 'assistant_message', sequence: 5, content: 'reply t1' },
  ])
  assert.equal(turn.modelUsed, 'probe-model')
})

test('the user message is the raw input, not the expanded message in the record', () => {
  const session = new PiSession()
  runTurn(session, 't1', { inputs: ['/skill:hivememory look it up', 'and also this'] })

  const [turn] = sessionTurns(session.entries, session.sessionId)

  assert.equal(turn.userMessage, '/skill:hivememory look it up\n\nand also this')
})

test('the four memory tools become capability calls without the turn the model made up', () => {
  const session = new PiSession()
  runTurn(session, 't1', {
    body: s => {
      s.assistant([
        { type: 'toolCall', id: 'c1', name: 'memory_search', arguments: { query: 'q' } },
        { type: 'toolCall', id: 'c2', name: 'memory_read', arguments: { aliases: ['a1'] } },
        { type: 'toolCall', id: 'c3', name: 'memory_write', arguments: { content: 'note', turn: FORGED_TURN } },
        { type: 'toolCall', id: 'c4', name: 'memory_update', arguments: { alias: 'a2', instruction: 'fix', turn: FORGED_TURN } },
        { type: 'toolCall', id: 'c5', name: 'memory_read', arguments: { aliases: [] } },
      ], { stopReason: 'toolUse' })
    },
  })

  const [turn] = sessionTurns(session.entries, session.sessionId)
  const calls = turn.events.filter(event => event.kind === 'tool_call')

  assert.deepEqual(calls.map(({ verb, targets, tool_args, tool_name }) => ({ verb, targets, tool_args, tool_name })), [
    { verb: 'SEARCH', targets: undefined, tool_args: { query: 'q' }, tool_name: undefined },
    { verb: 'READ', targets: ['a1'], tool_args: undefined, tool_name: undefined },
    { verb: 'WRITE', targets: undefined, tool_args: { content: 'note' }, tool_name: undefined },
    { verb: 'UPDATE', targets: ['a2'], tool_args: { instruction: 'fix' }, tool_name: undefined },
    // 作用对象形状不对的调用按普通工具写出，服务端要求读取带作用对象。
    { verb: undefined, targets: undefined, tool_args: { aliases: [] }, tool_name: 'memory_read' },
  ])
})

test('a turn holds only the entries between its start and its marker, not other branches', () => {
  const session = new PiSession()
  runTurn(session, 't1')
  const branchPoint = session.entries.find(entry => entry.message?.content?.[0]?.text === 'reply t1').id
  runTurn(session, 't2')
  session.navigate(branchPoint)
  runTurn(session, 't3')

  const turns = sessionTurns(session.entries, session.sessionId)

  assert.deepEqual(turns.map(turn => turn.turnId), ['t1', 't2', 't3'])
  assert.deepEqual(turns[2].events.map(event => event.content), ['reply t3'])
})

test('markers copied by a fork or clone belong to the source session and are not imported', () => {
  const source = new PiSession('source-1')
  runTurn(source, 't1')
  const copy = source.clone('copy-2')
  runTurn(copy, 't2')

  assert.deepEqual(sessionTurns(copy.entries, 'copy-2').map(turn => turn.turnId), ['t2'])
})

test('the import carries identity, outcome and recent turns from the current branch', () => {
  const session = new PiSession()
  runTurn(session, 't1', { inputs: ['first'] })
  const branchPoint = session.leafId
  runTurn(session, 'abandoned', { inputs: ['abandoned'] })
  session.navigate(branchPoint)
  runTurn(session, 't2', { inputs: ['second'], outcome: 'failed' })

  const turns = sessionTurns(session.entries, session.sessionId)
  const body = importBody(session.entries, session.sessionId, turns.at(-1), 'pi')

  assert.equal(body.source, 'pi')
  assert.equal(body.external_conversation_id, 'session-1')
  assert.equal(body.external_turn_id, 't2')
  assert.equal(body.user_message, 'second')
  assert.equal(body.agent_id, 'pi')
  assert.equal(body.outcome, 'failed')
  assert.equal(body.model_used, 'probe-model')
  assert.deepEqual(body.recent_turns, [{ user_message: 'first', assistant_text: 'reply t1', external_turn_id: 't1' }])
})

test('the turn info at write time matches the fields the import later carries', () => {
  const session = new PiSession()
  runTurn(session, 't1', { inputs: ['first'] })
  const run = { turn: 't2', from: session.leafId, inputs: ['second'] }
  session.user('second')

  const atWrite = inFlightTurnInfo(session.entries, session.sessionId, run)
  session.assistant([{ type: 'text', text: 'done' }])
  session.marker({ turn: 't2', from: run.from, outcome: 'completed', inputs: ['second', 'steered later'] })
  const turn = sessionTurns(session.entries, session.sessionId).at(-1)
  const body = importBody(session.entries, session.sessionId, turn, 'pi')

  for (const field of ['source', 'external_conversation_id', 'external_turn_id', 'recent_turns']) {
    assert.deepEqual(atWrite[field], body[field], field)
  }
  assert.equal(atWrite.user_message, 'second')
})

test('outcomes come from the abort flag and the last stop reason', () => {
  assert.equal(outcomeOf(false, 'stop'), 'completed')
  assert.equal(outcomeOf(false, 'toolUse'), 'completed')
  assert.equal(outcomeOf(true, 'stop'), 'cancelled')
  assert.equal(outcomeOf(false, 'aborted'), 'cancelled')
  assert.equal(outcomeOf(false, 'error'), 'failed')
  assert.equal(outcomeOf(false, 'length'), 'failed')
  assert.equal(outcomeOf(false, undefined), 'failed')
})
