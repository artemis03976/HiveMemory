/** 测试辅助：按 dsh 会话事件的形状构造 `sessionQuery.readSession` 的读取结果。 */

export class SessionLog {
  /** `parentSession` 给出时是 dsh 为子 Agent 创建的会话，会话头带来源 `subagent`。 */
  constructor({ id = 'session-1', inheritedEventCount = 0, parentSession } = {}) {
    this.id = id
    this.inheritedEventCount = inheritedEventCount
    this.header = parentSession === undefined ? { id } : { id, parentSession, origin: 'subagent' }
    this.events = []
  }

  add(type, data) {
    this.events.push({ type, seq: this.events.length, time: 0, data })
    return this
  }

  startTurn(turn) {
    return this.add('turn/start', { turn })
  }

  endTurn(turn, reason = { kind: 'completed' }) {
    return this.add('turn/end', { turn, reason })
  }

  user(text, source = { kind: 'user' }) {
    return this.add('user/message', { role: 'user', content: [{ type: 'text', text }], source })
  }

  assistant(content, model = 'deepseek-flash') {
    return this.add('assistant/message', {
      turn: 0,
      step: 0,
      message: { role: 'assistant', content, source: { kind: 'model', provider: 'deepseek', model } },
      stream: [],
    })
  }

  toolCall(callId, name, args) {
    const raw = typeof args === 'string' ? args : JSON.stringify(args)
    return this.add('tool/call', { turn: 0, step: 0, callId, name, arguments: raw })
  }

  toolResult(callId, text, isError = false) {
    return this.add('tool/result', {
      turn: 0,
      step: 0,
      message: {
        role: 'tool',
        toolCallId: callId,
        content: [{ type: 'text', text }],
        source: { kind: 'tool', callId },
        ...(isError ? { isError: true } : {}),
      },
    })
  }

  /** 一轮完整的对话：用户消息、一段回复，以 `completed` 结束。 */
  turn(number, prompt, reply = `reply ${number}`) {
    return this.startTurn(number).user(prompt).assistant([{ type: 'text', text: reply }]).endTurn(number)
  }

  snapshot() {
    return {
      session: structuredClone(this.header),
      inheritedEventCount: this.inheritedEventCount,
      events: structuredClone(this.events),
    }
  }
}
