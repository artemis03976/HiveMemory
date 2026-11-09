/**
 * pi 会话记录的测试替身：按 pi 的方式追加条目（挂在当前叶子之下），可以回到较早
 * 的条目继续，并把全部条目写入一份会话文件。
 */

import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { MARKER_TYPE } from '../pi/turns.js'

export class PiSession {
  constructor(sessionId = 'session-1', { entries = [], dir } = {}) {
    this.sessionId = sessionId
    this.entries = entries.map(entry => ({ ...entry }))
    this.leafId = this.entries.at(-1)?.id ?? null
    this.nextId = 1
    this.file = join(dir ?? mkdtempSync(join(tmpdir(), 'pi-session-')), `${sessionId}.jsonl`)
    if (this.entries.length === 0) this.append({ type: 'model_change', provider: 'mock', modelId: 'probe-model' })
    this.flush()
  }

  /** 挂在当前叶子之下追加一条条目，返回它的标识。 */
  append(fields) {
    const id = `${this.sessionId.slice(-2)}e${this.nextId++}`
    this.entries.push({ id, parentId: this.leafId, timestamp: '2026-11-08T00:00:00.000Z', ...fields })
    this.leafId = id
    this.flush()
    return id
  }

  user(text) {
    return this.append({ type: 'message', message: { role: 'user', content: [{ type: 'text', text }] } })
  }

  assistant(content, { stopReason = 'stop', model = 'probe-model' } = {}) {
    return this.append({ type: 'message', message: { role: 'assistant', content, stopReason, model, provider: 'mock' } })
  }

  toolResult(toolCallId, text, { isError = false } = {}) {
    return this.append({
      type: 'message',
      message: { role: 'toolResult', toolCallId, toolName: 'x', content: [{ type: 'text', text }], isError },
    })
  }

  /** 写一条结束标记，模拟扩展在一轮结束时的写入。 */
  marker(data) {
    return this.append({ type: 'custom', customType: MARKER_TYPE, data: { session: this.sessionId, ...data } })
  }

  /** 回到较早的条目，之后的条目从那里分出新的分支。 */
  navigate(entryId) {
    this.leafId = entryId
  }

  /** 以当前路径上的条目（连同标识）新建一个会话，与 pi 的 clone 相同。 */
  clone(sessionId) {
    const byId = new Map(this.entries.map(entry => [entry.id, entry]))
    const path = []
    for (let id = this.leafId; id; id = byId.get(id).parentId) path.unshift(byId.get(id))
    return new PiSession(sessionId, { entries: path })
  }

  flush() {
    const header = { type: 'session', version: 3, id: this.sessionId, timestamp: '2026-11-08T00:00:00.000Z', cwd: 'C:\\work' }
    writeFileSync(this.file, [header, ...this.entries].map(entry => JSON.stringify(entry) + '\n').join(''), 'utf8')
  }

  /** pi 交给扩展的 `ctx.sessionManager` 中用到的部分。 */
  get sessionManager() {
    return {
      getSessionId: () => this.sessionId,
      getSessionFile: () => this.file,
      getEntries: () => this.entries,
      getLeafId: () => this.leafId,
    }
  }
}
