/**
 * 同步循环：提交已经写下结束标记、尚未确认的轮次，确认之后才记为完成。
 *
 * 规则：
 *
 * 1. **待提交的轮次**：一轮结束时先把它的轮次标识记入这个会话的待提交列表，再
 *    写结束标记；状态保存会话文件的路径与待提交列表，列表清空后移除。按待提交
 *    列表而不是会话文件中的全部标记提交：接入之前的轮次没有标记，自然不回填；
 *    状态目录丢失时也不会把已经确认过的轮次全部重提。待提交而文件中找不到标记
 *    的轮次（记下之后、写标记之前进程退出）直接移出列表。
 * 2. **触发**：一轮结束时同步这个会话；会话开始时同步全部还有待提交轮次的会话。
 *    同一会话同时只有一个同步在运行，运行期间再次触发时，结束后再同步一次。
 * 3. **读取**：从会话文件逐行读取条目，不经 pi 的会话管理（它会迁移旧格式的
 *    文件）；末尾不完整的行与无法解析的行跳过。pi 逐条追加写入，一轮结束时它的
 *    条目已经在文件中。
 * 4. **确认后才算完成**：可重试错误（网络错误、服务暂不可用）与配置错误停止处理
 *    这个会话、保留待提交，等下次触发；因这一轮内容被拒绝的 4xx 写入拒绝记录后
 *    移出；同一轮连续三次服务器内部错误也按拒绝处理，避免一轮阻塞整个会话。
 * 5. **会话文件被删除**：移除这个会话的状态，不再补发。
 *
 * 重发由服务端按推导出的交互标识去重：在服务端查重的保留窗口内（同一服务
 * 进程、最近接纳的若干交互），确认丢失后的重发不会形成重复交互。
 */

import { readFileSync } from 'node:fs'

import { importBody, sessionTurns } from './turns.js'

const MAX_SERVER_ERRORS = 3

/** 会话文件中的条目，按写入顺序；会话头、末尾不完整的行与无法解析的行跳过。 */
export function readSessionFile(path) {
  const text = readFileSync(path, 'utf8')
  const complete = text.slice(0, text.lastIndexOf('\n') + 1)
  const entries = []
  for (const line of complete.split('\n')) {
    if (!line.trim()) continue
    let entry
    try {
      entry = JSON.parse(line)
    } catch {
      continue
    }
    if (entry?.type !== 'session' && typeof entry?.id === 'string') entries.push(entry)
  }
  return entries
}

export class SessionSync {
  /**
   * `readEntries(path)` 读取一份会话文件的条目；`signal` 在扩展卸载时中止，进行中的
   * 提交随之取消，待提交列表不变。
   */
  constructor({ store, client, log, agentId, signal, readEntries = readSessionFile }) {
    this.store = store
    this.client = client
    this.log = log
    this.agentId = agentId
    this.signal = signal
    this.readEntries = readEntries
    this.running = new Map()
  }

  /** 记下一轮已经结束、待提交；在写结束标记之前调用。 */
  noteEnded(sessionId, file, turnId) {
    const state = this.store.load(sessionId) ?? { sessionId, file, pending: [], failures: {} }
    state.file = file
    if (!state.pending.includes(turnId)) state.pending.push(turnId)
    this.store.save(state)
  }

  /** 同步全部还有待提交轮次的会话。 */
  async recover() {
    await Promise.all(
      this.store.sessions()
        .filter(state => state.pending.length > 0)
        .map(state => this.request(state.sessionId)),
    )
  }

  /** 同步一个会话；返回的 promise 不会拒绝：同步的失败只进日志，不能影响 pi。 */
  request(sessionId) {
    const running = this.running.get(sessionId)
    if (running !== undefined) {
      running.again = true
      return running.done
    }
    const run = { again: false }
    this.running.set(sessionId, run)
    run.done = this.loop(sessionId, run)
    return run.done
  }

  /** 等待进行中的同步结束，最多等 `timeoutMs`；超时的留到下次触发。 */
  async drain(timeoutMs) {
    const pending = [...this.running.values()].map(run => run.done)
    if (pending.length === 0) return
    let timer
    const timeout = new Promise(resolve => {
      timer = setTimeout(resolve, timeoutMs)
    })
    await Promise.race([Promise.all(pending), timeout])
    clearTimeout(timer)
  }

  async loop(sessionId, run) {
    do {
      run.again = false
      try {
        await this.syncOnce(sessionId)
      } catch (error) {
        if (!this.signal.aborted) {
          this.log.write(`sync failed session=${sessionId} error=${error?.name}: ${error?.message}`)
        }
      }
    } while (run.again && !this.signal.aborted)
    // 与上面的检查之间没有 await：此后到达的触发会开始新的同步，不会落在已经退出的循环上。
    this.running.delete(sessionId)
  }

  async syncOnce(sessionId) {
    const state = this.store.load(sessionId)
    if (state === null || state.pending.length === 0) return
    let entries
    try {
      entries = this.readEntries(state.file)
    } catch (error) {
      if (error?.code !== 'ENOENT') throw error
      this.store.forget(sessionId)
      this.log.write(`forgot deleted session=${sessionId}`)
      return
    }
    const turns = sessionTurns(entries, sessionId)
    const marked = new Set(turns.map(turn => turn.turnId))
    const missing = state.pending.filter(turnId => !marked.has(turnId))
    if (missing.length > 0) {
      this.update(sessionId, current => {
        current.pending = current.pending.filter(turnId => marked.has(turnId))
      })
      this.log.write(`dropped unmarked session=${sessionId} turns=${missing.join(',')}`)
    }
    for (const turn of turns) {
      if (!state.pending.includes(turn.turnId)) continue
      this.signal.throwIfAborted()
      const result = await this.client.submit(importBody(entries, sessionId, turn, this.agentId), this.signal)
      if (!this.settle(sessionId, turn, result)) return
    }
  }

  /** 按提交结果处理待提交列表；返回是否继续提交这个会话的后续轮次。 */
  settle(sessionId, turn, result) {
    if (result.kind === 'confirmed') {
      this.log.write(`confirmed session=${sessionId} turn=${turn.turnId} status=${result.importStatus}`)
      this.complete(sessionId, turn)
      return true
    }
    if (result.kind === 'rejected') {
      this.reject(sessionId, turn, result)
      return true
    }
    if (result.kind === 'server_error') {
      const failures = this.update(sessionId, state => {
        state.failures[turn.turnId] = (state.failures[turn.turnId] ?? 0) + 1
      }).failures[turn.turnId]
      if (failures >= MAX_SERVER_ERRORS) {
        this.reject(sessionId, turn, result)
        return true
      }
    }
    this.log.write(
      `stopped session=${sessionId} turn=${turn.turnId} kind=${result.kind} `
      + `status=${result.status} reason=${result.reason ?? null}`,
    )
    return false
  }

  complete(sessionId, turn) {
    this.update(sessionId, state => {
      state.pending = state.pending.filter(turnId => turnId !== turn.turnId)
      delete state.failures[turn.turnId]
    })
  }

  reject(sessionId, turn, result) {
    this.store.appendRejection({
      at: Date.now(),
      session_id: sessionId,
      turn: turn.turnId,
      status: result.status,
      error: result.error,
      reason: result.reason,
    })
    this.log.write(`rejected session=${sessionId} turn=${turn.turnId} status=${result.status} error=${result.error}`)
    this.complete(sessionId, turn)
  }

  /** 读改写一个会话的状态；待提交列表清空时移除状态。 */
  update(sessionId, change) {
    const state = this.store.load(sessionId)
    change(state)
    if (state.pending.length === 0) this.store.forget(sessionId)
    else this.store.save(state)
    return state
  }
}
