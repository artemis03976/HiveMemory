/**
 * 同步循环：提交水位之后已经结束的轮次，按顺序提交，确认之后水位才前进。
 *
 * 规则：
 *
 * 1. **触发**：dsh 中一轮结束（`turn/end`）与一个 Agent 创建或恢复时，同步该轮
 *    所在的会话；插件启动时，同步全部还有已结束但未确认轮次的会话。同一会话
 *    同时只有一个同步在运行，运行期间再次触发时，结束后重新同步一次。
 * 2. **结束的判定**：只提交有 `turn/end` 的轮次，仍在进行的一轮及其后的轮次
 *    不提交。会话在本进程中存活时读到的是实时记录，崩溃时未结束的一轮由 dsh
 *    在恢复会话时补上 `interrupted` 收尾后提交。会话不在本进程中时读到的是
 *    持久记录，读取方会为最后一轮未结束的记录补上同样的收尾，而该会话可能
 *    正在另一个 dsh 进程中运行这一轮，因此只提交实际看到结束的轮次。
 * 3. **先落盘再读取**：会话在本进程中存活时，读取之前先让 dsh 把缓冲的事件写入
 *    持久记录：确认之后 dsh 崩溃，已确认的轮次不会从会话记录中消失、再被新的
 *    一轮以同一轮号取代。
 * 4. **水位只在确认后前进**：可重试错误（网络错误、服务暂不可用）与插件配置错误
 *    停止处理这个会话并保留水位，等下次触发；因这一轮内容被拒绝的 4xx 写入
 *    拒绝记录后前进；同一轮连续三次服务器内部错误也按不可重试处理，避免一轮
 *    阻塞整个会话。
 * 5. **首次见到的会话**：在插件首次运行之后创建的从头提交；更早创建的（接入之前
 *    已有、后来被恢复的会话）不回填，水位在第一次读取时放在当时最后一个已经
 *    结束的轮次，之后开始的轮次照常提交。
 * 6. **会话被删除**：移除它的状态，不再补发。
 *
 * 重发由服务端按推导出的交互标识去重：在服务端查重的保留窗口内（同一服务
 * 进程、最近接纳的若干交互），确认丢失后的重发不会形成重复交互。
 */

import { importBody, sessionTurns } from './turns.js'

const MAX_SERVER_ERRORS = 3

// dsh 的会话查询服务在会话不存在（已被删除）时给出的错误码。
const SESSION_NOT_FOUND = 'SESSION_QUERY_SESSION_NOT_FOUND'

export class SessionSync {
  /**
   * `sessions` 是对 dsh 会话服务的最小访问：`isLive(id)` 会话是否在本进程中
   * 存活，`flush(id)` 把存活会话缓冲的事件写入持久记录，`read(id)` 读取会话
   * 的完整逻辑记录（存活时读实时记录）。`signal` 在插件卸载时中止，进行中的
   * 提交随之取消，水位不前进。
   */
  constructor({ store, client, log, sessions, agentId, signal, now = Date.now() }) {
    this.store = store
    this.client = client
    this.log = log
    this.sessions = sessions
    this.agentId = agentId
    this.signal = signal
    this.firstRunAt = store.firstRunAt(now)
    this.running = new Map()
  }

  /**
   * 首次见到会话时登记；已登记时不改动。
   *
   * 会话的状态为：来源（`new` 或 `existing`）；水位 `lastTurn`，轮号不大于它的
   * 轮次都已得到确认、被不可重试地拒绝，或在登记时就已存在而不回填；实际看到
   * 结束的最大轮号 `endedTurn`，大于水位时说明还有已结束的轮次没有得到确认；
   * 各轮连续得到服务器内部错误的次数 `failures`。插件首次运行之后创建的会话
   * 从头提交；更早创建的不回填，水位留空，到第一次读取时再确定。
   */
  register(sessionId, { createdAt }) {
    if (this.store.load(sessionId) !== null) return
    const origin = createdAt >= this.firstRunAt ? 'new' : 'existing'
    const lastTurn = origin === 'new' ? 0 : null
    this.store.save({ sessionId, origin, lastTurn, endedTurn: lastTurn ?? 0, failures: {} })
    this.log.write(`registered session=${sessionId} origin=${origin}`)
  }

  /** 记下 dsh 中实际结束的一轮；会话必须已经登记。 */
  noteEnded(sessionId, turn) {
    this.store.update(sessionId, state => {
      state.endedTurn = Math.max(state.endedTurn, turn)
    })
  }

  /** 同步全部还有已结束但未确认轮次的会话。 */
  async recover() {
    await Promise.all(
      this.store.sessions()
        .filter(state => state.lastTurn === null || state.endedTurn > state.lastTurn)
        .map(state => this.request(state.sessionId)),
    )
  }

  /**
   * 同步一个会话；返回的 promise 不会拒绝：同步的失败只进日志，不能影响 dsh。
   */
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
    let state = this.store.load(sessionId)
    if (state === null) return
    const live = this.sessions.isLive(sessionId)
    let snapshot
    try {
      if (live) await this.sessions.flush(sessionId)
      snapshot = await this.sessions.read(sessionId)
    } catch (error) {
      if (error?.code !== SESSION_NOT_FOUND) throw error
      this.store.forget(sessionId)
      this.log.write(`forgot deleted session=${sessionId}`)
      return
    }
    const turns = sessionTurns(snapshot)
    if (state.lastTurn === null) {
      const existing = turns.filter(turn => turn.closed).at(-1)?.number ?? 0
      state = this.store.update(sessionId, current => {
        current.lastTurn = existing
      })
      this.log.write(`watermark session=${sessionId} last_turn=${existing}`)
    }
    const lastEnded = live ? Infinity : state.endedTurn
    for (const [index, turn] of turns.entries()) {
      if (turn.number <= state.lastTurn) continue
      if (!turn.closed || turn.number > lastEnded) return
      this.signal.throwIfAborted()
      const result = await this.client.submit(importBody(sessionId, turns, index, this.agentId), this.signal)
      state = this.settle(sessionId, turn, result)
      if (state === null) return
    }
  }

  /** 按提交结果处理水位；返回更新后的状态，停止处理这个会话时返回 `null`。 */
  settle(sessionId, turn, result) {
    if (result.kind === 'confirmed') {
      this.log.write(`confirmed session=${sessionId} turn=${turn.number} status=${result.importStatus}`)
      return this.advance(sessionId, turn)
    }
    if (result.kind === 'rejected') return this.reject(sessionId, turn, result)
    if (result.kind === 'server_error') {
      const state = this.store.update(sessionId, current => {
        current.failures[turn.number] = (current.failures[turn.number] ?? 0) + 1
      })
      if (state.failures[turn.number] >= MAX_SERVER_ERRORS) return this.reject(sessionId, turn, result)
    }
    this.log.write(
      `stopped session=${sessionId} turn=${turn.number} kind=${result.kind} `
      + `status=${result.status} reason=${result.reason ?? null}`,
    )
    return null
  }

  advance(sessionId, turn) {
    return this.store.update(sessionId, state => {
      state.lastTurn = turn.number
      delete state.failures[turn.number]
    })
  }

  reject(sessionId, turn, result) {
    this.store.appendRejection({
      at: Date.now(),
      session_id: sessionId,
      turn: turn.number,
      status: result.status,
      error: result.error,
      reason: result.reason,
    })
    this.log.write(`rejected session=${sessionId} turn=${turn.number} status=${result.status} error=${result.error}`)
    return this.advance(sessionId, turn)
  }
}
