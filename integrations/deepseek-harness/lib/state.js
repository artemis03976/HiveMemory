/**
 * 插件状态：每个会话的水位与失败计数，以及首次运行时间与拒绝记录。
 *
 * 状态按会话分文件保存在状态目录中，以写临时文件再替换的方式原子写入。读写
 * 都是同步的文件操作：dsh 的插件代码在一个线程中运行，一次读改写之间不会穿插
 * 其他代码，同一进程内不需要文件锁。
 *
 * 状态与拒绝记录只保存标识、轮号与状态码，不保存会话正文。
 */

import { createHash } from 'node:crypto'
import { appendFileSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'

export class StateStore {
  constructor(stateDir) {
    this.dir = stateDir
    this.sessionsDir = join(stateDir, 'sessions')
  }

  /**
   * 读取一个会话的状态；尚未登记时返回 `null`。
   *
   * `lastTurn` 是水位：轮号不大于它的轮次都已得到确认、被不可重试地拒绝，或在
   * 登记时就已存在而不回填；登记时已有的会话在第一次读取之前为 `null`。
   * `endedTurn` 是在 dsh 中实际看到结束的最大轮号，大于水位时说明还有已结束
   * 的轮次没有得到确认；`failures` 记录各轮连续得到服务器内部错误的次数。
   */
  load(sessionId) {
    try {
      return JSON.parse(readFileSync(this.sessionPath(sessionId), 'utf8'))
    } catch (error) {
      if (error.code === 'ENOENT') return null
      throw error
    }
  }

  /** 首次见到一个会话时登记水位；已登记时不改动，返回是否新登记。 */
  register(sessionId, { origin, lastTurn }) {
    if (this.load(sessionId) !== null) return false
    this.save({ sessionId, origin, lastTurn, endedTurn: lastTurn ?? 0, failures: {} })
    return true
  }

  /** 读改写一个已登记会话的状态并写回。 */
  update(sessionId, change) {
    const state = this.load(sessionId)
    if (state === null) throw new Error(`session is not registered: ${sessionId}`)
    change(state)
    this.save(state)
    return state
  }

  /** 会话已被删除时移除它的状态，启动时的补发不再处理它。 */
  forget(sessionId) {
    rmSync(this.sessionPath(sessionId), { force: true })
  }

  /** 全部已登记会话的状态。 */
  sessions() {
    let names
    try {
      names = readdirSync(this.sessionsDir)
    } catch (error) {
      if (error.code === 'ENOENT') return []
      throw error
    }
    return names
      .filter(name => name.endsWith('.json'))
      .map(name => JSON.parse(readFileSync(join(this.sessionsDir, name), 'utf8')))
  }

  /** 插件首次运行的时间（Unix 毫秒）；早于它创建的会话不回填。 */
  firstRunAt(now = Date.now()) {
    const marker = join(this.dir, 'connector.json')
    try {
      return JSON.parse(readFileSync(marker, 'utf8')).first_run_at
    } catch (error) {
      if (error.code !== 'ENOENT') throw error
    }
    atomicWrite(marker, { first_run_at: now })
    return now
  }

  /** 追加一条拒绝记录（只含标识、状态码与原因，不含正文）。 */
  appendRejection(record) {
    mkdirSync(this.dir, { recursive: true })
    appendFileSync(join(this.dir, 'rejections.jsonl'), JSON.stringify(record) + '\n', 'utf8')
  }

  save(state) {
    atomicWrite(this.sessionPath(state.sessionId), state)
  }

  sessionPath(sessionId) {
    const digest = createHash('sha256').update(sessionId).digest('hex')
    return join(this.sessionsDir, `${digest.slice(0, 24)}.json`)
  }
}

/** 写临时文件后替换：读取方只会看到完整的旧版本或新版本。 */
function atomicWrite(path, data) {
  mkdirSync(dirname(path), { recursive: true })
  const temporary = `${path}.${process.pid}.tmp`
  writeFileSync(temporary, JSON.stringify(data), 'utf8')
  renameSync(temporary, path)
}
