/**
 * 连接器状态：按会话分文件保存，以及拒绝记录。
 *
 * 状态以写临时文件再替换的方式原子写入。读写都是同步的文件操作：harness 的
 * 扩展代码在一个线程中运行，一次读改写之间不会穿插其他代码，同一进程内不需要
 * 文件锁。状态与拒绝记录只保存标识、路径与状态码，不保存会话正文。
 */

import { createHash } from 'node:crypto'
import { appendFileSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'

export class StateStore {
  constructor(stateDir) {
    this.dir = stateDir
    this.sessionsDir = join(stateDir, 'sessions')
  }

  /** 读取一个会话的状态；没有保存时返回 `null`。 */
  load(sessionId) {
    try {
      return JSON.parse(readFileSync(this.sessionPath(sessionId), 'utf8'))
    } catch (error) {
      if (error.code === 'ENOENT') return null
      throw error
    }
  }

  /** 保存一个会话的状态；状态中的 `sessionId` 决定文件名。 */
  save(state) {
    atomicWrite(this.sessionPath(state.sessionId), state)
  }

  /** 读改写一个已保存会话的状态并写回；没有保存时抛出错误。 */
  update(sessionId, change) {
    const state = this.load(sessionId)
    if (state === null) throw new Error(`session is not registered: ${sessionId}`)
    change(state)
    this.save(state)
    return state
  }

  /** 连接器首次运行的时间（Unix 毫秒）：第一次调用时记下，之后读回同一个值。 */
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

  /** 移除一个会话的状态。 */
  forget(sessionId) {
    rmSync(this.sessionPath(sessionId), { force: true })
  }

  /** 全部已保存会话的状态。 */
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

  /** 追加一条拒绝记录（只含标识、状态码与原因，不含正文）。 */
  appendRejection(record) {
    mkdirSync(this.dir, { recursive: true })
    appendFileSync(join(this.dir, 'rejections.jsonl'), JSON.stringify(record) + '\n', 'utf8')
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
