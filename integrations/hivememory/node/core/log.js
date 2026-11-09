/**
 * 连接器日志：写入状态目录，只记录标识与状态，不记录会话正文。
 *
 * 日志超过上限时轮转为一份旧文件。写不进去时静默放弃：日志不能影响回流，
 * 也不能打断 harness。
 */

import { appendFileSync, mkdirSync, renameSync, statSync } from 'node:fs'
import { join } from 'node:path'

const MAX_LOG_BYTES = 1024 * 1024

export class ConnectorLog {
  constructor(stateDir) {
    this.dir = stateDir
    this.path = join(stateDir, 'connector.log')
  }

  write(message) {
    try {
      mkdirSync(this.dir, { recursive: true })
      if (sizeOf(this.path) > MAX_LOG_BYTES) renameSync(this.path, `${this.path}.1`)
      appendFileSync(this.path, `${new Date().toISOString()} ${message}\n`, 'utf8')
    } catch {
      // 见模块说明：日志失败不向外传播。
    }
  }
}

function sizeOf(path) {
  try {
    return statSync(path).size
  } catch {
    return 0
  }
}
