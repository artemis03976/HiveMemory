/**
 * 本机服务：Qdrant 与 HiveMemory 后端的启动、就绪等待与正常关闭。
 *
 * 程序与数据的位置由 layout.js 给出；后端以数据根为工作目录，日志写在数据根的 .hivememory/logs。
 * 端口与仓库配置一致，外部 harness 的插件都连 8769，因此被占用时不换端口。
 */

import { spawn } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import { closeSync, existsSync, mkdirSync, openSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

export const BACKEND_ORIGIN = 'http://127.0.0.1:8769'
export const MANAGEMENT_CREDENTIAL_COOKIE = 'hivememory_management'

const BACKEND_HEALTH_URL = `${BACKEND_ORIGIN}/health`
const BACKEND_READY_URL = `${BACKEND_ORIGIN}/health/ready`
const QDRANT_READY_URL = 'http://127.0.0.1:6333/readyz'
const BACKEND_START_TIMEOUT_MS = 120_000
const QDRANT_START_TIMEOUT_MS = 60_000
// 模型未就绪、模型缓存也这么久没有增长时，不再等待：下载失败或加载失败不应让窗口停在启动页。
const MODEL_STALL_TIMEOUT_MS = 60_000

export class LocalServices {
  constructor({ layout, log }) {
    this.layout = layout
    this.logDir = join(layout.dataRoot, '.hivememory', 'logs')
    this.runDir = join(layout.dataRoot, '.hivememory', 'run')
    // 配置中模型缓存的默认位置，相对后端的工作目录（数据根）。
    this.modelCacheDir = join(layout.dataRoot, 'data', 'model_cache')
    this.log = log
    this.backend = null
    this.backendExited = null
    this.qdrant = null
    this.qdrantExited = null
    this.stopping = false
    this.unexpectedExitHandler = () => {}
  }

  /**
   * 启动 Qdrant 与后端，等后端能响应 HTTP 后返回。
   *
   * 8769 上已有后端时直接附着，不启动也不关闭它（只在原型中允许，方便与开发脚本并用）；
   * 这时没有内存中的凭据，页面由那个后端照旧下发 cookie。
   * 返回 `{ attached, credential }`，credential 是交给后端的管理凭据。
   */
  async start() {
    if (await isUp(BACKEND_HEALTH_URL)) {
      this.log('8769 上已有后端，附着到它，不启动也不关闭它')
      return { attached: true, credential: null }
    }
    await this.ensureQdrant()
    const credential = randomBytes(32).toString('base64url')
    await this.startBackend(credential)
    return { attached: false, credential }
  }

  /**
   * 等推理模型就绪。首次启动时后端在预热中下载模型，期间以已下载的字节数回调 onDownload。
   *
   * 返回 true 表示就绪；模型缓存停止增长超过时限仍未就绪时返回 false，由调用方先打开界面。
   */
  async waitForModels(onDownload) {
    const initialBytes = directorySize(this.modelCacheDir)
    let lastBytes = initialBytes
    let lastGrowth = Date.now()
    while (!(await isUp(BACKEND_READY_URL))) {
      const bytes = directorySize(this.modelCacheDir)
      if (bytes !== lastBytes) {
        lastBytes = bytes
        lastGrowth = Date.now()
      }
      if (bytes > initialBytes) onDownload(bytes)
      if (Date.now() - lastGrowth > MODEL_STALL_TIMEOUT_MS) return false
      await delay(1000)
    }
    if (lastBytes > initialBytes) this.log(`推理模型已下载: bytes=${lastBytes - initialBytes}`)
    return true
  }

  /** 后端在运行中（不是关闭时）退出时调用 handler(exitCode)。 */
  onUnexpectedExit(handler) {
    this.unexpectedExitHandler = handler
  }

  /**
   * 关闭自己启动的服务：先请后端正常关闭并等它退出，再结束 Qdrant。
   *
   * 后端退出前要结算内存中的话题，其中有 LLM 调用，所以不设时限、不强制结束。
   * Qdrant 先写 WAL 再确认写入，直接结束后下次启动能够恢复。
   */
  async stop() {
    this.stopping = true
    if (this.backend) {
      const started = Date.now()
      this.log('请后端正常关闭：关闭它的 stdin')
      this.backend.stdin.end()
      const code = await this.backendExited
      this.log(`后端已退出: code=${code} elapsed_s=${((Date.now() - started) / 1000).toFixed(1)}`)
    }
    if (this.qdrant) {
      this.qdrant.kill()
      await this.qdrantExited
      rmSync(join(this.runDir, 'qdrant.pid'), { force: true })
      this.log('Qdrant 已结束')
    }
  }

  async ensureQdrant() {
    if (await isUp(QDRANT_READY_URL)) {
      this.log('Qdrant 已在运行，直接使用，退出时不结束它')
      return
    }
    const { qdrantBin, qdrantStorage } = this.layout
    if (!qdrantBin || !existsSync(qdrantBin)) {
      throw new Error(`Qdrant 可执行文件不存在：'${qdrantBin ?? ''}'`)
    }
    mkdirSync(this.runDir, { recursive: true })
    mkdirSync(this.logDir, { recursive: true })
    mkdirSync(qdrantStorage, { recursive: true })

    this.log(`启动 Qdrant: storage=${qdrantStorage}`)
    const stdout = openSync(join(this.logDir, 'qdrant.log'), 'a')
    const stderr = openSync(join(this.logDir, 'qdrant.err.log'), 'a')
    // 工作目录是 run 目录：Qdrant 在工作目录下写快照与初始化标记。
    this.qdrant = spawn(qdrantBin, [], {
      cwd: this.runDir,
      env: {
        ...process.env,
        QDRANT__STORAGE__STORAGE_PATH: qdrantStorage,
        QDRANT__SERVICE__HOST: '127.0.0.1',
        QDRANT__SERVICE__HTTP_PORT: '6333',
        QDRANT__SERVICE__GRPC_PORT: '6334',
      },
      stdio: ['ignore', stdout, stderr],
      windowsHide: true,
      // 不放进主进程的 job（见 startBackend）：主进程意外退出后，后端结算话题时还要写 Qdrant。
      detached: true,
    })
    closeSync(stdout)
    closeSync(stderr)
    // 与开发脚本共用 pid 文件：主进程意外退出后，可以用 hivememory-dev.ps1 stop-qdrant 结束它。
    writeFileSync(join(this.runDir, 'qdrant.pid'), String(this.qdrant.pid))

    this.qdrantExited = new Promise((resolveExit) => this.qdrant.once('exit', resolveExit))
    if (!(await waitUntilUp(QDRANT_READY_URL, QDRANT_START_TIMEOUT_MS, this.qdrantExited))) {
      throw new Error('Qdrant 没有启动成功，见日志目录下的 qdrant.log 与 qdrant.err.log')
    }
    this.log('Qdrant 已就绪')
  }

  async startBackend(credential) {
    const { python, pythonEnv, frontendIndex, dataRoot } = this.layout
    for (const required of [python, frontendIndex]) {
      if (!existsSync(required)) throw new Error(`找不到 ${required}`)
    }
    mkdirSync(this.logDir, { recursive: true })

    this.log('启动后端（桌面模式）')
    // 后端的日志配置生效之前的输出（导入错误、uvicorn 的启动提示）写到这里，与开发脚本同名。
    const output = openSync(join(this.logDir, 'backend.stdout.log'), 'a')
    // 用不带控制台的 pythonw.exe。venv 的 python.exe 是转发器，真正的解释器是它的子进程；
    // detached 的转发器没有控制台，Windows 会给控制台程序的解释器新开一个空的控制台窗口，
    // 用户关掉这个窗口时后端立即被结束，来不及结算话题。
    this.backend = spawn(python, ['-m', 'hivememory.server', '--desktop'], {
      // 后端按工作目录解析 .env 与相对路径的数据目录（模型缓存、归档、Artifact）。
      cwd: dataRoot,
      env: {
        ...process.env,
        ...pythonEnv,
        HIVEMEMORY_SERVE_FRONTEND: 'true',
        HIVEMEMORY__LOGGING__FILE_PATH: join(this.logDir, 'backend.log'),
        HIVEMEMORY__LOGGING__EVENT_FILE_PATH: join(this.logDir, 'runtime-events.jsonl'),
        // 输出重定向到文件时，Python 默认按系统代码页编码，中文日志会编码失败。
        PYTHONIOENCODING: 'utf-8',
      },
      stdio: ['pipe', output, output],
      windowsHide: true,
      // 非 detached 的子进程会被放进一个随主进程结束而结束全部成员的 job：主进程意外退出时，
      // 后端会在结算话题的中途被结束。detached 后后端读到 stdin 结束，自己正常关闭再退出。
      detached: true,
    })
    closeSync(output)
    this.backendExited = new Promise((resolveExit) => this.backend.once('exit', resolveExit))
    this.backendExited.then((code) => {
      if (!this.stopping) this.unexpectedExitHandler(code)
    })
    // 后端提前退出时写 stdin 会出错；退出本身由上面的 exit 处理，这里只记录。
    this.backend.stdin.on('error', (error) => this.log(`写后端 stdin 失败: ${error.code}`))
    // stdin 的第一行是管理凭据；stdin 保持打开，关闭它即请求后端正常关闭。
    this.backend.stdin.write(`${credential}\n`)

    if (!(await waitUntilUp(BACKEND_HEALTH_URL, BACKEND_START_TIMEOUT_MS, this.backendExited))) {
      throw new Error('后端没有启动成功，见日志目录下的 backend.stdout.log 与 backend.log')
    }
    this.log('后端已能响应 HTTP')
  }
}

function directorySize(path) {
  if (!existsSync(path)) return 0
  let total = 0
  for (const entry of readdirSync(path, { withFileTypes: true, recursive: true })) {
    // 下载中的文件可能在两次读取之间被改名或删除。
    if (entry.isFile()) total += statSync(join(entry.parentPath, entry.name), { throwIfNoEntry: false })?.size ?? 0
  }
  return total
}

async function isUp(url) {
  try {
    const response = await fetch(url, { signal: AbortSignal.timeout(2000) })
    return response.ok
  } catch {
    return false
  }
}

function delay(ms) {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, ms))
}

/** 轮询直到 url 可用；进程先退出或超时都返回 false。 */
async function waitUntilUp(url, timeoutMs, processExited) {
  let exited = false
  processExited.then(() => {
    exited = true
  })
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline && !exited) {
    if (await isUp(url)) return true
    await delay(500)
  }
  return false
}
