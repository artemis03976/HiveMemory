/**
 * 本机服务：Qdrant 与 HiveMemory 后端的启动、就绪等待与正常关闭。
 *
 * 程序与数据的位置由 layout.js 给出；后端以数据根为工作目录，日志写在数据根的 .hivememory/logs。
 * 端口与仓库配置一致，外部 harness 的插件都连 8769，因此被占用时不换端口。
 */

import { spawn } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import {
  closeSync,
  existsSync,
  mkdirSync,
  openSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs'
import { createServer } from 'node:net'
import { join } from 'node:path'

export const BACKEND_ORIGIN = 'http://127.0.0.1:8769'
export const MANAGEMENT_CREDENTIAL_COOKIE = 'hivememory_management'

const BACKEND_HEALTH_URL = `${BACKEND_ORIGIN}/health`
const BACKEND_READY_URL = `${BACKEND_ORIGIN}/health/ready`
const QDRANT_READY_URL = 'http://127.0.0.1:6333/readyz'
const BACKEND_START_TIMEOUT_MS = 120_000
const QDRANT_START_TIMEOUT_MS = 60_000
// 模型未就绪、模型缓存也这么久没有增长、后端也没有报告失败时，不再等待：下载卡住而没有报错时
// 不应让窗口一直停在启动页。
const MODEL_STALL_TIMEOUT_MS = 60_000
// 后端 8769，Qdrant 的 HTTP 6333 与 gRPC 6334。
const SERVICE_PORTS = [8769, 6333, 6334]
// 错误框里显示的后端输出行数：足够看到 Python 的异常类与信息（例如配置中残留的键）。
const OUTPUT_TAIL_LINES = 12

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
   * 允许附着时（用 checkout 运行），8769 上已有后端就直接附着，不启动也不关闭它；这时没有
   * 内存中的凭据，页面由那个后端照旧下发 cookie。不允许附着时，任一端口被占用都报错。
   * 返回 `{ attached, credential }`，credential 是交给后端的管理凭据。
   */
  async start() {
    if (this.layout.attachToRunning && (await isUp(BACKEND_HEALTH_URL))) {
      this.log('8769 上已有后端，附着到它，不启动也不关闭它')
      return { attached: true, credential: null }
    }
    if (!this.layout.attachToRunning) await refuseOccupiedPorts()
    await this.ensureQdrant()
    const credential = randomBytes(32).toString('base64url')
    await this.startBackend(credential)
    return { attached: false, credential }
  }

  /**
   * 等推理模型就绪。首次启动时后端在预热中下载模型，期间以已下载的字节数回调 onDownload。
   *
   * 返回 `{ state: 'ready' }`；后端报告模型加载失败时返回 `{ state: 'failed', models }`；
   * 模型缓存停止增长超过时限、后端也没有报告失败时返回 `{ state: 'stalled' }`。
   */
  async waitForModels(onDownload) {
    const initialBytes = directorySize(this.modelCacheDir)
    let lastBytes = initialBytes
    let lastGrowth = Date.now()
    for (;;) {
      const readiness = await readJson(BACKEND_READY_URL)
      if (readiness?.models_ready) break
      if (readiness?.status === 'model_load_failed') {
        this.log(`推理模型加载失败: models=${readiness.failed_models}`)
        return { state: 'failed', models: readiness.failed_models }
      }
      const bytes = directorySize(this.modelCacheDir)
      if (bytes !== lastBytes) {
        lastBytes = bytes
        lastGrowth = Date.now()
      }
      if (bytes > initialBytes) onDownload(bytes)
      if (Date.now() - lastGrowth > MODEL_STALL_TIMEOUT_MS) return { state: 'stalled' }
      await delay(1000)
    }
    if (lastBytes > initialBytes) this.log(`推理模型已下载: bytes=${lastBytes - initialBytes}`)
    return { state: 'ready' }
  }

  /** Qdrant 与后端都由本应用启动；附着到已在运行的服务时不能删除数据或关闭它们。 */
  get ownsServices() {
    return this.backend !== null && this.qdrant !== null
  }

  /** 后端输出文件的最后几行，用于启动失败与意外退出时的错误框。 */
  backendOutputTail() {
    const path = join(this.logDir, 'backend.stdout.log')
    if (!existsSync(path)) return ''
    return readFileSync(path, 'utf8').trimEnd().split(/\r?\n/).slice(-OUTPUT_TAIL_LINES).join('\n')
  }

  /**
   * 删除记忆数据，在 stop() 之后调用：先删 Qdrant 存储，再以一次性模式运行后端删除它拥有的
   * 记忆数据（Artifact、长期归档、保存的生成工作与预设 Agent 标记）。配置、密钥、模型缓存与
   * 日志保留，数据根本身不删。
   */
  async deleteMemoryData() {
    const { python, pythonEnv, dataRoot, qdrantStorage } = this.layout
    this.log(`删除 Qdrant 存储: ${qdrantStorage}`)
    rmSync(qdrantStorage, { recursive: true, force: true })
    const output = openSync(join(this.logDir, 'backend.stdout.log'), 'a')
    const child = spawn(python, ['-m', 'hivememory.server', '--delete-memory-data'], {
      cwd: dataRoot,
      env: { ...process.env, ...pythonEnv, PYTHONIOENCODING: 'utf-8' },
      stdio: ['ignore', output, output],
      windowsHide: true,
    })
    closeSync(output)
    const code = await new Promise((resolveExit) => child.once('exit', resolveExit))
    if (code !== 0) {
      throw new Error(`删除记忆数据没有完成，退出代码 ${code}\n\n${this.backendOutputTail()}`)
    }
    this.log('记忆数据已删除')
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
    if (this.layout.attachToRunning && (await isUp(QDRANT_READY_URL))) {
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
      throw new Error(`后端没有启动成功。后端最后的输出：\n\n${this.backendOutputTail()}`)
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

/** 安装版不附着：端口被占用多半是开发用的服务（ps1 serve 或 Qdrant）或另一个 HiveMemory。 */
async function refuseOccupiedPorts() {
  for (const port of SERVICE_PORTS) {
    if (!(await isPortFree(port))) {
      throw new Error(
        `端口 ${port} 已被占用。HiveMemory 的后端与记忆库固定使用 8769、6333 与 6334 端口，` +
          '请先关闭占用它的程序（例如开发用的 ps1 serve、Qdrant 或另一个正在运行的 HiveMemory）再打开。',
      )
    }
  }
}

function isPortFree(port) {
  return new Promise((resolvePort) => {
    const probe = createServer()
    probe.once('error', () => resolvePort(false))
    probe.once('listening', () => probe.close(() => resolvePort(true)))
    probe.listen(port, '127.0.0.1')
  })
}

/** 读取 JSON 响应；未就绪时后端以 503 返回 JSON，同样读取。连不上时返回 null。 */
async function readJson(url) {
  try {
    const response = await fetch(url, { signal: AbortSignal.timeout(2000) })
    return await response.json()
  } catch {
    return null
  }
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
