/**
 * 程序与数据的位置。
 *
 * 用仓库的 checkout 运行时（npm start），仓库根既是程序根也是数据根：后端用仓库的 .venv，
 * Qdrant 的可执行文件与存储目录读自 .hivememory/dev.env，与 scripts/hivememory-dev.ps1 相同。
 * 安装后，程序在安装目录的 resources 下（布局见 scripts/package.mjs），数据在用户选择的数据根。
 * 数据根之下的布局与仓库根相同：configs/、data/、.hivememory/。
 *
 * attachToRunning：端口上已有后端或 Qdrant 时是否直接使用。用 checkout 运行时允许，方便与开发
 * 脚本并用；安装后不允许，否则开发用的 Qdrant 正在运行时，安装版会读写 checkout 的记忆库。
 *
 * ports：后端与 Qdrant 的端口。安装版固定为缺省的一组，外部 harness 的插件都连 8769。用 checkout
 * 运行的是开发实例，端口与开发脚本一样读自 .hivememory/dev.env，在那里换一组端口就能与安装版同时运行。
 *
 * 只有安装版代为安装插件：代为安装会把 harness 改指到本实例的插件目录，并改写用户环境变量中的凭据。
 */

import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

const DEFAULT_PORTS = { backend: 8769, qdrantHttp: 6333, qdrantGrpc: 6334 }

export function checkoutLayout(repoRoot) {
  const devEnv = readKeyValues(join(repoRoot, '.hivememory', 'dev.env'))
  return {
    dataRoot: repoRoot,
    attachToRunning: true,
    python: join(repoRoot, '.venv', 'Scripts', 'pythonw.exe'),
    pythonEnv: {},
    frontendIndex: join(repoRoot, 'frontend', 'dist', 'index.html'),
    qdrantBin: devEnv.QDRANT_BIN,
    qdrantStorage: devEnv.QDRANT_STORAGE_PATH || join(repoRoot, '.hivememory', 'qdrant'),
    ports: {
      backend: portSetting(devEnv, 'HIVEMEMORY_BACKEND_PORT', DEFAULT_PORTS.backend),
      qdrantHttp: portSetting(devEnv, 'QDRANT_HTTP_PORT', DEFAULT_PORTS.qdrantHttp),
      qdrantGrpc: portSetting(devEnv, 'QDRANT_GRPC_PORT', DEFAULT_PORTS.qdrantGrpc),
    },
  }
}

export function installedLayout(resourcesDir, dataRoot) {
  const programRoot = join(resourcesDir, 'backend')
  return {
    dataRoot,
    attachToRunning: false,
    python: join(resourcesDir, 'python', 'pythonw.exe'),
    pythonEnv: {
      // 后端没有安装成包：从随包的源码导入，按源码位置找到随程序发布的配置与前端。
      PYTHONPATH: join(programRoot, 'src'),
      // 不让用户目录下装过的包混进随包的依赖。
      PYTHONNOUSERSITE: '1',
      HIVEMEMORY_DATA_ROOT: dataRoot,
      HIVEMEMORY__SYSTEM__PLUGIN_INSTALLATION_ENABLED: 'true',
    },
    frontendIndex: join(programRoot, 'frontend', 'dist', 'index.html'),
    qdrantBin: join(resourcesDir, 'qdrant', 'qdrant.exe'),
    qdrantStorage: join(dataRoot, '.hivememory', 'qdrant'),
    ports: DEFAULT_PORTS,
  }
}

function portSetting(settings, key, fallback) {
  if (!settings[key]) return fallback
  const port = Number(settings[key])
  if (!Number.isInteger(port) || port <= 0 || port > 65535) {
    throw new Error(`dev.env 中的 ${key} 不是有效端口：'${settings[key]}'`)
  }
  return port
}

/** 每行一个 KEY=VALUE，去掉值两侧的引号，不展开变量，与开发脚本的读法相同；文件不存在时为空。 */
function readKeyValues(path) {
  const settings = {}
  if (!existsSync(path)) return settings
  for (const line of readFileSync(path, 'utf8').split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#') || !trimmed.includes('=')) continue
    const index = trimmed.indexOf('=')
    settings[trimmed.slice(0, index).trim()] = trimmed.slice(index + 1).trim().replace(/^["']|["']$/g, '')
  }
  return settings
}
