/**
 * 程序与数据的位置。
 *
 * 用仓库的 checkout 运行时（npm start），仓库根既是程序根也是数据根：后端用仓库的 .venv，
 * Qdrant 的可执行文件与存储目录读自 .hivememory/dev.env，与 scripts/hivememory-dev.ps1 相同。
 * 安装后，程序在安装目录的 resources 下（布局见 scripts/package.mjs），数据在用户选择的数据根。
 * 数据根之下的布局与仓库根相同：configs/、data/、.hivememory/。
 */

import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

export function checkoutLayout(repoRoot) {
  const devEnv = readKeyValues(join(repoRoot, '.hivememory', 'dev.env'))
  return {
    dataRoot: repoRoot,
    python: join(repoRoot, '.venv', 'Scripts', 'pythonw.exe'),
    pythonEnv: {},
    frontendIndex: join(repoRoot, 'frontend', 'dist', 'index.html'),
    qdrantBin: devEnv.QDRANT_BIN,
    qdrantStorage: devEnv.QDRANT_STORAGE_PATH || join(repoRoot, '.hivememory', 'qdrant'),
  }
}

export function installedLayout(resourcesDir, dataRoot) {
  const programRoot = join(resourcesDir, 'backend')
  return {
    dataRoot,
    python: join(resourcesDir, 'python', 'pythonw.exe'),
    pythonEnv: {
      // 后端没有安装成包：从随包的源码导入，按源码位置找到随程序发布的配置与前端。
      PYTHONPATH: join(programRoot, 'src'),
      // 不让用户目录下装过的包混进随包的依赖。
      PYTHONNOUSERSITE: '1',
      HIVEMEMORY_DATA_ROOT: dataRoot,
    },
    frontendIndex: join(programRoot, 'frontend', 'dist', 'index.html'),
    qdrantBin: join(resourcesDir, 'qdrant', 'qdrant.exe'),
    qdrantStorage: join(dataRoot, '.hivememory', 'qdrant'),
  }
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
