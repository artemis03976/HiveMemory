/**
 * 打包桌面客户端：准备随包的后端，再用 electron-builder 生成按用户安装的 Windows 安装包。
 *
 * 随包的内容放在 desktop/build/，作为安装目录中 resources/ 下的三个目录：
 *   python/   uv 管理的独立 CPython，版本与仓库 .venv 相同；锁定的运行依赖装进它自己的
 *             site-packages，整个目录可以随安装位置移动
 *   backend/  与仓库根相同的布局：src/hivememory、configs（config.yaml 与各模板）、
 *             frontend/dist 与 integrations（外部 harness 的插件）；后端按源码位置找到
 *             随程序发布的配置、前端与插件
 *   qdrant/   qdrant.exe，取自 .hivememory/dev.env 的 QDRANT_BIN，与开发时同一个版本
 * 模型不随包，首次启动时由后端下载到数据根。运行时用不到的 pip 与各包自带的测试目录不随包：
 * 随包 Python 的文件数决定安装与覆盖安装的耗时。
 *
 * 用法：node scripts/package.mjs [--dir]；--dir 只生成未打包的目录，不生成安装包。
 */

import { execFileSync, execSync } from 'node:child_process'
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync } from 'node:fs'
import { basename, dirname, join, relative, resolve } from 'node:path'

import builder from 'electron-builder'

const DESKTOP_DIR = resolve(import.meta.dirname, '..')
const REPO_ROOT = resolve(DESKTOP_DIR, '..')
const BUILD_DIR = join(DESKTOP_DIR, 'build')
// 随包的 CPython 装在这里，不进入用户的 uv 目录、PATH 与注册表；保留在两次打包之间。
const PYTHON_INSTALL_DIR = join(DESKTOP_DIR, '.cache', 'python')

function run(command, args, cwd = REPO_ROOT) {
  console.log(`> ${command} ${args.join(' ')}`)
  return execFileSync(command, args, {
    cwd,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'inherit'],
    env: { ...process.env, UV_PYTHON_INSTALL_DIR: PYTHON_INSTALL_DIR },
  }).trim()
}

/** 读取 KEY=VALUE 形式的文件，读法与开发脚本相同。 */
function readKeyValues(path) {
  const values = {}
  for (const line of readFileSync(path, 'utf8').split(/\r?\n/)) {
    const index = line.indexOf('=')
    if (index < 0 || line.trim().startsWith('#')) continue
    values[line.slice(0, index).trim()] = line.slice(index + 1).trim().replace(/^["']|["']$/g, '')
  }
  return values
}

function appVersion() {
  const source = readFileSync(join(REPO_ROOT, 'src', 'hivememory', '_version.py'), 'utf8')
  const match = source.match(/__version__\s*=\s*"([^"]+)"/)
  if (!match) throw new Error('src/hivememory/_version.py 中没有 __version__')
  return match[1]
}

function preparePython() {
  const version = readKeyValues(join(REPO_ROOT, '.venv', 'pyvenv.cfg')).version_info
  run('uv', ['python', 'install', '--no-bin', '--no-registry', version])
  const python = run('uv', ['python', 'find', '--managed-python', version])
  const target = join(BUILD_DIR, 'python')
  cpSync(dirname(python), target, { recursive: true })

  const requirements = join(BUILD_DIR, 'requirements.txt')
  run('uv', ['export', '--frozen', '--no-dev', '--no-emit-project', '--output-file', requirements])
  // 依赖直接装进这个解释器（不建 venv）：venv 记录基础解释器的绝对路径，不能随安装位置移动。
  // 包本身不安装，后端经 PYTHONPATH 从 backend/src 导入。
  run('uv', [
    'pip', 'install',
    '--python', join(target, 'python.exe'),
    '--break-system-packages',
    '--link-mode', 'copy',
    '--requirement', requirements,
  ])
  run('uv', ['pip', 'uninstall', '--python', join(target, 'python.exe'), '--break-system-packages', 'pip'])
  removeTestDirs(join(target, 'Lib', 'site-packages'))
}

/** 删除各包自带的测试目录（tests、test）：运行时不导入它们。 */
function removeTestDirs(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue
    const path = join(dir, entry.name)
    if (entry.name === 'tests' || entry.name === 'test') {
      rmSync(path, { recursive: true, force: true })
    } else {
      removeTestDirs(path)
    }
  }
}

function prepareBackend() {
  const target = join(BUILD_DIR, 'backend')
  cpSync(join(REPO_ROOT, 'src', 'hivememory'), join(target, 'src', 'hivememory'), {
    recursive: true,
    filter: (source) => basename(source) !== '__pycache__',
  })
  // 只带主配置的默认值与模板；个人配置在数据根，由后端在首次启动时从模板生成。
  mkdirSync(join(target, 'configs'), { recursive: true })
  for (const name of ['config.yaml', 'models.example.yaml', 'providers.secrets.example.yaml',
    'system_principals.example.yaml', 'workspace_actors.example.yaml']) {
    cpSync(join(REPO_ROOT, 'configs', name), join(target, 'configs', name))
  }
  // 外部 harness 的插件与本地 marketplace，与后端同版本；插件页从这里为 harness 安装，
  // 覆盖安装时随程序换成新版本。插件自己的测试与 Python 缓存不随包；harness 读取的 skill
  // 由后端启动时按语言设置写入，开发时生成的那一份也不随包。
  const integrationsDir = join(REPO_ROOT, 'integrations')
  const generatedSkill = join('hivememory', 'skills', 'hivememory', 'SKILL.md')
  cpSync(integrationsDir, join(target, 'integrations'), {
    recursive: true,
    filter: (source) =>
      !['__pycache__', 'test'].includes(basename(source)) && relative(integrationsDir, source) !== generatedSkill,
  })
  console.log('> npm run build (frontend)')
  execSync('npm run build', { cwd: join(REPO_ROOT, 'frontend'), stdio: 'inherit' })
  cpSync(join(REPO_ROOT, 'frontend', 'dist'), join(target, 'frontend', 'dist'), { recursive: true })
}

function prepareQdrant() {
  const qdrantBin = readKeyValues(join(REPO_ROOT, '.hivememory', 'dev.env')).QDRANT_BIN
  if (!qdrantBin || !existsSync(qdrantBin)) {
    throw new Error('.hivememory/dev.env 中的 QDRANT_BIN 不存在，无法随包 Qdrant')
  }
  console.log(`> ${run(qdrantBin, ['--version'])}`)
  cpSync(qdrantBin, join(BUILD_DIR, 'qdrant', 'qdrant.exe'))
}

const dirOnly = process.argv.includes('--dir')
rmSync(BUILD_DIR, { recursive: true, force: true })
preparePython()
prepareBackend()
prepareQdrant()

await builder.build({
  targets: builder.Platform.WINDOWS.createTarget(dirOnly ? 'dir' : 'nsis', builder.Arch.x64),
  projectDir: DESKTOP_DIR,
  config: {
    appId: 'local.hivememory.desktop',
    productName: 'HiveMemory',
    // 版本的唯一来源是后端的 _version.py，package.json 中的版本不使用。
    extraMetadata: { version: appVersion() },
    electronDist: join(DESKTOP_DIR, 'node_modules', 'electron', 'dist'),
    directories: { output: 'dist' },
    files: [
      'src/**/*',
      'assets/**/*',
      'package.json',
      // 启动页的字体取自前端依赖，与前端打包进页面的是同一个文件。
      {
        from: '../frontend/node_modules/@fontsource-variable/manrope/files',
        to: 'src/fonts',
        filter: ['manrope-latin-wght-normal.woff2'],
      },
    ],
    extraResources: [
      { from: 'build/python', to: 'python' },
      { from: 'build/backend', to: 'backend' },
      { from: 'build/qdrant', to: 'qdrant' },
    ],
    // Electron 只带中文与英文两个界面语言包。
    electronLanguages: ['zh-CN', 'en-US'],
    win: { icon: 'assets/icon.png' },
    // 按用户安装，不需要管理员权限；卸载只删除程序，数据根与 userData 都保留。
    nsis: {
      oneClick: false,
      perMachine: false,
      allowToChangeInstallationDirectory: true,
      deleteAppDataOnUninstall: false,
      // zip 载荷直接解压到安装目录；7z 载荷先解压到临时目录再复制，一万多个文件要多写、多删一遍。
      // 不做自动升级，用不上差分包，而差分包只能用 7z 载荷。
      useZip: true,
      differentialPackage: false,
      // 覆盖安装与卸载之前先请运行中的程序正常退出。
      include: 'installer/installer.nsh',
    },
  },
})
