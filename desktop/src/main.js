/**
 * HiveMemory 桌面客户端的主进程：托盘常驻，启动并持有本机的 Qdrant 与后端。
 *
 * 关窗只隐藏到托盘；从托盘退出时先请后端正常关闭（结算内存中的话题），等它退出后应用再退出。
 * 管理凭据只在主进程与后端的内存中：经 stdin 交给后端，放进窗口的内存会话，不落盘。
 * 页面由后端提供，只用公开的 HTTP 接口；主进程只做进程管理与原生对话框，不执行页面或远程内容。
 */

import { app, BrowserWindow, dialog, Menu, nativeImage, shell, Tray } from 'electron'
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'

import { checkoutLayout, installedLayout } from './layout.js'
import { BACKEND_ORIGIN, LocalServices, MANAGEMENT_CREDENTIAL_COOKIE } from './services.js'

const ICON_PATH = join(import.meta.dirname, '..', 'assets', 'icon.png')
// 与 data/model_cache 中两份推理模型的实际大小一致，只用于在启动页提示首次下载的总量。
const MODEL_DOWNLOAD_HINT = '约 1.6 GB'
// Qdrant 不能使用超过 260 个字符的路径，记忆库的文件在数据根之下最深约 142 个字符；
// 数据根更长时建集合会失败，后端无法启动。
const MAX_DATA_ROOT_LENGTH = 100
// 登录 Windows 时启动的应用带上这个参数，只进托盘、不弹出窗口。
const HIDDEN_ARG = '--hidden'
const PRODUCT_TITLE = `HiveMemory ${app.getVersion()}`

let logDir = null
let services = null
let window = null
let tray = null
// running → stopping（等待后端结算话题）→ stopped（可以退出）
let lifecycle = 'running'

function log(message) {
  mkdirSync(logDir, { recursive: true })
  appendFileSync(join(logDir, 'desktop.log'), `${new Date().toISOString()} ${message}\n`, 'utf8')
}

if (!app.requestSingleInstanceLock()) {
  app.quit()
} else {
  app.on('second-instance', showWindow)
  // 托盘常驻：窗口全部关闭时不退出。
  app.on('window-all-closed', () => {})
  app.on('before-quit', (event) => {
    if (lifecycle === 'stopped' || !services) return
    event.preventDefault()
    shutdown('quit')
  })
  app.whenReady().then(startup)
}

async function startup() {
  let layout
  if (app.isPackaged) {
    const dataRoot = await resolveDataRoot()
    if (!dataRoot) {
      lifecycle = 'stopped'
      app.quit()
      return
    }
    layout = installedLayout(process.resourcesPath, dataRoot)
  } else {
    layout = checkoutLayout(resolve(import.meta.dirname, '..', '..'))
  }
  logDir = join(layout.dataRoot, '.hivememory', 'logs')
  services = new LocalServices({ layout, log })
  log(`桌面端启动: version=${app.getVersion()} packaged=${app.isPackaged} data_root=${layout.dataRoot}`)

  tray = createTray()
  window = createWindow()
  await window.loadFile(join(import.meta.dirname, 'starting.html'))
  if (!process.argv.includes(HIDDEN_ARG)) window.show()

  let started
  try {
    started = await services.start()
  } catch (error) {
    log(`启动失败: ${error.message}`)
    dialog.showErrorBox('HiveMemory 没有启动', `${error.message}\n\n日志目录：${logDir}`)
    shutdown('startup-failed')
    return
  }
  services.onUnexpectedExit((code) => {
    log(`后端意外退出: code=${code}`)
    dialog.showErrorBox(
      'HiveMemory 后端意外退出',
      `退出代码 ${code}。后端最后的输出：\n\n${services.backendOutputTail()}\n\n日志目录：${logDir}`,
    )
    shutdown('backend-exited')
  })
  if (started.credential) {
    // 凭据只放进窗口的内存会话：页面的请求、事件流与 WebSocket 都会带上它，后端不再下发。
    await window.webContents.session.cookies.set({
      url: BACKEND_ORIGIN,
      name: MANAGEMENT_CREDENTIAL_COOKIE,
      value: started.credential,
      path: '/',
      httpOnly: true,
      sameSite: 'strict',
    })
  }
  tray.setToolTip(started.attached ? `${PRODUCT_TITLE}：已附着到已在运行的后端` : `${PRODUCT_TITLE}：运行中`)

  setStartingStatus('正在加载推理模型…')
  const models = await services.waitForModels((bytes) => {
    const megabytes = Math.round(bytes / 1024 / 1024)
    setStartingStatus(`正在下载推理模型：已下载 ${megabytes} MB，首次启动共需下载${MODEL_DOWNLOAD_HINT}`)
  })
  if (lifecycle !== 'running') return
  if (models.state === 'failed') {
    // 没有推理模型时检索与记忆生成都不可用，停在启动页说明原因；重开应用时重新加载。
    setStartingStatus(
      `推理模型下载或加载失败（${models.models.join('、')}）。请检查网络后，从托盘退出并重新打开 HiveMemory。日志目录：${logDir}`,
    )
    showWindow()
    return
  }
  if (models.state === 'stalled') log('推理模型仍未就绪，模型缓存也不再增长，先打开界面')
  await window.loadURL(`${BACKEND_ORIGIN}/`)
}

/**
 * 安装后的数据根：第一次启动时由用户选择，记在 userData 下，之后沿用。
 * 返回 null 表示用户选择退出。
 */
async function resolveDataRoot() {
  const saved = readDesktopSettings().dataRoot
  if (saved) return saved
  const dataRoot = await chooseDataRoot()
  if (dataRoot) writeDesktopSettings({ dataRoot })
  return dataRoot
}

function desktopSettingsPath() {
  return join(app.getPath('userData'), 'desktop-settings.json')
}

function readDesktopSettings() {
  const path = desktopSettingsPath()
  return existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) : {}
}

function writeDesktopSettings(settings) {
  mkdirSync(app.getPath('userData'), { recursive: true })
  writeFileSync(desktopSettingsPath(), JSON.stringify(settings, null, 2), 'utf8')
}

async function chooseDataRoot() {
  const defaultRoot = join(process.env.LOCALAPPDATA, 'HiveMemory')
  for (;;) {
    const { response } = await dialog.showMessageBox({
      type: 'question',
      title: 'HiveMemory',
      message: '选择 HiveMemory 的数据目录',
      detail:
        '配置、记忆库、模型缓存与日志都放在这里，卸载 HiveMemory 时保留。\n\n' +
        `默认位置：${defaultRoot}\n\n` +
        '已经有 HiveMemory 的数据（例如仓库的 checkout）时，可以直接选择那个目录，不需要搬动文件。',
      buttons: ['使用默认位置', '选择已有目录…', '取消'],
      defaultId: 0,
      cancelId: 2,
      noLink: true,
    })
    if (response === 2) return null
    let chosen = defaultRoot
    if (response === 1) {
      const { canceled, filePaths } = await dialog.showOpenDialog({
        title: '选择已有的 HiveMemory 数据目录',
        properties: ['openDirectory'],
      })
      if (canceled) continue
      chosen = filePaths[0]
    }
    if (chosen.length <= MAX_DATA_ROOT_LENGTH) return chosen
    await dialog.showMessageBox({
      type: 'warning',
      title: 'HiveMemory',
      message: '数据目录的路径太长',
      detail: `${chosen}\n\n记忆库的文件放在数据目录之下很深的位置，数据目录的路径不能超过 ${MAX_DATA_ROOT_LENGTH} 个字符。请选择一个更短的目录。`,
      noLink: true,
    })
  }
}

/** 更换数据目录：不搬移原目录的数据，重启应用后生效。 */
async function changeDataRoot() {
  if (lifecycle !== 'running') return
  const chosen = await chooseDataRoot()
  if (!chosen || chosen === services.layout.dataRoot) return
  writeDesktopSettings({ ...readDesktopSettings(), dataRoot: chosen })
  log(`数据目录改为 ${chosen}，重启应用`)
  app.relaunch({ args: process.argv.slice(1).filter((arg) => arg !== HIDDEN_ARG) })
  shutdown('change-data-root')
}

/**
 * 删除全部记忆数据：二次确认后关闭服务，删除记忆库、Artifact、长期归档与保存的生成工作，
 * 再重启应用。配置、密钥、接入凭据、模型缓存与日志保留；不删除数据目录本身。
 */
async function deleteMemoryData() {
  if (lifecycle !== 'running') return
  if (!services.ownsServices) {
    await dialog.showMessageBox({
      type: 'info',
      title: 'HiveMemory',
      message: '现在不能删除记忆数据',
      detail: '后端或记忆库不是由本应用启动的（附着到了已在运行的服务），请先关闭它们再打开 HiveMemory。',
      noLink: true,
    })
    return
  }
  const first = await dialog.showMessageBox({
    type: 'warning',
    title: 'HiveMemory',
    message: '删除全部记忆数据？',
    detail:
      `数据目录：${services.layout.dataRoot}\n\n` +
      '会删除：全部记忆与 Agent（包括预设 Agent，重启后重新写入）、来源与版本记录、长期归档，以及还没生成完的记忆工作。\n\n' +
      '会保留：配置、Provider 密钥、harness 接入凭据、推理模型与日志。删除后 HiveMemory 会重新启动，可以直接继续使用。',
    buttons: ['继续', '取消'],
    defaultId: 1,
    cancelId: 1,
    noLink: true,
  })
  if (first.response !== 0) return
  const second = await dialog.showMessageBox({
    type: 'warning',
    title: 'HiveMemory',
    message: '确定删除？此操作不能撤销',
    detail: '没有备份，删除的记忆无法找回。',
    buttons: ['删除', '取消'],
    defaultId: 1,
    cancelId: 1,
    noLink: true,
  })
  if (second.response !== 0) return

  log('开始删除记忆数据')
  lifecycle = 'stopping'
  tray.setToolTip(`${PRODUCT_TITLE}：正在删除记忆数据`)
  window?.destroy()
  try {
    await services.stop()
    await services.deleteMemoryData()
  } catch (error) {
    log(`删除记忆数据失败: ${error.message}`)
    dialog.showErrorBox('删除记忆数据没有完成', `${error.message}\n\n日志目录：${logDir}`)
  }
  lifecycle = 'stopped'
  app.relaunch({ args: process.argv.slice(1).filter((arg) => arg !== HIDDEN_ARG) })
  app.quit()
}

/** 更新启动页上的状态文字；启动页不运行脚本，由主进程写入。 */
function setStartingStatus(text) {
  if (!window || window.isDestroyed()) return
  window.webContents
    .executeJavaScript(`document.getElementById('status').textContent = ${JSON.stringify(text)}`)
    .catch(() => {})
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    show: false,
    title: PRODUCT_TITLE,
    icon: ICON_PATH,
    webPreferences: {
      // 没有 persist: 前缀的分区只在内存中，cookie 与缓存都不落盘。
      partition: 'hivememory',
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  })
  // 标题固定带版本号，不随页面标题变化：用户能看出装的是哪一版。
  win.on('page-title-updated', (event) => event.preventDefault())
  win.on('close', (event) => {
    if (lifecycle !== 'running') return
    event.preventDefault()
    win.hide()
  })
  // Windows 关机、重启或注销时不会发出 before-quit，在这里推迟会话结束，等后端结算话题。
  win.on('query-session-end', (event) => {
    log(`Windows 即将结束会话: reasons=${event.reasons}`)
    if (lifecycle !== 'running') return
    event.preventDefault()
    shutdown('windows-session-end')
  })
  win.on('session-end', (event) => log(`Windows 会话结束: reasons=${event.reasons}`))
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('https://') || url.startsWith('http://')) shell.openExternal(url)
    return { action: 'deny' }
  })
  win.webContents.on('will-navigate', (event, url) => {
    if (!url.startsWith(`${BACKEND_ORIGIN}/`)) event.preventDefault()
  })
  return win
}

function createTray() {
  const created = new Tray(nativeImage.createFromPath(ICON_PATH).resize({ width: 32, height: 32 }))
  created.setToolTip(`${PRODUCT_TITLE}：正在启动`)
  // 登录时启动与更换数据目录只对安装版有意义：用 checkout 运行时数据目录就是仓库根。
  const installedOnly = app.isPackaged
    ? [
        {
          label: '登录 Windows 时启动',
          type: 'checkbox',
          checked: app.getLoginItemSettings({ args: [HIDDEN_ARG] }).openAtLogin,
          click: (item) => app.setLoginItemSettings({ openAtLogin: item.checked, args: [HIDDEN_ARG] }),
        },
        { label: '更改数据目录…', click: changeDataRoot },
      ]
    : []
  created.setContextMenu(
    Menu.buildFromTemplate([
      { label: '打开 HiveMemory', click: showWindow },
      { label: '打开数据目录', click: () => shell.openPath(services.layout.dataRoot) },
      { type: 'separator' },
      ...installedOnly,
      { label: '删除全部记忆数据…', click: deleteMemoryData },
      { type: 'separator' },
      { label: '退出', click: () => app.quit() },
    ]),
  )
  created.on('click', showWindow)
  return created
}

function showWindow() {
  if (!window || window.isDestroyed()) return
  window.show()
  window.focus()
}

async function shutdown(reason) {
  if (lifecycle !== 'running') return
  lifecycle = 'stopping'
  log(`开始关闭: reason=${reason}`)
  tray?.setToolTip(`${PRODUCT_TITLE}：正在关闭`)
  // 先关掉页面：页面持有的事件流与 WebSocket 连接会让后端等待它们关闭。
  window?.destroy()
  try {
    await services.stop()
  } catch (error) {
    log(`关闭服务失败: ${error.message}`)
  }
  lifecycle = 'stopped'
  log('桌面端退出')
  app.quit()
}
