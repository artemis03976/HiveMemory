/**
 * HiveMemory 桌面客户端原型的主进程：托盘常驻，启动并持有本机的 Qdrant 与后端。
 *
 * 关窗只隐藏到托盘；从托盘退出时先请后端正常关闭（结算内存中的话题），等它退出后应用再退出。
 * 管理凭据只在主进程与后端的内存中：经 stdin 交给后端，放进窗口的内存会话，不落盘。
 * 页面由后端提供，只用公开的 HTTP 接口；主进程不执行页面或远程内容。
 */

import { app, BrowserWindow, dialog, Menu, nativeImage, shell, Tray } from 'electron'
import { appendFileSync, mkdirSync } from 'node:fs'
import { join, resolve } from 'node:path'

import { BACKEND_ORIGIN, LocalServices, MANAGEMENT_CREDENTIAL_COOKIE } from './services.js'

const REPO_ROOT = resolve(import.meta.dirname, '..', '..')
const LOG_DIR = join(REPO_ROOT, '.hivememory', 'logs')
const ICON_PATH = join(import.meta.dirname, '..', 'assets', 'icon.png')

function log(message) {
  mkdirSync(LOG_DIR, { recursive: true })
  appendFileSync(join(LOG_DIR, 'desktop.log'), `${new Date().toISOString()} ${message}\n`, 'utf8')
}

const services = new LocalServices({ repoRoot: REPO_ROOT, logDir: LOG_DIR, log })
let window = null
let tray = null
// running → stopping（等待后端结算话题）→ stopped（可以退出）
let lifecycle = 'running'

if (!app.requestSingleInstanceLock()) {
  app.quit()
} else {
  app.on('second-instance', showWindow)
  // 托盘常驻：窗口全部关闭时不退出。
  app.on('window-all-closed', () => {})
  app.on('before-quit', (event) => {
    if (lifecycle === 'stopped') return
    event.preventDefault()
    shutdown('quit')
  })
  app.whenReady().then(startup)
}

async function startup() {
  log(`桌面端启动: repo=${REPO_ROOT}`)
  tray = createTray()
  window = createWindow()
  window.loadFile(join(import.meta.dirname, 'starting.html'))
  window.show()

  let started
  try {
    started = await services.start()
  } catch (error) {
    log(`启动失败: ${error.message}`)
    dialog.showErrorBox('HiveMemory 没有启动', `${error.message}\n\n日志目录：${LOG_DIR}`)
    shutdown('startup-failed')
    return
  }
  services.onUnexpectedExit((code) => {
    log(`后端意外退出: code=${code}`)
    dialog.showErrorBox('HiveMemory 后端意外退出', `退出代码 ${code}。\n\n日志目录：${LOG_DIR}`)
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
  tray.setToolTip(started.attached ? 'HiveMemory：已附着到已在运行的后端' : 'HiveMemory：运行中')
  await window.loadURL(`${BACKEND_ORIGIN}/`)
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    show: false,
    title: 'HiveMemory',
    icon: ICON_PATH,
    webPreferences: {
      // 没有 persist: 前缀的分区只在内存中，cookie 与缓存都不落盘。
      partition: 'hivememory',
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  })
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
  const created = new Tray(nativeImage.createFromPath(ICON_PATH))
  created.setToolTip('HiveMemory：正在启动')
  created.setContextMenu(
    Menu.buildFromTemplate([
      { label: '打开 HiveMemory', click: showWindow },
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
  tray?.setToolTip('HiveMemory：正在结算话题，完成后退出')
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
