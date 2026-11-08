<#
HiveMemory project services on native Windows, without Docker or WSL.

Usage, from the repository root:
  powershell -File scripts\hivememory-dev.ps1 serve [-Build]
  powershell -File scripts\hivememory-dev.ps1 dev
  powershell -File scripts\hivememory-dev.ps1 status
  powershell -File scripts\hivememory-dev.ps1 stop-qdrant

Qdrant runs in the background in its own hidden console and keeps running
after the backend stops. The backend runs in the foreground of the current
terminal and stops with Ctrl+C. uvicorn treats Ctrl+C as a graceful exit, so
the backend still settles in-memory topics before it exits. Windows has no
SIGTERM for another process, and Stop-Process kills without that settlement,
so this script never stops the backend itself.

  serve  long-running local use: no reload, the backend serves frontend/dist.
  dev    development: the backend reloads on code changes. Run the Vite dev
         server in another terminal: cd frontend; npm run dev

stop-qdrant kills the Qdrant process this script started. Qdrant writes every
change to its WAL before acknowledging it, so a killed instance recovers on
the next start.

Copy scripts\hivememory-dev.env.example to .hivememory\dev.env and set
QDRANT_BIN to the absolute path of qdrant.exe before the first start.
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('serve', 'dev', 'status', 'stop-qdrant', 'help')]
    [string]$Command = 'help',
    [switch]$Build
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvFile = Join-Path $ProjectRoot '.hivememory\dev.env'
$RunDir = Join-Path $ProjectRoot '.hivememory\run'
$LogDir = Join-Path $ProjectRoot '.hivememory\logs'
$QdrantPidFile = Join-Path $RunDir 'qdrant.pid'
$FrontendDistIndex = Join-Path $ProjectRoot 'frontend\dist\index.html'

# Ports match the repository configuration and the WSL/Linux script defaults.
$QdrantReadyUrl = 'http://127.0.0.1:6333/readyz'
$BackendHealthUrl = 'http://127.0.0.1:8769/health'
$FrontendUrl = 'http://127.0.0.1:5173'

# dev.env holds KEY=VALUE lines. Quotes around values are removed; variables
# such as $HOME are not expanded, so paths must be absolute.
function Read-DevEnv {
    $settings = @{}
    if (-not (Test-Path $EnvFile -PathType Leaf)) {
        return $settings
    }
    foreach ($line in Get-Content -Path $EnvFile -Encoding UTF8) {
        $trimmed = $line.Trim()
        if ($trimmed -eq '' -or $trimmed.StartsWith('#') -or -not $trimmed.Contains('=')) {
            continue
        }
        $key, $value = $trimmed.Split('=', 2)
        $settings[$key.Trim()] = $value.Trim().Trim('"').Trim("'")
    }
    return $settings
}

# Health checks bypass the system proxy: a local proxy client must not answer
# for 127.0.0.1.
function Test-Url([string]$Url) {
    try {
        $request = [System.Net.WebRequest]::Create($Url)
        $request.Proxy = $null
        $request.Timeout = 2000
        $request.GetResponse().Close()
        return $true
    } catch {
        return $false
    }
}

function Wait-Url([string]$Url, [string]$Label, [int]$TimeoutSeconds) {
    for ($second = 0; $second -lt $TimeoutSeconds; $second++) {
        if (Test-Url $Url) {
            Write-Host "${Label}: ready"
            return $true
        }
        Start-Sleep -Seconds 1
    }
    Write-Host "${Label}: 在 $TimeoutSeconds 秒内没有就绪" -ForegroundColor Red
    return $false
}

function Get-ManagedQdrant {
    if (-not (Test-Path $QdrantPidFile -PathType Leaf)) {
        return $null
    }
    $managedPid = [int](Get-Content -Path $QdrantPidFile -Raw).Trim()
    $process = Get-Process -Id $managedPid -ErrorAction SilentlyContinue
    if ($null -eq $process -or $process.ProcessName -notlike 'qdrant*') {
        Remove-Item -Path $QdrantPidFile -Force
        return $null
    }
    return $process
}

function Start-Qdrant {
    if (Test-Url $QdrantReadyUrl) {
        Write-Host 'qdrant: 已在运行，直接使用。如果它是 WSL 中的 Qdrant（经本机端口转发），先在 WSL 中停止它。'
        return
    }

    $settings = Read-DevEnv
    $qdrantBin = $settings['QDRANT_BIN']
    if (-not $qdrantBin -or -not (Test-Path $qdrantBin -PathType Leaf)) {
        throw "Qdrant 可执行文件不存在：'$qdrantBin'。请在 $EnvFile 中把 QDRANT_BIN 设为 qdrant.exe 的绝对路径。"
    }
    $storagePath = $settings['QDRANT_STORAGE_PATH']
    if (-not $storagePath) {
        $storagePath = Join-Path $ProjectRoot '.hivememory\qdrant'
    }
    New-Item -ItemType Directory -Force -Path $RunDir, $LogDir, $storagePath | Out-Null

    Write-Host "qdrant: starting (storage=$storagePath)"
    # Start-Process in Windows PowerShell 5.1 has no per-process environment,
    # so the settings are set here for the child to inherit and removed again.
    $env:QDRANT__STORAGE__STORAGE_PATH = $storagePath
    $env:QDRANT__SERVICE__HOST = '127.0.0.1'
    $env:QDRANT__SERVICE__HTTP_PORT = '6333'
    $env:QDRANT__SERVICE__GRPC_PORT = '6334'
    try {
        # The working directory is the run directory because Qdrant writes
        # snapshots and its init marker relative to it.
        $process = Start-Process -FilePath $qdrantBin -WorkingDirectory $RunDir -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $LogDir 'qdrant.log') `
            -RedirectStandardError (Join-Path $LogDir 'qdrant.err.log') -PassThru
    } finally {
        Remove-Item -Path Env:QDRANT__STORAGE__STORAGE_PATH, Env:QDRANT__SERVICE__HOST, `
            Env:QDRANT__SERVICE__HTTP_PORT, Env:QDRANT__SERVICE__GRPC_PORT
    }
    Set-Content -Path $QdrantPidFile -Value $process.Id

    if (-not (Wait-Url $QdrantReadyUrl 'qdrant' 60)) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        Remove-Item -Path $QdrantPidFile -Force
        throw "Qdrant 没有启动成功，见 $LogDir 下的 qdrant.log 与 qdrant.err.log"
    }
}

function Stop-Qdrant {
    $process = Get-ManagedQdrant
    if ($null -eq $process) {
        Write-Host 'qdrant: 未由本脚本运行'
        return
    }
    Stop-Process -Id $process.Id -Force
    Remove-Item -Path $QdrantPidFile -Force
    Write-Host "qdrant: stopped (pid=$($process.Id))"
}

function Build-Frontend {
    Push-Location (Join-Path $ProjectRoot 'frontend')
    try {
        if (-not (Test-Path 'node_modules' -PathType Container)) {
            & npm ci
            if ($LASTEXITCODE -ne 0) { throw 'npm ci 失败' }
        }
        & npm run build
        if ($LASTEXITCODE -ne 0) { throw '前端构建失败' }
    } finally {
        Pop-Location
    }
}

# Runs the backend in the foreground until Ctrl+C. Mode "serve" never reloads
# and serves frontend/dist; mode "dev" reloads on code changes.
function Start-Backend([string]$Mode) {
    if (Test-Url $BackendHealthUrl) {
        throw "$BackendHealthUrl 上已有后端在运行（也可能是 WSL 中的后端），先停止它再启动。"
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw '找不到 uv；先安装 uv，并在仓库根目录执行 uv sync --frozen --extra dev'
    }
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

    $env:HIVEMEMORY_SERVE_FRONTEND = if ($Mode -eq 'serve') { 'true' } else { 'false' }
    $env:HIVEMEMORY__LOGGING__FILE_PATH = Join-Path $LogDir 'backend.log'
    $env:HIVEMEMORY__LOGGING__EVENT_FILE_PATH = Join-Path $LogDir 'runtime-events.jsonl'
    $arguments = @('run', 'uvicorn', 'hivememory.server.app:app', '--host', '127.0.0.1', '--port', '8769')
    if ($Mode -eq 'dev') {
        $arguments += '--reload'
    }

    Write-Host "backend: starting ($Mode)，按 Ctrl+C 停止；退出前会结算内存中的话题，请等它自己退出。"
    Push-Location $ProjectRoot
    try {
        & uv @arguments
    } finally {
        Pop-Location
    }
}

function Show-Status {
    $qdrant = Get-ManagedQdrant
    if ($null -ne $qdrant) {
        Write-Host "qdrant   managed (pid=$($qdrant.Id))"
    }
    foreach ($check in @(
            @('qdrant', $QdrantReadyUrl),
            @('backend', $BackendHealthUrl),
            @('frontend', $FrontendUrl))) {
        $state = if (Test-Url $check[1]) { 'ready' } else { 'not running' }
        Write-Host ('{0,-8} {1} ({2})' -f $check[0], $state, $check[1])
    }
}

switch ($Command) {
    'serve' {
        if ($Build) {
            Build-Frontend
        } elseif (-not (Test-Path $FrontendDistIndex -PathType Leaf)) {
            throw "前端构建产物不存在：$FrontendDistIndex；请执行 serve -Build"
        }
        Start-Qdrant
        Start-Backend 'serve'
    }
    'dev' {
        Start-Qdrant
        Start-Backend 'dev'
    }
    'status' { Show-Status }
    'stop-qdrant' { Stop-Qdrant }
    'help' {
        Write-Host @'
用法（在仓库根目录执行）：
  powershell -File scripts\hivememory-dev.ps1 serve [-Build]   Qdrant 在后台运行，后端在前台运行：不重载，由后端提供前端
  powershell -File scripts\hivememory-dev.ps1 dev              Qdrant 在后台运行，后端在前台运行：随代码改动重载
  powershell -File scripts\hivememory-dev.ps1 status
  powershell -File scripts\hivememory-dev.ps1 stop-qdrant

后端在前台运行，按 Ctrl+C 停止，并等它结算完内存中的话题后自己退出。
dev 模式下，另开一个终端运行前端：cd frontend; npm run dev
日志在 .hivememory\logs；可用 Get-Content -Wait -Tail 100 .hivememory\logs\backend.log 跟踪。

首次使用：
  Copy-Item scripts\hivememory-dev.env.example .hivememory\dev.env
  编辑 .hivememory\dev.env 中的 QDRANT_BIN
'@
    }
}
