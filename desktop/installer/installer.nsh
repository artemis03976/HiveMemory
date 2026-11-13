; 覆盖安装与卸载之前，先请运行中的 HiveMemory 正常退出。
;
; electron-builder 的安装程序发现程序在运行时，直接结束安装目录下的全部进程，后端与 Qdrant
; 也在其中：后端来不及结算话题，关闭时本该写盘的生成工作、写入句柄与任务历史随之丢失。
; 这里先以 --quit 再启动一次程序，运行中的实例收到后正常退出；等安装目录下的进程都退出，
; 最多 QUIT_WAIT_SECONDS 秒，之后照常走默认的检查，仍在运行的强制结束。
;
; 定义 customCheckAppRunning 后，模板不再包含 getProcessInfo.nsh、不再声明 $pid，
; 默认检查 _CHECK_APP_RUNNING 要用到它们，所以在这里补上。

!include "getProcessInfo.nsh"
Var pid

!define QUIT_WAIT_SECONDS 60

!macro customCheckAppRunning
  !insertmacro IS_POWERSHELL_AVAILABLE
  !insertmacro FIND_PROCESS "${APP_EXECUTABLE_FILENAME}" $R0
  ${if} $R0 == 0
  ${andIf} ${FileExists} "$INSTDIR\${APP_EXECUTABLE_FILENAME}"
    DetailPrint `Asking "${PRODUCT_NAME}" to quit.`
    Exec `"$INSTDIR\${APP_EXECUTABLE_FILENAME}" --quit`
    StrCpy $R1 0
    ${Do}
      Sleep 1000
      IntOp $R1 $R1 + 1
      !insertmacro FIND_PROCESS "${APP_EXECUTABLE_FILENAME}" $R0
      ${if} $R1 >= ${QUIT_WAIT_SECONDS}
        ${ExitDo}
      ${endIf}
    ${LoopUntil} $R0 != 0
  ${endIf}
  !insertmacro _CHECK_APP_RUNNING
!macroend
