@echo off
setlocal

set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"
set "WORKSPACE=D:\WorldModelTemp\ppo_oracle_search_psro_live_stable_actions_rerollfix"
set "LAUNCHER=%REPO_ROOT%\resume_event_loop_detached.ps1"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$script = [System.IO.Path]::GetFullPath('%LAUNCHER%');" ^
  "$proc = Start-Process powershell -WindowStyle Hidden -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$script) -WorkingDirectory '%REPO_ROOT%' -PassThru;" ^
  "Write-Host ('Launcher PID=' + $proc.Id);" ^
  "Write-Host ('Workspace=' + '%WORKSPACE%');"

echo.
echo Workspace: %WORKSPACE%
echo Double-click this file again after an unexpected exit to resume.
pause
