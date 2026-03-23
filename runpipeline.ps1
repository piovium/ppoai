$cmd = 'cmd /c "cd /d E:\Coding\WorldModel && resume_event_loop.cmd"'
$match = 'resume_event_loop.cmd'
while ($true) {
    $running = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*$match*" }
    if (-not $running) { Start-Process cmd.exe -ArgumentList '/c', "cd /d E:\Coding\WorldModel && resume_event_loop.cmd" -WindowStyle Minimized }
    Start-Sleep -Seconds 3600
}