$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSCommandPath))
$workspace = 'D:\WorldModelTemp\ppo_oracle_search_psro_live_stable_actions_rerollfix'
$manifestPath = Join-Path $workspace 'run_manifest.json'

function Test-RunningManifest {
    param(
        [string]$TargetManifestPath,
        [string]$TargetWorkspace
    )

    if (-not (Test-Path $TargetManifestPath)) {
        return $false
    }

    try {
        $manifest = Get-Content $TargetManifestPath -Raw | ConvertFrom-Json
    }
    catch {
        return $false
    }

    if (-not $manifest -or $manifest.workspace -ne $TargetWorkspace -or -not $manifest.pid) {
        return $false
    }

    return [bool](Get-Process -Id ([int]$manifest.pid) -ErrorAction SilentlyContinue)
}

New-Item -ItemType Directory -Force -Path $workspace | Out-Null

if (Test-RunningManifest -TargetManifestPath $manifestPath -TargetWorkspace $workspace) {
    exit 0
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $workspace ('stdout_resume_' + $ts + '.log')
$stderr = Join-Path $workspace ('stderr_resume_' + $ts + '.log')
$trace = Join-Path $workspace ('launcher_resume_' + $ts + '.log')

function Write-TraceLine {
    param([string]$Message)
    Add-Content -Path $trace -Value ('[' + (Get-Date -Format o) + '] ' + $Message) -Encoding utf8
}

$pythonCommand = Get-Command python -CommandType Application -ErrorAction Stop | Select-Object -First 1
$pythonExe = $pythonCommand.Source
$researchSrc = Join-Path $repoRoot 'research\world_model\src'
$scriptPath = Join-Path $repoRoot 'research\world_model\scripts\run_ppo_main_loop.py'

$existingPythonPath = [System.Environment]::GetEnvironmentVariable('PYTHONPATH', 'Process')
if ([string]::IsNullOrWhiteSpace($existingPythonPath)) {
    $env:PYTHONPATH = $researchSrc
}
else {
    $env:PYTHONPATH = $researchSrc + ';' + $existingPythonPath
}
Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue

$args = @(
    '-u',
    $scriptPath,
    '--workspace', $workspace,
    '--rounds', '100',
    '--bootstrap-episodes-per-matchup', '8',
    '--self-play-episodes-per-matchup', '8',
    '--max-decisions', '512',
    '--draw-penalty', '1.0',
    '--batch-size', '64',
    '--micro-batch-size', '8',
    '--epochs', '12',
    '--bootstrap-epochs', '5',
    '--learning-rate', '0.0003',
    '--weight-decay', '0.0001',
    '--grad-clip-norm', '1.0',
    '--gamma', '0.99',
    '--gae-lambda', '0.95',
    '--entropy-coef', '0.01',
    '--policy-coef', '4.0',
    '--value-coef', '0.5',
    '--belief-coef', '0.2',
    '--oracle-coef', '0.2',
    '--reference-kl-coef', '0.007',
    '--target-kl-low', '0.01',
    '--target-kl-high', '0.05',
    '--use-amp',
    '--device', 'cuda',
    '--workers', '6',
    '--inference-max-batch-size', '192',
    '--inference-max-wait-ms', '4',
    '--retain-round-directories', '6',
    '--resume',
    '--status-print'
)

Write-TraceLine ('repo_root=' + $repoRoot)
Write-TraceLine ('workspace=' + $workspace)
Write-TraceLine ('python=' + $pythonExe)
Write-TraceLine ('stdout=' + $stdout)
Write-TraceLine ('stderr=' + $stderr)
Write-TraceLine ('command=' + $pythonExe + ' ' + ($args -join ' '))

& $pythonExe @args 1>> $stdout 2>> $stderr
$exitCode = $LASTEXITCODE
Write-TraceLine ('exit_code=' + $exitCode)
exit $exitCode
