$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\MinhQuang\DRL'
$python = 'C:\Users\MinhQuang\DRL\venv\Scripts\python.exe'
$trainScript = 'C:\Users\MinhQuang\DRL\DRL_Pathplanning_trainning\Training\train_sac.py'
$config = 'C:\Users\MinhQuang\DRL\DRL_Pathplanning_trainning\config\environment.yaml'
$trainLog = 'C:\Users\MinhQuang\DRL\DRL_Pathplanning_trainning\logs\train_sac_20260612_020637.log'
$metaLog = 'C:\Users\MinhQuang\DRL\DRL_Pathplanning_trainning\logs\train_sac_shutdown_20260612_020637.log'

function Write-Meta([string]$msg) {
    $line = ('[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'), $msg)
    Add-Content -LiteralPath $metaLog -Value $line
}

$now = Get-Date
$deadline = Get-Date -Year $now.Year -Month $now.Month -Day $now.Day -Hour 6 -Minute 30 -Second 0
if ($now -ge $deadline) {
    $deadline = $deadline.AddDays(1)
}

Write-Meta "Wrapper started. Deadline = $deadline"
Write-Meta "Training command: $python $trainScript --config $config --gui false --show false"

Push-Location $repo
try {
    & $python $trainScript --config $config --gui false --show false *>> $trainLog
    $exitCode = $LASTEXITCODE
    $finished = Get-Date
    Write-Meta "Training finished with exit code $exitCode at $finished"

    if ($exitCode -eq 0 -and $finished -lt $deadline) {
        Write-Meta "Training completed before deadline. Scheduling shutdown in 60 seconds."
        shutdown.exe /s /t 60 /c "Training SAC da hoan thanh truoc 06:30 GMT+7. May se tat sau 60 giay."
    }
    elseif ($exitCode -eq 0) {
        Write-Meta "Training completed but after deadline. No shutdown will be triggered."
    }
    else {
        Write-Meta "Training failed. No shutdown will be triggered."
    }
}
catch {
    Write-Meta ("Wrapper exception: " + $_)
}
finally {
    Pop-Location
}
