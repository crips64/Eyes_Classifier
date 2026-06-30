param(
    [string]$Namespace = "mlops-eyes",
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 8501
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot
New-Item -ItemType Directory -Force -Path artifacts/smoke-minikube | Out-Null

function Test-LocalPortAvailable {
    param([int]$Port)
    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new(
            [System.Net.IPAddress]::Parse("127.0.0.1"),
            $Port
        )
        $listener.Start()
        return $true
    }
    catch {
        return $false
    }
    finally {
        if ($listener) {
            $listener.Stop()
        }
    }
}

function Get-FreeLocalPort {
    $listener = [System.Net.Sockets.TcpListener]::new(
        [System.Net.IPAddress]::Parse("127.0.0.1"),
        0
    )
    try {
        $listener.Start()
        return $listener.LocalEndpoint.Port
    }
    finally {
        $listener.Stop()
    }
}

function Resolve-ForwardPort {
    param(
        [int]$PreferredPort,
        [string]$Name
    )
    if (Test-LocalPortAvailable $PreferredPort) {
        return $PreferredPort
    }
    $fallbackPort = Get-FreeLocalPort
    Write-Host "$Name local port $PreferredPort is unavailable; using $fallbackPort"
    return $fallbackPort
}

$BackendPort = Resolve-ForwardPort $BackendPort "Backend"
$FrontendPort = Resolve-ForwardPort $FrontendPort "Frontend"

$backendForward = Start-Process kubectl `
    -ArgumentList "port-forward", "svc/backend-service", "$BackendPort`:8000", "-n", $Namespace `
    -RedirectStandardOutput artifacts/smoke-minikube/backend-port-forward.log `
    -RedirectStandardError artifacts/smoke-minikube/backend-port-forward.err.log `
    -WindowStyle Hidden -PassThru
$frontendForward = Start-Process kubectl `
    -ArgumentList "port-forward", "svc/frontend-service", "$FrontendPort`:8501", "-n", $Namespace `
    -RedirectStandardOutput artifacts/smoke-minikube/frontend-port-forward.log `
    -RedirectStandardError artifacts/smoke-minikube/frontend-port-forward.err.log `
    -WindowStyle Hidden -PassThru

try {
    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        try {
            $readiness = Invoke-RestMethod "http://127.0.0.1:$BackendPort/ready" -TimeoutSec 3
            if ($readiness.ready) {
                $ready = $true
                break
            }
        }
        catch {
            Start-Sleep -Seconds 2
        }
    }
    if (-not $ready) {
        throw "Backend readiness check failed"
    }
    $initialVersion = (Invoke-RestMethod "http://127.0.0.1:$BackendPort/health").model_version

    $samples = Get-ChildItem data/reference/opened -File | Select-Object -First 20
    if ($samples.Count -lt 20) {
        throw "At least 20 opened reference samples are required for the smoke test"
    }
    foreach ($sample in $samples) {
        $response = & curl.exe -sS -X POST "http://127.0.0.1:$BackendPort/predict" `
            -F "file=@$($sample.FullName)" `
            -F "true_label=opened"
        if ($LASTEXITCODE -ne 0) {
            throw "Inference upload failed for $($sample.FullName)"
        }
        $prediction = $response | ConvertFrom-Json
        if (-not $prediction.prediction_id) {
            throw "Prediction response is invalid"
        }
    }

    $drift = Invoke-RestMethod `
        -Method Post `
        -Uri "http://127.0.0.1:$BackendPort/drift/run" `
        -TimeoutSec 600
    if ($drift.status -ne "warning") {
        throw "Expected drift warning, received: $($drift.status)"
    }
    $retrainJobId = $drift.auto_retrain_job_id
    if (-not $retrainJobId -and $drift.message -like "*cooldown*") {
        Write-Host "Auto retrain cooldown is active; using manual retrain for rerun"
        $manualRetrain = Invoke-RestMethod `
            -Method Post `
            -Uri "http://127.0.0.1:$BackendPort/retrain" `
            -ContentType "application/json" `
            -Body '{"epochs":2}' `
            -TimeoutSec 30
        $retrainJobId = $manualRetrain.job_id
    }
    if (-not $retrainJobId) {
        throw "Retraining was not scheduled: $($drift.message)"
    }

    kubectl wait `
        --for=condition=complete `
        "job/$retrainJobId" `
        -n $Namespace `
        --timeout=1200s
    if ($LASTEXITCODE -ne 0) {
        throw "Automatic retraining Job did not complete"
    }
    Invoke-RestMethod "http://127.0.0.1:$BackendPort/retrain/status" | Out-Null

    $activeVersion = $initialVersion
    for ($attempt = 0; $attempt -lt 24; $attempt++) {
        Start-Sleep -Seconds 5
        $activeVersion = (
            Invoke-RestMethod "http://127.0.0.1:$BackendPort/health"
        ).model_version
        if ($activeVersion -ne $initialVersion) {
            break
        }
    }
    if ($activeVersion -eq $initialVersion) {
        Write-Host "Readiness diagnostics:"
        Invoke-RestMethod "http://127.0.0.1:$BackendPort/ready" |
            ConvertTo-Json -Depth 8 | Write-Host
        Write-Host "Registered model diagnostics:"
        Invoke-RestMethod "http://127.0.0.1:$BackendPort/models" |
            ConvertTo-Json -Depth 8 | Write-Host
        Write-Host "Retrain history:"
        Invoke-RestMethod "http://127.0.0.1:$BackendPort/retrain/status" |
            ConvertTo-Json -Depth 8 | Write-Host
        Write-Host "Retrain Job logs:"
        kubectl logs "job/$retrainJobId" -n $Namespace --tail=200
        Write-Host "Backend logs:"
        kubectl logs deployment/backend -n $Namespace --tail=200
        throw "The promoted MLflow model was not hot-reloaded"
    }

    $experiments = Invoke-RestMethod "http://127.0.0.1:$BackendPort/experiments"
    if ($experiments.status -ne "ok") {
        throw "MLflow experiments API is unavailable"
    }
    $metrics = Invoke-WebRequest `
        "http://127.0.0.1:$BackendPort/metrics" `
        -UseBasicParsing
    if (-not $metrics.Content.Contains("mlops_prediction_drift_detected")) {
        throw "Expected Prometheus drift metrics are missing"
    }
    $frontend = Invoke-WebRequest `
        "http://127.0.0.1:$FrontendPort/_stcore/health" `
        -UseBasicParsing
    if ($frontend.StatusCode -ne 200) {
        throw "Frontend health check failed"
    }
    Write-Host "End-to-end smoke passed: $initialVersion -> $activeVersion"
}
finally {
    Stop-Process -Id $backendForward.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $frontendForward.Id -Force -ErrorAction SilentlyContinue
}
