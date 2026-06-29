param(
    [Parameter(Mandatory = $true)]
    [string]$GitHubUsername,
    [Parameter(Mandatory = $true)]
    [System.Security.SecureString]$GitHubToken,
    [Parameter(Mandatory = $true)]
    [string]$RepositoryUrl,
    [string]$TargetRevision = "gitops",
    [int]$MinikubeCpus = 4,
    [int]$MinikubeMemoryMb = 6144,
    [int]$MinioLocalPort = 19000
)

$ErrorActionPreference = "Stop"
$plainToken = [System.Net.NetworkCredential]::new("", $GitHubToken).Password
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$dvcPython = Join-Path $projectRoot ".venv/Scripts/python.exe"

foreach ($command in @("minikube", "kubectl")) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required command is not installed: $command"
    }
}
if (-not (Test-Path -LiteralPath $dvcPython) -and
    -not (Get-Command dvc -ErrorAction SilentlyContinue)) {
    throw "DVC is not available in .venv or PATH"
}
function Invoke-Dvc {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$DvcArgs)
    if (Test-Path -LiteralPath $dvcPython) {
        & $dvcPython -m dvc @DvcArgs
    }
    else {
        & dvc @DvcArgs
    }
    if ($LASTEXITCODE -ne 0) {
        throw "DVC command failed: $($DvcArgs -join ' ')"
    }
}

minikube status | Out-Null
if ($LASTEXITCODE -ne 0) {
    minikube start --cpus $MinikubeCpus --memory $MinikubeMemoryMb
}

kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -
kubectl apply --server-side --force-conflicts -n argocd `
    -f "https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml"
kubectl wait --for=condition=Established crd/applications.argoproj.io --timeout=120s
kubectl rollout status statefulset/argocd-application-controller `
    -n argocd --timeout=300s
kubectl rollout status deployment/argocd-repo-server `
    -n argocd --timeout=300s
kubectl create namespace mlops-eyes --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret docker-registry ghcr-pull-secret `
    --namespace mlops-eyes `
    --docker-server ghcr.io `
    --docker-username $GitHubUsername `
    --docker-password $plainToken `
    --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic minio-credentials `
    --namespace mlops-eyes `
    --from-literal=MINIO_ROOT_USER=minioadmin `
    --from-literal=MINIO_ROOT_PASSWORD=minioadmin `
    --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic mlops-eyes-private-repo `
    --namespace argocd `
    --from-literal=type=git `
    --from-literal=url=$RepositoryUrl `
    --from-literal=username=$GitHubUsername `
    --from-literal=password=$plainToken `
    --dry-run=client -o yaml | kubectl apply -f -
kubectl label secret mlops-eyes-private-repo `
    --namespace argocd argocd.argoproj.io/secret-type=repository --overwrite

$application = Get-Content -Raw "k8s/argocd/application.yaml"
$application = $application.Replace(
    "REPOSITORY_URL",
    $RepositoryUrl
).Replace("targetRevision: gitops", "targetRevision: $TargetRevision")
$application | kubectl apply -f -
$activeOperation = kubectl get application mlops-eyes -n argocd `
    -o jsonpath='{.operation.sync.revision}'
if ($activeOperation) {
    $clearOperationPatch = '{\"operation\":null}'
    kubectl patch application mlops-eyes -n argocd `
        --type=merge -p $clearOperationPatch
}
kubectl annotate application mlops-eyes -n argocd `
    argocd.argoproj.io/refresh=hard --overwrite

$minioCreated = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    kubectl get deployment/minio -n mlops-eyes | Out-Null
    $deploymentExists = $LASTEXITCODE -eq 0
    kubectl get service/minio-service -n mlops-eyes | Out-Null
    $serviceExists = $LASTEXITCODE -eq 0
    if ($deploymentExists -and $serviceExists) {
        $minioCreated = $true
        break
    }
    Start-Sleep -Seconds 5
}
if (-not $minioCreated) {
    throw "Argo CD did not create the MinIO deployment"
}
kubectl wait --for=condition=available deployment/minio `
    -n mlops-eyes --timeout=240s
$portForward = Start-Process kubectl `
    -ArgumentList "port-forward", "svc/minio-service", "${MinioLocalPort}:9000", "-n", "mlops-eyes" `
    -WindowStyle Hidden -PassThru
try {
    $minioForwardReady = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        try {
            $health = Invoke-WebRequest `
                "http://127.0.0.1:$MinioLocalPort/minio/health/ready" `
                -UseBasicParsing -TimeoutSec 2
            if ($health.StatusCode -eq 200) {
                $minioForwardReady = $true
                break
            }
        }
        catch {
            Start-Sleep -Seconds 2
        }
    }
    if (-not $minioForwardReady) {
        throw "MinIO port-forward did not become ready"
    }
    $env:AWS_ACCESS_KEY_ID = "minioadmin"
    $env:AWS_SECRET_ACCESS_KEY = "minioadmin"
    Invoke-Dvc remote modify --local minio endpointurl "http://localhost:$MinioLocalPort"
    Invoke-Dvc push data/reference.dvc eye_cnn_best_val_final.pth.dvc
}
finally {
    Stop-Process -Id $portForward.Id -ErrorAction SilentlyContinue
    Remove-Item Env:AWS_ACCESS_KEY_ID -ErrorAction SilentlyContinue
    Remove-Item Env:AWS_SECRET_ACCESS_KEY -ErrorAction SilentlyContinue
    $plainToken = $null
}

kubectl delete pod -n mlops-eyes -l app=backend --ignore-not-found
Write-Host "Argo CD application configured. Check: kubectl get application -n argocd mlops-eyes"
