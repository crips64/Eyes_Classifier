param(
    [string]$MinikubeCommand = "minikube",
    [string]$TargetRevision = "gitops",
    [int]$GitDaemonPort = 9418,
    [bool]$VerifySelfHeal = $true
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

foreach ($command in @("git", "docker", "kubectl", "dvc", $MinikubeCommand)) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required command is not installed: $command"
    }
}
if (-not (Test-Path ".git/HEAD")) {
    throw "The project must be initialized as a Git repository"
}

& $MinikubeCommand status | Out-Null
if ($LASTEXITCODE -ne 0) {
    & $MinikubeCommand start --driver=docker --cpus 4 --memory 8192
}

$sha = (git rev-parse --short=12 HEAD).Trim()
$imageTag = "sha-$sha"
docker build -f docker/Dockerfile.backend -t "mlopseyes-backend:$imageTag" .
docker build -f docker/Dockerfile.frontend -t "mlopseyes-frontend:$imageTag" .
& $MinikubeCommand image load "mlopseyes-backend:$imageTag"
& $MinikubeCommand image load "mlopseyes-frontend:$imageTag"

$worktree = Join-Path $env:TEMP "mlops-eyes-gitops-worktree"
$bareRoot = Join-Path $env:TEMP "mlops-eyes-git-server"
$bareRepo = Join-Path $bareRoot "mlops-eyes.git"
if (Test-Path $worktree) {
    Remove-Item -LiteralPath $worktree -Recurse -Force
}
if (Test-Path $bareRoot) {
    Remove-Item -LiteralPath $bareRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $bareRoot | Out-Null
git clone . $worktree
git -C $worktree config user.name "local-gitops"
git -C $worktree config user.email "local-gitops@example.invalid"
git -C $worktree switch -C $TargetRevision

$overlay = Join-Path $worktree "k8s/overlays/minikube-local/kustomization.yaml"
$overlayText = Get-Content -Raw $overlay
$overlayText = $overlayText.Replace("newTag: latest", "newTag: $imageTag")
Set-Content -LiteralPath $overlay -Value $overlayText -Encoding utf8
$backendManifest = Join-Path $worktree "k8s/base/backend.yaml"
$backendText = Get-Content -Raw $backendManifest
$backendText = $backendText.Replace(
    "value: mlopseyes-backend:latest",
    "value: mlopseyes-backend:$imageTag"
)
Set-Content -LiteralPath $backendManifest -Value $backendText -Encoding utf8
git -C $worktree add k8s
git -C $worktree commit -m "chore(deploy): promote $imageTag"
git clone --bare $worktree $bareRepo

$gitDaemon = Start-Process git `
    -ArgumentList "daemon", "--reuseaddr", "--export-all", "--base-path=$bareRoot", "--port=$GitDaemonPort", $bareRoot `
    -WindowStyle Hidden -PassThru
New-Item -ItemType Directory -Force -Path artifacts | Out-Null
Set-Content -LiteralPath artifacts/local-git-daemon.pid -Value $gitDaemon.Id

kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n argocd -f "https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml"
kubectl wait --for=condition=Established crd/applications.argoproj.io --timeout=120s
kubectl create namespace mlops-eyes --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic ghcr-pull-secret `
    --namespace mlops-eyes `
    --type=kubernetes.io/dockerconfigjson `
    --from-literal=.dockerconfigjson='{"auths":{}}' `
    --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic minio-credentials `
    --namespace mlops-eyes `
    --from-literal=MINIO_ROOT_USER=minioadmin `
    --from-literal=MINIO_ROOT_PASSWORD=minioadmin `
    --dry-run=client -o yaml | kubectl apply -f -

$repositoryUrl = "git://host.minikube.internal:$GitDaemonPort/mlops-eyes.git"
$application = Get-Content -Raw "k8s/argocd/application.yaml"
$application = $application.Replace("REPOSITORY_URL", $repositoryUrl)
$application = $application.Replace("targetRevision: gitops", "targetRevision: $TargetRevision")
$application = $application.Replace(
    "path: k8s/overlays/minikube",
    "path: k8s/overlays/minikube-local"
)
$application | kubectl apply -f -

kubectl rollout status deployment/minio -n mlops-eyes --timeout=240s
$portForward = Start-Process kubectl `
    -ArgumentList "port-forward", "svc/minio-service", "9000:9000", "-n", "mlops-eyes" `
    -WindowStyle Hidden -PassThru
try {
    Start-Sleep -Seconds 3
    $env:AWS_ACCESS_KEY_ID = "minioadmin"
    $env:AWS_SECRET_ACCESS_KEY = "minioadmin"
    dvc remote modify --local minio endpointurl http://localhost:9000
    dvc push data/reference.dvc eye_cnn_best_val_final.pth.dvc
}
finally {
    Stop-Process -Id $portForward.Id -ErrorAction SilentlyContinue
    Remove-Item Env:AWS_ACCESS_KEY_ID -ErrorAction SilentlyContinue
    Remove-Item Env:AWS_SECRET_ACCESS_KEY -ErrorAction SilentlyContinue
}

kubectl wait --for=condition=available deployment/backend -n mlops-eyes --timeout=600s
kubectl wait --for=condition=available deployment/frontend -n mlops-eyes --timeout=300s
if ($VerifySelfHeal) {
    kubectl scale deployment/frontend -n mlops-eyes --replicas=2
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to change frontend replicas for the self-heal check"
    }
    $healed = $false
    for ($attempt = 0; $attempt -lt 24; $attempt++) {
        Start-Sleep -Seconds 5
        $replicas = kubectl get deployment/frontend -n mlops-eyes -o jsonpath='{.spec.replicas}'
        if ($replicas -eq "1") {
            $healed = $true
            break
        }
    }
    if (-not $healed) {
        throw "Argo CD self-heal did not restore frontend replicas"
    }
}
kubectl get application mlops-eyes -n argocd
kubectl get pods,jobs,cronjobs -n mlops-eyes
Write-Host "Local GitOps deployment is ready. Git daemon PID: $($gitDaemon.Id)"
