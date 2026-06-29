param(
    [string]$MinikubeCommand = "minikube",
    [string]$TargetRevision = "gitops",
    [int]$GitDaemonPort = 9418,
    [int]$MinikubeCpus = 4,
    [int]$MinikubeMemoryMb = 6144,
    [int]$MinioLocalPort = 19000,
    [bool]$IncludeWorkingTree = $true,
    [bool]$VerifySelfHeal = $true,
    [bool]$VerifyRepeatedSync = $true
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

foreach ($command in @("git", "docker", "kubectl", $MinikubeCommand)) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required command is not installed: $command"
    }
}
$dvcPython = Join-Path $projectRoot ".venv/Scripts/python.exe"
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
if (-not (Test-Path ".git/HEAD")) {
    throw "The project must be initialized as a Git repository"
}

& $MinikubeCommand status | Out-Null
if ($LASTEXITCODE -ne 0) {
    & $MinikubeCommand start --driver=docker `
        --cpus $MinikubeCpus --memory $MinikubeMemoryMb
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
$gitDaemonPidFile = Join-Path $projectRoot "artifacts/local-git-daemon.pid"
if (Test-Path -LiteralPath $gitDaemonPidFile) {
    $previousGitDaemonPid = Get-Content -LiteralPath $gitDaemonPidFile
    Stop-Process -Id $previousGitDaemonPid -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $gitDaemonPidFile -Force
}
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
if ($IncludeWorkingTree) {
    $snapshotPatch = Join-Path $bareRoot "working-tree.patch"
    git diff --binary HEAD --output=$snapshotPatch
    if ((Get-Item -LiteralPath $snapshotPatch).Length -gt 0) {
        git -C $worktree apply --whitespace=nowarn $snapshotPatch
        if ($LASTEXITCODE -ne 0) {
            throw "Unable to apply tracked working-tree changes to local GitOps snapshot"
        }
    }
    Remove-Item -LiteralPath $snapshotPatch -Force
    $untrackedFiles = git ls-files --others --exclude-standard
    foreach ($relativePath in $untrackedFiles) {
        $sourcePath = Join-Path $projectRoot $relativePath
        $targetPath = Join-Path $worktree $relativePath
        $targetParent = Split-Path -Parent $targetPath
        New-Item -ItemType Directory -Force -Path $targetParent | Out-Null
        Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Force
    }
}

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
git -C $worktree add --all
git -C $worktree commit -m "chore(deploy): promote $imageTag"
git clone --bare $worktree $bareRepo

$gitDaemon = Start-Process git `
    -ArgumentList "daemon", "--reuseaddr", "--export-all", "--base-path=$bareRoot", "--port=$GitDaemonPort", $bareRoot `
    -WindowStyle Hidden -PassThru
New-Item -ItemType Directory -Force -Path artifacts | Out-Null
Set-Content -LiteralPath $gitDaemonPidFile -Value $gitDaemon.Id

kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -
kubectl apply --server-side --force-conflicts -n argocd `
    -f "https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml"
kubectl wait --for=condition=Established crd/applications.argoproj.io --timeout=120s
kubectl rollout status statefulset/argocd-application-controller `
    -n argocd --timeout=300s
kubectl rollout status deployment/argocd-repo-server `
    -n argocd --timeout=300s
kubectl create namespace mlops-eyes --dry-run=client -o yaml | kubectl apply -f -
$emptyDockerConfig = '{\"auths\":{}}'
kubectl create secret generic ghcr-pull-secret `
    --namespace mlops-eyes `
    --type=kubernetes.io/dockerconfigjson `
    --from-literal=.dockerconfigjson=$emptyDockerConfig `
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
if ($VerifyRepeatedSync) {
    $repeatMarker = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $repeatOverlay = Get-Content -Raw $overlay
    $repeatOverlay += @"

patches:
  - target:
      kind: Deployment
      name: frontend
    patch: |-
      - op: add
        path: /metadata/annotations
        value:
          mlops-eyes/repeated-sync: "$repeatMarker"
"@
    Set-Content -LiteralPath $overlay -Value $repeatOverlay -Encoding utf8
    git -C $worktree add "k8s/overlays/minikube-local/kustomization.yaml"
    git -C $worktree commit -m "chore(deploy): verify repeated sync"
    $repeatRevision = (git -C $worktree rev-parse HEAD).Trim()
    git -C $worktree push $bareRepo "HEAD:refs/heads/$TargetRevision"
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to publish the repeated-sync revision"
    }
    kubectl annotate application mlops-eyes -n argocd `
        argocd.argoproj.io/refresh=hard --overwrite
    $repeatedSyncReady = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        Start-Sleep -Seconds 5
        $applicationJson = kubectl get application mlops-eyes -n argocd -o json
        if ($LASTEXITCODE -ne 0) {
            continue
        }
        $applicationStatus = $applicationJson | ConvertFrom-Json
        if (
            $applicationStatus.status.sync.revision -eq $repeatRevision -and
            $applicationStatus.status.sync.status -eq "Synced" -and
            $applicationStatus.status.health.status -eq "Healthy"
        ) {
            $repeatedSyncReady = $true
            break
        }
    }
    if (-not $repeatedSyncReady) {
        throw "Repeated Argo CD sync did not reach Synced / Healthy"
    }
    kubectl wait --for=condition=complete job/bootstrap-model `
        -n mlops-eyes --timeout=300s
    if ($LASTEXITCODE -ne 0) {
        throw "Idempotent bootstrap hook did not complete after repeated sync"
    }
}
kubectl get application mlops-eyes -n argocd
kubectl get pods,jobs,cronjobs -n mlops-eyes
Write-Host "Local GitOps deployment is ready. Git daemon PID: $($gitDaemon.Id)"
