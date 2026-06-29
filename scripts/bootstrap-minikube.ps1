param(
    [Parameter(Mandatory = $true)]
    [string]$GitHubUsername,
    [Parameter(Mandatory = $true)]
    [System.Security.SecureString]$GitHubToken,
    [Parameter(Mandatory = $true)]
    [string]$RepositoryUrl,
    [string]$TargetRevision = "gitops"
)

$ErrorActionPreference = "Stop"
$plainToken = [System.Net.NetworkCredential]::new("", $GitHubToken).Password

foreach ($command in @("minikube", "kubectl", "dvc")) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required command is not installed: $command"
    }
}

minikube status | Out-Null
if ($LASTEXITCODE -ne 0) {
    minikube start --cpus 4 --memory 8192
}

kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n argocd -f "https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml"
kubectl wait --for=condition=Established crd/applications.argoproj.io --timeout=120s
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

kubectl rollout status deployment/minio -n mlops-eyes --timeout=180s
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
    $plainToken = $null
}

kubectl delete pod -n mlops-eyes -l app=backend --ignore-not-found
Write-Host "Argo CD application configured. Check: kubectl get application -n argocd mlops-eyes"
