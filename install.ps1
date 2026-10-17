#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Build, deploy and open Industrial Cognition on a local Kubernetes cluster.

.DESCRIPTION
    The whole sequence in one place, so nobody has to remember the order or paste six
    commands that half-fail:

      1. preflight   - docker, minikube, kubectl, and a *working* Docker engine
      2. cluster     - start minikube if needed, and refresh kubeconfig
      3. images      - build api + vision, load them into the node
      4. deploy      - kubectl apply -k
      5. database    - run Alembic as a Job
      6. rollout     - wait for the API
      7. console     - print the tokenised URL and open the browser

    Two things this deliberately does NOT do, because both were sources of confusing
    failures: it never `kubectl exec`s into a pod (a Pending pod cannot be exec'd into -
    "does not have a host assigned"), and it never port-forwards a Deployment. Migrations
    run as a Job; the console attaches to the Service.

    Idempotent: re-running rebuilds the images, re-applies the manifests and restarts the
    rollouts. Safe to run as a repair command.

.PARAMETER MemoryMB
    Memory for the minikube node (default 10240). The 293 MB model plus Postgres,
    RabbitMQ, Redis and MinIO do not fit in the default 2 GB.

.PARAMETER Cpus
    CPUs for the minikube node (default 8).

.PARAMETER Overlay
    Kustomize overlay to deploy: dev or prod (default dev).

.PARAMETER SkipCluster
    Use the current cluster as-is; do not start or recreate minikube.

.PARAMETER Recreate
    Delete the minikube cluster first. The reliable fix when its API server stops
    answering after a Docker Desktop restart.

.PARAMETER SkipBuild
    Reuse the images already in the node (faster iteration on manifests only).

.PARAMETER NoConsole
    Stop after the rollout instead of opening the console.

.EXAMPLE
    ./install.ps1
    ./install.ps1 -Recreate -MemoryMB 12288
    ./install.ps1 -SkipBuild -NoConsole
#>
param(
    [int]$MemoryMB = 10240,
    [int]$Cpus = 8,
    [ValidateSet("dev", "prod")][string]$Overlay = "dev",
    [string]$Namespace = "cogni-ns",
    [string]$Tag = "dev",
    [switch]$SkipCluster,
    [switch]$Recreate,
    [switch]$SkipBuild,
    [switch]$NoConsole
)

$ErrorActionPreference = "Stop"
# install.ps1 lives at the repository root, so $PSScriptRoot IS the root. Taking its
# parent - the natural-looking choice - walks above the checkout, and `docker build .`
# then fails with "The system cannot find the file specified" for src/.
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

function Step([string]$Message) { Write-Host "`n==> $Message" -ForegroundColor Cyan }
function Ok([string]$Message) { Write-Host "    $Message" -ForegroundColor Green }
function Warn([string]$Message) { Write-Host "warning: $Message" -ForegroundColor Yellow }
function Fail([string]$Message, [string]$Hint = "") {
    Write-Host "error: $Message" -ForegroundColor Red
    if ($Hint) { Write-Host $Hint -ForegroundColor Yellow }
    exit 1
}

function Invoke-Checked([string]$What, [scriptblock]$Action) {
    & $Action
    if ($LASTEXITCODE -ne 0) { Fail "$What failed (exit $LASTEXITCODE)" }
}

# --------------------------------------------------------------------------- #
# 1. preflight
# --------------------------------------------------------------------------- #
Step "preflight"
foreach ($tool in @("docker", "minikube", "kubectl")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Fail "$tool is not installed or not on PATH."
    }
}

$engine = (docker version --format '{{.Server.Version}}' 2>$null)
if (-not $engine) {
    $desktop = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $desktop) {
        Write-Host "    starting Docker Desktop..."
        Start-Process $desktop | Out-Null
        $deadline = (Get-Date).AddSeconds(180)
        while ((Get-Date) -lt $deadline -and -not ($engine = (docker version --format '{{.Server.Version}}' 2>$null))) {
            Start-Sleep -Seconds 3
        }
    }
    if (-not $engine) { Fail "the Docker engine is not responding. Start Docker Desktop and re-run." }
}
Ok "docker $engine"

# --------------------------------------------------------------------------- #
# 2. cluster
# --------------------------------------------------------------------------- #
if ($Recreate) {
    Step "recreating the minikube cluster"
    minikube delete | Out-Null
}

if (-not $SkipCluster) {
    Step "cluster (minikube, ${MemoryMB} MB, $Cpus CPUs)"
    $running = (minikube status --format '{{.Host}}' 2>$null)
    if ($running -notmatch "Running") {
        Invoke-Checked "minikube start" { minikube start --driver=docker --memory=$MemoryMB --cpus=$Cpus }
    } else {
        Ok "already running"
    }
    # After a Docker Desktop restart the API server port can change and the kubeconfig
    # keeps pointing at the old one - which looks exactly like "the cluster is down".
    minikube update-context | Out-Null
}

Invoke-Checked "cluster is reachable" { kubectl get --raw /healthz --request-timeout=15s | Out-Null }
Ok "API server reachable"

# --------------------------------------------------------------------------- #
# 3. images
# --------------------------------------------------------------------------- #
if (-not $SkipBuild) {
    Step "building images"
    Invoke-Checked "api image build" {
        docker build -f src/api/Dockerfile -t "industrial-cognition/api:$Tag" .
    }
    Ok "industrial-cognition/api:$Tag"
    Invoke-Checked "vision image build" {
        docker build -f src/vision/Dockerfile.worker -t "industrial-cognition/vision:$Tag" .
    }
    Ok "industrial-cognition/vision:$Tag"

    Step "loading images into the node"
    # `minikube docker-env` does not work with the containerd runtime on Windows
    # ("SSH_AGENT_START ... not yet supported"), so images are built with the default
    # daemon and loaded explicitly.
    Invoke-Checked "load api image" { minikube image load "industrial-cognition/api:$Tag" }
    Invoke-Checked "load vision image" { minikube image load "industrial-cognition/vision:$Tag" }
    Ok "loaded"
}

# --------------------------------------------------------------------------- #
# 4. deploy
# --------------------------------------------------------------------------- #
Step "applying k8s/overlays/$Overlay"
Invoke-Checked "kubectl apply" {
    kubectl apply -k "k8s/overlays/$Overlay" --request-timeout=120s
}

# --------------------------------------------------------------------------- #
# 5. database (a Job, not exec: exec needs a scheduled pod and this must not)
# --------------------------------------------------------------------------- #
Step "running database migrations as a Job"
$jobName = "alembic-migrate"
kubectl -n $Namespace delete job $jobName --ignore-not-found --request-timeout=30s | Out-Null

$job = @"
apiVersion: batch/v1
kind: Job
metadata:
  name: $jobName
  namespace: $Namespace
spec:
  backoffLimit: 3
  ttlSecondsAfterFinished: 600
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: migrate
          image: industrial-cognition/api:$Tag
          imagePullPolicy: IfNotPresent
          command: ["alembic", "upgrade", "head"]
          envFrom:
            - configMapRef: { name: industrial-cognition-config }
            - secretRef: { name: industrial-cognition-secrets }
"@
$jobPath = Join-Path ([IO.Path]::GetTempPath()) "cogni-migrate-job.yaml"
$job | Set-Content -Path $jobPath -Encoding utf8
Invoke-Checked "migration job" { kubectl apply -f $jobPath --request-timeout=60s }

# Wait for postgres to be ready first; the Job's own retries would mask the reason.
Invoke-Checked "postgres ready" {
    kubectl -n $Namespace wait --for=condition=ready pod/postgres-0 --timeout=240s
}
Invoke-Checked "migration completed" {
    kubectl -n $Namespace wait --for=condition=complete "job/$jobName" --timeout=300s
}
Ok "alembic upgrade head"
kubectl -n $Namespace logs "job/$jobName" --tail=5 2>$null | ForEach-Object { "    $_" }

# --------------------------------------------------------------------------- #
# 6. rollout
# --------------------------------------------------------------------------- #
Step "waiting for the api rollout"
Invoke-Checked "api rollout" {
    kubectl -n $Namespace rollout status deploy/api --timeout=420s
}

if ($NoConsole) {
    Step "done"
    Write-Host "    Start the console with: ./scripts/console.ps1"
    exit 0
}

# --------------------------------------------------------------------------- #
# 7. console
# --------------------------------------------------------------------------- #
& "$PSScriptRoot/scripts/console.ps1" -Namespace $Namespace
