#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Check that a local cluster can actually host Industrial Cognition - and repair it.

.DESCRIPTION
    `minikube start` can exit 0 while its addons failed. The tell-tale is:

        Enabling 'storage-provisioner' returned an error: ... failed to download openapi:
        Get "https://localhost:8443/openapi/v2": dial tcp [::1]:8443: connect: connection
        refused

    The API server inside the node was not answering when minikube applied its addons. The
    cluster then has **no default StorageClass**, so every PersistentVolumeClaim stays
    Pending, the Postgres StatefulSet never schedules, and the install fails much later
    with a misleading error ("pod does not have a host assigned"). This script asks the
    questions in the right order and names the remedy.

    Checks, in dependency order:
      1. minikube host and Kubernetes are Running/Configured
      2. the API server answers from the host
      3. the node is Ready
      4. a default StorageClass exists
      5. (with -Fix) missing addons are re-enabled once the API server is up

.PARAMETER Fix
    Try to repair what is repairable: re-enable storage-provisioner and default-storageclass.

.PARAMETER TimeoutSeconds
    How long to wait for the API server and node (default 180).

.EXAMPLE
    ./scripts/cluster-doctor.ps1
    ./scripts/cluster-doctor.ps1 -Fix
#>
param(
    [switch]$Fix,
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = "Continue"

function Step([string]$Message) { Write-Host "`n==> $Message" -ForegroundColor Cyan }
function Good([string]$Message) { Write-Host "    ok: $Message" -ForegroundColor Green }
function Bad([string]$Message) { Write-Host "    FAIL: $Message" -ForegroundColor Red }

# Native stderr must not be treated as fatal (see scripts/console.ps1 for the full story):
# kubectl and minikube both write progress and deprecation notices there.
function Invoke-Native([string]$Exe, [string[]]$Arguments) {
    $app = Get-Command $Exe -CommandType Application -ErrorAction SilentlyContinue
    if (-not $app) { return $null }
    $out = & $app.Source @Arguments 2>$null
    $global:LASTEXITCODE = $LASTEXITCODE
    return $out
}

function Explain-AddonFailure([string]$Text) {
    <#
        Map the failure text a user actually sees onto a cause and an action. Kept as a
        pure function of text so it can be checked against the real message.
    #>
    if ($Text -match "failed to download openapi|connection refused|8443") {
        return @{
            Cause  = "the API server inside the node was not answering when minikube applied its addons"
            Action = @(
                "The node is usually too small. Recreate it with more room:",
                "    minikube delete",
                "    minikube start --driver=docker --memory=12288 --cpus=8",
                "then re-run: ./scripts/cluster-doctor.ps1 -Fix"
            )
        }
    }
    if ($Text -match "no space left|evicted|disk") {
        return @{
            Cause  = "the node ran out of disk"
            Action = @("Prune unused images, then recreate:  minikube delete && docker system prune -f")
        }
    }
    if ($Text -match "forbidden|unauthorized|RBAC") {
        return @{
            Cause  = "the kubeconfig lacks permission to apply addons"
            Action = @("Run 'minikube update-context' and retry; if it persists, recreate the cluster.")
        }
    }
    return @{ Cause = "unknown"; Action = @("Inspect: minikube addons list") }
}

function Test-Minikube {
    $hostState = (Invoke-Native "minikube" @("status", "--format", "{{.Host}}")) -join ""
    $k8sState = (Invoke-Native "minikube" @("status", "--format", "{{.Kubelet}}")) -join ""
    if ($hostState -notmatch "Running") {
        Bad "minikube host is '$hostState'"
        return $false
    }
    Good "minikube host running"
    return $true
}

function Wait-ApiServer {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $null = Invoke-Native "kubectl" @("get", "--raw", "/healthz", "--request-timeout=10s")
        if ($LASTEXITCODE -eq 0) { return $true }
        Start-Sleep -Seconds 3
    }
    return $false
}

function Get-DefaultStorageClass {
    $json = Invoke-Native "kubectl" @("get", "storageclass", "-o", "json", "--request-timeout=15s")
    if ([string]::IsNullOrWhiteSpace(($json -join ""))) { return $null }
    try {
        $parsed = ($json -join "`n") | ConvertFrom-Json
    } catch {
        return $null
    }
    foreach ($class in $parsed.items) {
        if ($class.metadata.annotations.'storageclass.kubernetes.io/is-default-class' -eq "true") {
            return $class.metadata.name
        }
        if ($class.metadata.annotations.'storageclass.beta.kubernetes.io/is-default-class' -eq "true") {
            return $class.metadata.name
        }
    }
    return $null
}

# --------------------------------------------------------------------------- #
Step "cluster doctor"
# --------------------------------------------------------------------------- #
if (-not (Test-Minikube)) {
    Write-Host "`n  The cluster is not up. Start it with: minikube start --memory=10240 --cpus=8" -ForegroundColor Yellow
    exit 2
}

# A stale API-server port in the kubeconfig looks exactly like a dead cluster.
$null = Invoke-Native "minikube" @("update-context")

Step "API server reachable from this host"
if (-not (Wait-ApiServer)) {
    Bad "the API server did not answer within ${TimeoutSeconds}s"
    Write-Host @"

  If minikube reported addon failures like:
    failed to download openapi: Get "https://localhost:8443/openapi/v2": connection refused
  then the node's control plane never became healthy. Recreate it with more room:

      minikube delete
      minikube start --driver=docker --memory=12288 --cpus=8
      ./scripts/cluster-doctor.ps1 -Fix
"@ -ForegroundColor Yellow
    exit 2
}
Good "API server answering"

Step "node Ready"
$nodeReady = $false
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    $node = (Invoke-Native "kubectl" @("get", "nodes", "--no-headers", "--request-timeout=10s")) -join "`n"
    if ($node -match "\sReady\s") { $nodeReady = $true; break }
    Start-Sleep -Seconds 3
}
if (-not $nodeReady) {
    Bad "no node reached Ready within ${TimeoutSeconds}s"
    Invoke-Native "kubectl" @("get", "nodes") | ForEach-Object { "    $_" }
    exit 2
}
Good "node Ready"

Step "default StorageClass"
$default = Get-DefaultStorageClass
if ($default) {
    Good "default StorageClass is '$default'"
} else {
    Bad "no default StorageClass: every PersistentVolumeClaim will stay Pending"

    if ($Fix) {
        Write-Host "    re-enabling the storage addons (the API server is up now)..."
        $output = Invoke-Native "minikube" @("addons", "enable", "storage-provisioner")
        $output += Invoke-Native "minikube" @("addons", "enable", "default-storageclass")
        $text = ($output -join "`n")

        if ($text -match "error|returned an error") {
            $diagnosis = Explain-AddonFailure $text
            Write-Host "`n  Addon failure." -ForegroundColor Yellow
            Write-Host "  Cause:  $($diagnosis.Cause)" -ForegroundColor Yellow
            Write-Host "  Action:" -ForegroundColor Yellow
            $diagnosis.Action | ForEach-Object { Write-Host "    $_" -ForegroundColor Yellow }
            exit 1
        }

        $default = Get-DefaultStorageClass
        if ($default) {
            Good "default StorageClass is now '$default'"
        } else {
            Bad "addons reported success but no default StorageClass exists"
            Invoke-Native "minikube" @("addons", "list") | ForEach-Object { "    $_" }
            exit 1
        }
    } else {
        Write-Host "    run: ./scripts/cluster-doctor.ps1 -Fix" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "`n  Cluster is ready for Industrial Cognition." -ForegroundColor Green
exit 0
