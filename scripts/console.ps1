#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Open the Industrial Cognition operator console, like `jupyter lab` does.

.DESCRIPTION
    Resolves the deployment token, forwards the API service to localhost, waits until it
    actually answers, prints the URL and opens the browser.

    Two deliberate choices, both learned from trying to do this by hand:

    * It forwards the **Service**, not a pod. `kubectl port-forward deploy/api` fails while
      a pod is Pending ("does not have a host assigned") and `exec` needs a scheduled pod,
      but the Service has endpoints as soon as one replica is ready - and this script waits
      for that instead of failing.
    * It never uses `kubectl exec`. Nothing about reaching the console should depend on a
      pod being schedulable.

    The token is printed because that is what Jupyter does: it is a throwaway development
    credential, and hiding it in a Secret while asking a human to type it is friction
    without security. Rotate AUTH_TOKEN/JWT_SECRET for anything shared.

.PARAMETER Port
    Local port to forward to (default 8080).

.PARAMETER Namespace
    Namespace of the deployment (default cogni-ns).

.PARAMETER PrintOnly
    Print the URL and exit without forwarding or opening a browser.

.PARAMETER NoBrowser
    Forward and print, but do not open a browser.

.PARAMETER TimeoutSeconds
    How long to wait for the API to answer (default 180).

.EXAMPLE
    ./scripts/console.ps1
    ./scripts/console.ps1 -Port 9000 -NoBrowser
#>
param(
    [int]$Port = 8080,
    [string]$Namespace = "cogni-ns",
    [string]$Service = "api",
    [switch]$PrintOnly,
    [switch]$NoBrowser,
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host "==> $Message" -ForegroundColor Cyan
}
function Write-Warn([string]$Message) {
    Write-Host "warning: $Message" -ForegroundColor Yellow
}
function Fail([string]$Message, [string]$Hint = "") {
    Write-Host "error: $Message" -ForegroundColor Red
    if ($Hint) { Write-Host $Hint -ForegroundColor Yellow }
    exit 1
}

function Test-Cluster {
    $null = kubectl get --raw /healthz --request-timeout=10s 2>&1
    if ($LASTEXITCODE -ne 0) {
        Fail "the Kubernetes API server is not answering (context: $(kubectl config current-context 2>$null))." @"
  The cluster is not reachable. If this is minikube after a Docker restart, its API
  server often comes back unhealthy; recreate it:

      minikube delete
      minikube start --memory=10240 --cpus=8

  Then deploy:  ./install.ps1
"@
    }
}

function Get-ConsoleToken {
    $encoded = kubectl -n $Namespace get secret industrial-cognition-secrets `
        --request-timeout=15s -o jsonpath='{.data.AUTH_TOKEN}' 2>$null
    if ([string]::IsNullOrWhiteSpace($encoded)) {
        Fail "the deployment has no AUTH_TOKEN in secret '$Namespace/industrial-cognition-secrets'." @"
  Deploy first (./install.ps1). To see the console without auth, set AUTH_REQUIRED=0 on the
  api Deployment - but only on a throwaway cluster.
"@
    }
    # Secret data is base64; jsonpath returns it raw.
    try {
        return [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($encoded)).Trim()
    } catch {
        return $encoded.Trim()
    }
}

function Wait-ForEndpoints {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $announced = $false
    while ((Get-Date) -lt $deadline) {
        $endpoints = kubectl -n $Namespace get endpoints $Service `
            --request-timeout=10s -o jsonpath='{.subsets[*].addresses[*].ip}' 2>$null
        if (-not [string]::IsNullOrWhiteSpace($endpoints)) {
            return $true
        }

        if (-not $announced) {
            Write-Warn "no ready endpoints for service '$Service' yet; waiting..."
            # A short, honest diagnosis beats a silent wait: the usual cause on a laptop
            # is a Pending pod because the node is out of memory.
            $pods = kubectl -n $Namespace get pods --request-timeout=10s `
                -o custom-columns='NAME:.metadata.name,PHASE:.status.phase,NODE:.spec.nodeName,REASON:.status.containerStatuses[0].state.waiting.reason' 2>$null
            if ($pods) {
                $pods | Select-Object -First 12 | ForEach-Object { Write-Host "    $_" }
            }
            $pending = kubectl -n $Namespace get pods --field-selector=status.phase=Pending `
                --request-timeout=10s -o name 2>$null
            if ($pending) {
                Write-Host "    Pending pods usually mean the node is out of CPU or memory." -ForegroundColor Yellow
                Write-Host "    Check: kubectl -n $Namespace describe pod <name>  |  kubectl top node" -ForegroundColor Yellow
            }
            $announced = $true
        }
        Start-Sleep -Seconds 3
    }
    Fail "service '$Service' still has no ready endpoints after ${TimeoutSeconds}s."
}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
Write-Host ""
Write-Host "Industrial Cognition - operator console" -ForegroundColor White
Write-Host ""

Test-Cluster
$token = Get-ConsoleToken
$url = "http://127.0.0.1:$Port/auth/login?token=$([uri]::EscapeDataString($token))"
$console = "http://127.0.0.1:$Port/ui/"

if ($PrintOnly) {
    Write-Output $url
    exit 0
}

Wait-ForEndpoints

Write-Step "forwarding $Namespace/$Service to 127.0.0.1:$Port"
$logOut = Join-Path ([IO.Path]::GetTempPath()) "cogni-portforward.out"
$logErr = Join-Path ([IO.Path]::GetTempPath()) "cogni-portforward.err"
$forward = Start-Process -FilePath "kubectl" `
    -ArgumentList @("-n", $Namespace, "port-forward", "svc/$Service", "${Port}:80") `
    -RedirectStandardOutput $logOut -RedirectStandardError $logErr `
    -NoNewWindow -PassThru

try {
    $ready = $false
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline) {
        if ($forward.HasExited) {
            $err = if (Test-Path $logErr) { (Get-Content $logErr -Raw) } else { "" }
            Fail "port-forward exited unexpectedly: $err"
        }
        try {
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" `
                -TimeoutSec 3 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                $ready = $true
                break
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $ready) {
        Fail "the API did not answer on 127.0.0.1:$Port within 60s."
    }

    Write-Host ""
    Write-Host "  Console is up." -ForegroundColor Green
    Write-Host ""
    Write-Host "    $url"
    Write-Host ""
    Write-Host "  The token in that URL is exchanged for a session cookie and you are"
    Write-Host "  redirected to $console . After that the browser stays signed in."
    Write-Host ""
    Write-Host "  Token: $token" -ForegroundColor DarkGray
    Write-Host "  Press Ctrl-C to stop the forward." -ForegroundColor DarkGray
    Write-Host ""

    if (-not $NoBrowser) {
        Start-Process $url | Out-Null
    }

    while (-not $forward.HasExited) {
        Start-Sleep -Seconds 1
    }
} finally {
    if ($forward -and -not $forward.HasExited) {
        Write-Step "stopping the forward"
        $forward.Kill()
    }
}
