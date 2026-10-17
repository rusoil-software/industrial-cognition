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

# Native commands write progress and deprecation notices to stderr, and under
# $ErrorActionPreference = 'Stop' PowerShell 5.1 turns *any* stderr line from a native
# program into a terminating NativeCommandError. That is exactly what killed this script:
# "Warning: v1 Endpoints is deprecated in v1.33+" was treated as a fatal error at the
# readiness check. Shadowing `kubectl` with a function that swallows stderr and
# re-publishes the exit code fixes it once, instead of at every call site, and keeps
# stdout parsing intact.
function kubectl {
    $app = Get-Command kubectl.exe -CommandType Application -ErrorAction SilentlyContinue
    if (-not $app) {
        $app = Get-Command kubectl -CommandType Application -ErrorAction SilentlyContinue
    }
    if (-not $app) { throw "kubectl is not on PATH" }

    $output = & $app.Source @args 2>$null
    $global:LASTEXITCODE = $LASTEXITCODE
    return $output
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

function Get-ReadyEndpointAddresses {
    <#
        Ready endpoint addresses for a Service, via EndpointSlices.

        JSON, not `-o jsonpath`, deliberately: kubectl's jsonpath needs a quoted newline
        ({end}{"\n"}), and PowerShell strips the inner double quotes when building a
        native command line, so kubectl receives a bare \n and fails with "unrecognized
        character in action: U+005C". Parsing JSON sidesteps Windows quoting entirely.

        EndpointSlices rather than v1 Endpoints because the latter is deprecated from
        v1.33 and prints a warning on every call; a slice also carries per-endpoint
        readiness, which is the question actually being asked.
    #>
    param([string]$Namespace, [string]$Service)

    $json = kubectl -n $Namespace get endpointslice `
        -l "kubernetes.io/service-name=$Service" --request-timeout=10s -o json 2>$null
    if ([string]::IsNullOrWhiteSpace($json)) { return @() }

    try {
        $parsed = $json | ConvertFrom-Json
    } catch {
        return @()
    }

    $addresses = @()
    foreach ($slice in $parsed.items) {
        foreach ($endpoint in $slice.endpoints) {
            # `conditions` may be absent on older API versions; treat that as ready.
            if ($null -eq $endpoint.conditions -or $endpoint.conditions.ready -ne $false) {
                $addresses += $endpoint.addresses
            }
        }
    }
    return $addresses
}

function Find-FreePort {
    <#
        First free local port at or after $Preferred.

        A previous run that died leaves an orphan `kubectl port-forward` holding 8080. The
        next run then starts kubectl, which fails to bind and either exits or (worse) stays
        alive while forwarding to a pod that no longer exists - the symptom was "the API
        did not answer within 60s" with a healthy cluster. Probing for a free port is
        cheaper and safer than hunting for stale processes to kill.
    #>
    param([int]$Preferred, [int]$Range = 20)

    for ($candidate = $Preferred; $candidate -lt ($Preferred + $Range); $candidate++) {
        $listener = $null
        try {
            $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $candidate)
            $listener.Start()
            return $candidate
        } catch {
            continue
        } finally {
            if ($listener) { $listener.Stop() }
        }
    }
    Fail "no free port in ${Preferred}..$($Preferred + $Range)"
}

function Wait-ForEndpoints {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $announced = $false
    while ((Get-Date) -lt $deadline) {
        if ((Get-ReadyEndpointAddresses -Namespace $Namespace -Service $Service).Count -gt 0) {
            return
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

if ($PrintOnly) {
    Write-Output "http://127.0.0.1:$Port/auth/login?token=$([uri]::EscapeDataString($token))"
    exit 0
}

Wait-ForEndpoints

# Bind a port that is actually free: a previous run that died leaves an orphan forward
# holding 8080, and reusing it produces a confusing "did not answer within 60s".
$requested = $Port
$Port = Find-FreePort -Preferred $Port
if ($Port -ne $requested) {
    Write-Warn "port $requested is busy; using $Port instead"
}
$url = "http://127.0.0.1:$Port/auth/login?token=$([uri]::EscapeDataString($token))"
$console = "http://127.0.0.1:$Port/ui/"

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
