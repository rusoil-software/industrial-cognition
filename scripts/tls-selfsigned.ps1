#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Create the TLS Secret the ingress expects, from a local internal CA.

.DESCRIPTION
    The MVP runs on-premise with no route to the internet, so Let's Encrypt is unavailable and
    cert-manager would have to be mirrored. An internal CA needs neither: one machine with
    openssl, once per rotation, and `scripts/tls-selfsigned.sh` is the POSIX twin of this file.

    Two decisions worth stating:

    * A **CA of its own** rather than a self-signed server certificate. A self-signed leaf must be
      trusted individually and re-taught on every rotation; with a CA, operators import ca.crt
      once and renewals are invisible.
    * The trust anchor ships in its own Secret (`<name>-ca`) rather than mixed into the server
      Secret, so the object the ingress serves and the file a client imports are not confused.

    openssl is run inside a container because a default Windows box does not have it; Docker is
    already a prerequisite for the cluster.

.PARAMETER HostName
    DNS name the ingress serves. Must match the host in k8s/base/ingress.yaml.

.PARAMETER Days
    Validity in days (default 825 - the maximum current browsers accept for a leaf).

.PARAMETER Namespace
    Namespace of the ingress and Secret.

.PARAMETER Out
    Optional directory to receive the public certificates (ca.crt, tls.crt). The CA key never
    leaves the machine that generated it.

.EXAMPLE
    ./scripts/tls-selfsigned.ps1
    ./scripts/tls-selfsigned.ps1 -HostName cogni.local -Days 365 -Out .\tls
#>
param(
    [string]$HostName = "mvm.example.com",
    [int]$Days = 825,
    [string]$Namespace = "cogni-ns",
    [string]$SecretName = "industrial-cognition-tls",
    [string]$Out = ""
)

$ErrorActionPreference = "Stop"

function Step([string]$Message) { Write-Host "`n==> $Message" -ForegroundColor Cyan }
function Fail([string]$Message) {
    Write-Host "error: $Message" -ForegroundColor Red
    exit 1
}

if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) { Fail "kubectl is not on PATH" }
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Fail "docker is required to run openssl" }

$work = Join-Path ([IO.Path]::GetTempPath()) ("cogni-tls-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $work -Force | Out-Null

try {
    Step "generating an internal CA and a server certificate for $HostName ($Days days)"

    # SANs cover the ingress host plus the in-cluster service names, so the same certificate
    # works for `curl https://api` from a pod and for the browser.
    $extension = @"
basicConstraints = CA:FALSE
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = DNS:$HostName,DNS:*.$HostName,DNS:industrial-cognition,DNS:api,DNS:localhost,IP:127.0.0.1
"@
    Set-Content -Path (Join-Path $work "server.ext") -Value $extension -Encoding ascii

    $subject = "/CN=$HostName/O=RusoilSoftware"

    # One container per step, each verified by looking for the file it should have produced.
    # Chaining all four openssl commands in a single `sh -c` hid the failing step behind a
    # non-zero exit and empty stderr, which cost more time than three extra containers.
    function Invoke-OpenSsl {
        param([string]$StepName, [string]$Command, [string]$ExpectFile)

        docker run --rm -v "${work}:/work" -w /work alpine:3.20 `
            sh -c "apk add -q --no-cache openssl >/dev/null 2>&1; $Command"
        if ($LASTEXITCODE -ne 0) { Fail "$StepName failed (openssl exit $LASTEXITCODE)" }

        if ($ExpectFile -and -not (Test-Path (Join-Path $work $ExpectFile))) {
            Fail "$StepName produced no $ExpectFile"
        }
    }

    # Subjects are single-quoted and space-free on purpose: Windows strips inner double quotes
    # when building a native command line, so `-subj "/CN=Some CA"` reaches openssl as
    # `-subj /CN=Some CA` and fails with "Unknown attribute". Single quotes survive, and no
    # spaces means there is nothing for the shell to split even if they did not.
    Invoke-OpenSsl -StepName "CA generation" -ExpectFile "ca.crt" -Command (
        "openssl req -x509 -newkey rsa:4096 -nodes -days $Days -keyout ca.key -out ca.crt " +
        "-subj '/CN=IndustrialCognitionInternalCA/O=RusoilSoftware'"
    )

    Invoke-OpenSsl -StepName "server key and CSR" -ExpectFile "tls.csr" -Command (
        "openssl req -newkey rsa:2048 -nodes -keyout tls.key -out tls.csr -subj '$subject'"
    )

    Invoke-OpenSsl -StepName "signing" -ExpectFile "tls.crt" -Command (
        "openssl x509 -req -in tls.csr -CA ca.crt -CAkey ca.key -CAcreateserial " +
        "-days $Days -extfile server.ext -out tls.crt"
    )

    Step "creating Secret $Namespace/$SecretName"
    # `-o yaml | kubectl apply -f -` makes this idempotent: rotation is the same command again.
    kubectl -n $Namespace create secret tls $SecretName `
        --cert="$(Join-Path $work 'tls.crt')" --key="$(Join-Path $work 'tls.key')" `
        --dry-run=client -o yaml | kubectl apply -f -
    if ($LASTEXITCODE -ne 0) { Fail "could not create the TLS secret" }

    kubectl -n $Namespace create secret generic "$SecretName-ca" `
        --from-file="ca.crt=$(Join-Path $work 'ca.crt')" `
        --dry-run=client -o yaml | kubectl apply -f -
    if ($LASTEXITCODE -ne 0) { Fail "could not create the CA secret" }

    Step "certificate"
    docker run --rm -v "${work}:/work" -w /work alpine:3.20 sh -c `
        "apk add -q --no-cache openssl >/dev/null 2>&1; openssl x509 -in tls.crt -noout -subject -issuer -dates"

    if ($Out) {
        New-Item -ItemType Directory -Path $Out -Force | Out-Null
        Copy-Item (Join-Path $work "ca.crt") $Out -Force
        Copy-Item (Join-Path $work "tls.crt") $Out -Force
        Step "wrote public certificates to $Out (the CA key stays nowhere)"
    }

    Write-Host @"

  The ingress now has a certificate for $HostName, signed by the internal CA.

  Trust it on a client by importing the CA:
      kubectl -n $Namespace get secret $SecretName-ca ``
          -o jsonpath='{.data.ca\.crt}' | ForEach-Object { [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($_)) } > cogni-ca.crt

  Then, with $HostName resolving to the ingress address:
      curl --cacert cogni-ca.crt https://$HostName/health

  Without DNS, pin the name to the ingress IP:
      curl --cacert cogni-ca.crt --resolve ${HostName}:443:`$(minikube ip) https://$HostName/health

  Re-running this command rotates the certificate. Clients keep trusting it because the CA does
  not change - only re-issue the CA (and re-import it) if the CA key itself is compromised.
"@ -ForegroundColor DarkGray
}
finally {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}
