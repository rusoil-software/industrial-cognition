# Copyright (c), The Rusoil Software Development Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# author : Konstantin Ustiuzhanin
# date   : 2026-Oct-17
# ==============================================================================
"""Token-based access, in the style of Jupyter Lab.

One deployment secret (`AUTH_TOKEN`). The console asks for it once, exchanges it for a
short-lived **JWT** and keeps that in an HttpOnly cookie:

    GET    /ui/                  login form (static, carries no cell data)
    POST   /auth/login {token}   -> Set-Cookie: ic_session=<jwt>
    GET    /auth/login?token=... -> same, then redirect to /ui/   (Jupyter-style shortcut)
    DELETE /auth/login           -> clears the cookie
    GET    /auth/status          -> is a session required, and is this one valid?

Why a cookie rather than a bearer header: the browser must authenticate three things
that cannot all send headers - `<img src>` for the MJPEG preview, `WebSocket` (the
browser API cannot set headers), and `fetch`. A same-origin HttpOnly cookie is the only
mechanism all three carry automatically.

Signing is delegated to **PyJWT** (`pyjwt` in requirements/api.txt): the maintained
implementation, with an algorithm allow-list on decode (so `alg: none` and
algorithm-confusion are rejected structurally) and standard `exp`/`nbf` handling.

What this is *not*: multi-user identity, roles, or per-camera authorisation. It answers
"may this browser see the cell at all", which is the minimum before the console faces a
network.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from src.api.exceptions import AuthError

__all__ = [
    "ALGORITHM",
    "AuthSettings",
    "COOKIE_NAME",
    "PROTECTED_PREFIXES",
    "PUBLIC_PATHS",
    "decode_token",
    "encode_token",
    "get_settings",
    "is_protected",
    "matches_login_token",
]

logger = logging.getLogger(__name__)

COOKIE_NAME = "ic_session"

#: Reachable without a session. `/health` must stay open or Kubernetes probes fail;
#: `/ui` serves the login form; `/auth/*` is how a session is obtained.
PUBLIC_PATHS = frozenset(
    {"/health", "/auth/login", "/auth/status", "/docs", "/openapi.json", "/redoc"}
)

#: Everything under these prefixes needs a session. Listing *protected* prefixes rather
#: than exempting individual paths means a newly added endpoint is protected by default,
#: which is the safe direction for a mistake.
PROTECTED_PREFIXES = ("/cameras", "/status", "/ws", "/metrics")

#: Pinned algorithm, also passed to PyJWT as the decode allow-list.
ALGORITHM = "HS256"


def is_protected(path: str) -> bool:
    """Whether *path* requires a session.

    Static UI assets and the root document are public: without them a browser could not
    reach the login form. They carry no cell data - every piece of data comes from a
    protected endpoint.
    """
    if path in PUBLIC_PATHS:
        return False
    if path == "/" or path.startswith("/auth") or path.startswith("/ui"):
        return False
    return any(path.startswith(prefix) for prefix in PROTECTED_PREFIXES)


@dataclass(frozen=True, slots=True)
class AuthSettings:
    """Everything the auth layer needs, read once from the environment."""

    #: Shared secret the user types in to obtain a session.
    login_token: str
    #: JWT signing key. Falls back to the login token, which lives in the same Secret.
    jwt_secret: str
    ttl_seconds: int = 12 * 3600
    cookie_secure: bool = False
    required: bool = True

    @property
    def enabled(self) -> bool:
        """True when a session is required at all.

        `AUTH_REQUIRED=0` exists for local development only; the deployment never sets
        it, and the effective value is logged at startup so a deployment running open is
        visible rather than quietly unauthenticated.
        """
        return self.required


def get_settings() -> AuthSettings:
    token = os.getenv("AUTH_TOKEN", "").strip()
    secret = os.getenv("JWT_SECRET", "").strip() or token
    required = os.getenv("AUTH_REQUIRED", "1").strip().lower() not in ("0", "false", "no")
    return AuthSettings(
        login_token=token,
        jwt_secret=secret,
        ttl_seconds=int(os.getenv("AUTH_TTL_SECONDS", str(12 * 3600))),
        cookie_secure=os.getenv("AUTH_COOKIE_SECURE", "0").strip().lower()
        in ("1", "true", "yes"),
        required=required,
    )


def encode_token(
    settings: AuthSettings,
    *,
    subject: str = "operator",
    issued_at: Optional[int] = None,
) -> str:
    """Create a signed session token.

    Raises:
        AuthError: when no signing secret is configured. Refusing to mint an unsigned
            token is the point of this module.
    """
    if not settings.jwt_secret:
        raise AuthError(
            "no JWT_SECRET or AUTH_TOKEN is configured; refusing to issue a session. "
            "Set both in the deployment Secret."
        )

    import jwt

    now = int(time.time()) if issued_at is None else issued_at
    claims = {"sub": subject, "iat": now, "exp": now + settings.ttl_seconds}
    return jwt.encode(claims, settings.jwt_secret, algorithm=ALGORITHM)


def decode_token(settings: AuthSettings, token: str) -> Mapping[str, Any]:
    """Verify a session token and return its claims.

    Raises:
        AuthError: for a missing secret, a malformed or tampered token, a wrong
            algorithm, or an expired one. One exception type on purpose - telling an
            unauthenticated caller which part failed helps only an attacker.
    """
    if not settings.jwt_secret:
        raise AuthError("no signing secret is configured")

    import jwt

    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("invalid token") from exc

    return dict(claims)


def matches_login_token(settings: AuthSettings, candidate: str) -> bool:
    """Constant-time comparison against the shared login token."""
    if not settings.login_token:
        return False
    return secrets.compare_digest(candidate or "", settings.login_token)
