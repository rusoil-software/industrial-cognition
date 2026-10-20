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
"""Authentication endpoints and the request guard.

The guard is a pure function - :func:`session_from_request` - so the cookie/header/query
precedence is testable without a running server, and the FastAPI wiring stays a thin
translation of :class:`~src.api.exceptions.AuthError` into 401.

Precedence, and why:

1. the `ic_session` cookie - what the browser sends for `<img>`, `WebSocket` and `fetch`;
2. `Authorization: Bearer` - for scripts and for a future CLI;
3. `?token=` - the Jupyter-style shortcut, so a link with the deploy token opens the
   console and is immediately exchanged for a cookie.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Cookie, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel

import os

from src.api import auth
from src.api.exceptions import AuthError
from src.api.throttle import AttemptThrottle

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

#: Process-local failure counter. With several API replicas the effective limit multiplies by
#: the replica count - a real limitation, documented in src/api/throttle.py. It exists so a
#: guessed token cannot be tried at network speed, not to be a security boundary on its own.
_THROTTLE = AttemptThrottle(
    max_failures=int(os.getenv("AUTH_MAX_FAILURES", "5")),
    block_seconds=float(os.getenv("AUTH_LOCKOUT_SECONDS", "300")),
)


def _client_key(request: Request) -> str:
    """Identify the caller for throttling.

    The address, not the submitted token: keying on the token would let an attacker rotate
    guesses freely, which is exactly the attack the throttle is for.
    """
    return request.client.host if request.client else "unknown"


def _guard_throttle(request: Request) -> None:
    """Raise 429 when this caller is locked out."""
    retry_after = _THROTTLE.retry_after(_client_key(request))
    if retry_after > 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"too many failed attempts; retry in {int(retry_after)}s",
            headers={"Retry-After": str(int(retry_after))},
        )


class LoginRequest(BaseModel):
    """The shared deployment token."""

    token: str


def _set_session(response: Response, settings: auth.AuthSettings) -> str:
    token = auth.encode_token(settings)
    response.set_cookie(
        auth.COOKIE_NAME,
        token,
        max_age=settings.ttl_seconds,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        path="/",
    )
    return token


@router.post("/login", summary="Exchange the deployment token for a session")
def login(request: Request, payload: LoginRequest, response: Response) -> dict:
    _guard_throttle(request)
    settings = auth.get_settings()
    if not settings.login_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="this deployment has no AUTH_TOKEN configured",
        )
    if not auth.matches_login_token(settings, payload.token):
        # No detail about which part was wrong; a wrong token is simply a wrong token.
        # Counted against the caller, and a lockout is reported immediately rather than
        # on the next attempt so the client stops retrying.
        locked = _THROTTLE.record_failure(_client_key(request))
        if locked:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"too many failed attempts; retry in {int(locked)}s",
                headers={"Retry-After": str(int(locked))},
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token"
        )

    _THROTTLE.record_success(_client_key(request))
    _set_session(response, settings)
    logger.info("operator session issued")
    return {"authenticated": True, "expires_in": settings.ttl_seconds}


@router.get("/login", summary="Jupyter-style login link: ?token=... then redirect to /ui/")
def login_link(
    request: Request,
    token: str = Query(..., description="the deployment token"),
) -> Response:
    # The token-in-URL shortcut is throttled too: it is the same guess, just typed in a link,
    # and leaving it open would make the POST limit decorative.
    _guard_throttle(request)
    settings = auth.get_settings()
    if not settings.login_token or not auth.matches_login_token(settings, token):
        locked = _THROTTLE.record_failure(_client_key(request))
        raise HTTPException(
            status_code=(
                status.HTTP_429_TOO_MANY_REQUESTS if locked else status.HTTP_401_UNAUTHORIZED
            ),
            detail=(
                f"too many failed attempts; retry in {int(locked)}s"
                if locked
                else "invalid token"
            ),
        )

    _THROTTLE.record_success(_client_key(request))
    response = RedirectResponse(url="/ui/", status_code=status.HTTP_303_SEE_OTHER)
    _set_session(response, settings)
    return response


@router.delete("/login", summary="End the session")
def logout() -> Response:
    response = JSONResponse({"authenticated": False})
    response.delete_cookie(auth.COOKIE_NAME, path="/")
    return response


@router.get("/status", summary="Is a session required, and is this one valid?")
def auth_status(request: Request) -> dict:
    settings = auth.get_settings()
    try:
        claims = session_from_request(request, settings)
    except AuthError:
        claims = None
    return {
        "required": settings.enabled,
        "authenticated": claims is not None,
        "subject": (claims or {}).get("sub"),
    }


def session_from_request(request: Request, settings: auth.AuthSettings) -> dict:
    """Resolve the session from a request, or raise :class:`AuthError`.

    Raises:
        AuthError: when no credential is present or none of them verify.
    """
    if not settings.enabled:
        return {"sub": "anonymous", "auth_disabled": True}

    cookie = request.cookies.get(auth.COOKIE_NAME)
    if cookie:
        return dict(auth.decode_token(settings, cookie))

    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return dict(auth.decode_token(settings, header[7:].strip()))

    query_token = request.query_params.get("token")
    if query_token:
        if auth.matches_login_token(settings, query_token):
            return {"sub": "operator"}
        return dict(auth.decode_token(settings, query_token))

    raise AuthError("no session")


def guard(request: Request) -> None:
    """Raise 401 unless the request carries a valid session.

    Used as an HTTP middleware in `main.py`; the same function is called by the
    WebSocket handler, which cannot go through HTTP middleware.
    """
    settings = auth.get_settings()
    try:
        session_from_request(request, settings)
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
