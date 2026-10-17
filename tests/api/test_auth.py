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
# ==============================================================================
"""Tests for the token-based access layer.

`pyjwt` is imported lazily by `src.api.auth`, so these tests skip cleanly where it is not
installed (the repository's local test environment among them) and run in CI and in the
API image, where it is a pinned dependency.

The cases are the ones that matter for a door: right token, wrong token, no token,
expired token, tampered signature, algorithm substitution, and the guard's decision about
which paths are public.
"""

from __future__ import annotations

import pytest

jwt = pytest.importorskip("jwt")

from src.api.auth import (  # noqa: E402
    AuthSettings,
    decode_token,
    encode_token,
    is_protected,
    matches_login_token,
)
from src.api.exceptions import AuthError  # noqa: E402


def settings(**overrides) -> AuthSettings:
    base = dict(login_token="deploy-secret", jwt_secret="signing-key", ttl_seconds=3600)
    base.update(overrides)
    return AuthSettings(**base)


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
def test_a_token_round_trips() -> None:
    token = encode_token(settings())
    claims = decode_token(settings(), token)

    assert claims["sub"] == "operator"
    assert claims["exp"] > claims["iat"]


def test_a_token_signed_with_another_key_is_rejected() -> None:
    token = encode_token(settings(jwt_secret="attacker-key"))
    with pytest.raises(AuthError, match="invalid token"):
        decode_token(settings(), token)


def test_a_tampered_signature_is_rejected() -> None:
    token = encode_token(settings())
    head, claims, _signature = token.split(".")
    with pytest.raises(AuthError, match="invalid token"):
        decode_token(settings(), f"{head}.{claims}.deadbeef")


def test_an_expired_token_is_rejected() -> None:
    past = 1_000_000
    token = encode_token(settings(ttl_seconds=60), issued_at=past)
    with pytest.raises(AuthError, match="expired"):
        decode_token(settings(), token)


def test_the_none_algorithm_is_rejected() -> None:
    """`alg: none` is the classic JWT attack; PyJWT's allow-list must refuse it."""
    import base64
    import json

    def segment(payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    forged = f"{segment({'alg': 'none', 'typ': 'JWT'})}.{segment({'sub': 'attacker', 'exp': 9999999999})}."

    with pytest.raises(AuthError):
        decode_token(settings(), forged)


def test_minting_a_token_without_a_secret_is_refused() -> None:
    with pytest.raises(AuthError, match="refusing"):
        encode_token(settings(jwt_secret=""))


def test_decoding_without_a_secret_is_refused() -> None:
    token = encode_token(settings())
    with pytest.raises(AuthError, match="signing secret"):
        decode_token(settings(jwt_secret=""), token)


# --------------------------------------------------------------------------- #
# The shared login token
# --------------------------------------------------------------------------- #
def test_the_login_token_is_compared_exactly() -> None:
    assert matches_login_token(settings(), "deploy-secret") is True
    assert matches_login_token(settings(), "deploy-secreT") is False
    assert matches_login_token(settings(), "") is False


def test_an_unset_login_token_never_matches() -> None:
    """An empty deployment token must not become "any password works"."""
    assert matches_login_token(settings(login_token=""), "anything") is False
    assert matches_login_token(settings(login_token=""), "") is False


# --------------------------------------------------------------------------- #
# Which paths need a session
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "path",
    ["/health", "/auth/login", "/auth/status", "/ui", "/ui/", "/ui/app.js", "/"],
)
def test_public_paths_stay_public(path: str) -> None:
    assert is_protected(path) is False


@pytest.mark.parametrize(
    "path",
    [
        "/cameras",
        "/cameras/main/prompt",
        "/cameras/main/stream",
        "/cameras/main/snapshot",
        "/status",
        "/ws/detections",
        "/metrics",
    ],
)
def test_data_paths_require_a_session(path: str) -> None:
    assert is_protected(path) is True


def test_an_unknown_path_is_not_protected() -> None:
    """Unknown paths 404 anyway; the guard protects what exists."""
    assert is_protected("/nothing-here") is False
