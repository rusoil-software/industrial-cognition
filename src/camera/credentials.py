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
# date   : 2026-Oct-20
# ==============================================================================
"""Encrypt camera passwords at rest.

IP cameras in a plant network are usually protected by an HTTP digest or RTSP basic-auth
credential. Storing that in clear text in PostgreSQL would put every camera password in every
database backup, so the registry keeps ciphertext and the key lives in the Secret.

What this does and does not claim:

* It protects the **database** (and its backups, and anybody who can read a row) from a leaked
  camera password. It does not protect against an attacker who can already read the application's
  environment, because that is where the key is. Envelope encryption with an external KMS - the
  Vault/OpenBao item in the tasklist - is what upgrades that claim.
* Fernet is authenticated symmetric encryption (AES-128-CBC + HMAC-SHA256), which is the right
  primitive here: a wrong or tampered ciphertext fails loudly instead of decrypting to garbage that
  would be tried as a password against a camera.
* The key is a separate secret from `JWT_SECRET`, on purpose. Rotating session signing must not
  make stored camera credentials unreadable, and vice versa.

If `CAMERA_CRED_KEY` is absent the vault reports itself unavailable rather than inventing a key:
the camera credential feature degrades (503 with a clear message) while everything else keeps
working, which is the same rule the pipeline follows for a missing model.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

__all__ = [
    "CAMERA_CRED_KEY_ENV",
    "CredentialError",
    "CredentialVault",
    "generate_key",
]

logger = logging.getLogger(__name__)

#: Environment variable holding the Fernet key (url-safe base64, 32 bytes decoded).
CAMERA_CRED_KEY_ENV = "CAMERA_CRED_KEY"


class CredentialError(RuntimeError):
    """The vault is unavailable, or the stored value cannot be decrypted."""


def generate_key() -> str:
    """A fresh Fernet key, for the installers and for `doc/camera.md`."""
    return Fernet.generate_key().decode("ascii")


class CredentialVault:
    """Encrypts and decrypts camera passwords with a key from the environment."""

    def __init__(self, key: Optional[str]) -> None:
        self._fernet: Optional[Fernet] = None
        if not key:
            return

        try:
            self._fernet = Fernet(key.encode("ascii"))
        except (ValueError, TypeError) as exc:
            # A malformed key is a deployment mistake, not a reason to store plaintext: refuse to
            # operate and let the caller report "credentials unavailable".
            logger.error("CAMERA_CRED_KEY is not a valid Fernet key: %s", exc)
            self._fernet = None

    @classmethod
    def from_environment(cls) -> "CredentialVault":
        return cls(os.getenv(CAMERA_CRED_KEY_ENV))

    @property
    def available(self) -> bool:
        return self._fernet is not None

    def encrypt(self, secret: str) -> str:
        """Return the ciphertext for *secret*.

        Raises:
            CredentialError: when the vault has no usable key.
        """
        if self._fernet is None:
            raise CredentialError(
                f"camera credentials are unavailable: set {CAMERA_CRED_KEY_ENV} "
                "(see doc/camera.md)"
            )
        return self._fernet.encrypt(secret.encode("utf-8")).decode("ascii")

    def decrypt(self, token: str) -> str:
        """Return the plaintext behind *token*.

        Raises:
            CredentialError: when the vault has no key, or the token was not produced by this key
                (rotated key, wrong database row, tampering).
        """
        if self._fernet is None:
            raise CredentialError(
                f"camera credentials are unavailable: set {CAMERA_CRED_KEY_ENV}"
            )
        try:
            return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")
        except (InvalidToken, ValueError) as exc:
            raise CredentialError(
                "stored camera credential cannot be decrypted with the current "
                f"{CAMERA_CRED_KEY_ENV}; the key was rotated or the row was tampered with"
            ) from exc


def encode_key_for_log(key: str) -> str:
    """A short, safe description of a key for diagnostics - never the key itself."""
    try:
        raw = base64.urlsafe_b64decode(key.encode("ascii"))
    except (ValueError, TypeError):
        return "<malformed>"
    return f"<{len(raw)}-byte Fernet key>"
