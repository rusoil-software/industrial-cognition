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
"""Credential vault tests.

The claim being tested is narrow and specific: a camera password stored in the database is
unreadable without the key, and a rotated key produces a clear error instead of a silent
"wrong password" against the camera.
"""

from __future__ import annotations

import pytest

cryptography = pytest.importorskip("cryptography")

from src.camera.credentials import (  # noqa: E402
    CredentialError,
    CredentialVault,
    encode_key_for_log,
    generate_key,
)


def test_a_generated_key_round_trips() -> None:
    vault = CredentialVault(generate_key())
    token = vault.encrypt("sup3r-s3cret")

    assert token != "sup3r-s3cret"
    assert vault.decrypt(token) == "sup3r-s3cret"


def test_the_ciphertext_does_not_contain_the_password() -> None:
    """A database dump must not be a password dump."""
    vault = CredentialVault(generate_key())
    token = vault.encrypt("EntranceGate2024")
    assert "EntranceGate2024" not in token


def test_encryption_is_not_deterministic() -> None:
    """Fernet includes an IV and a timestamp; identical passwords must not collide."""
    vault = CredentialVault(generate_key())
    assert vault.encrypt("same") != vault.encrypt("same")


def test_a_different_key_cannot_decrypt() -> None:
    """Key rotation without re-encrypting is a deployment error, and must be reported as one."""
    token = CredentialVault(generate_key()).encrypt("secret")

    with pytest.raises(CredentialError, match="cannot be decrypted"):
        CredentialVault(generate_key()).decrypt(token)


def test_tampered_ciphertext_is_rejected() -> None:
    vault = CredentialVault(generate_key())
    token = vault.encrypt("secret")
    tampered = token[:-4] + ("AAAA" if not token.endswith("AAAA") else "BBBB")

    with pytest.raises(CredentialError):
        vault.decrypt(tampered)


def test_no_key_means_unavailable_and_a_clear_error() -> None:
    vault = CredentialVault(None)

    assert vault.available is False
    with pytest.raises(CredentialError, match="CAMERA_CRED_KEY"):
        vault.encrypt("secret")
    with pytest.raises(CredentialError, match="CAMERA_CRED_KEY"):
        vault.decrypt("anything")


def test_a_malformed_key_is_unavailable_rather_than_ignored() -> None:
    """Storing plaintext because the key looked odd would be the worst possible fallback."""
    vault = CredentialVault("not-a-fernet-key")

    assert vault.available is False
    with pytest.raises(CredentialError):
        vault.encrypt("secret")


def test_the_environment_is_the_source_of_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    key = generate_key()
    monkeypatch.setenv("CAMERA_CRED_KEY", key)

    vault = CredentialVault.from_environment()
    assert vault.available is True
    assert vault.decrypt(vault.encrypt("x")) == "x"


def test_the_log_description_never_returns_the_key() -> None:
    key = generate_key()
    description = encode_key_for_log(key)

    assert key not in description
    assert "Fernet" in description


def test_unicode_passwords_survive() -> None:
    """Camera passwords are typed by people, in whatever script they use."""
    vault = CredentialVault(generate_key())
    assert vault.decrypt(vault.encrypt("пароль-01-Пароль")) == "пароль-01-Пароль"
