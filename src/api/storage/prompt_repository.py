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
"""Persistence for per-camera detection prompts.

Stored in the existing `system_metadata` table under the key
``camera_prompt:<camera_id>``, with the human-readable prompt in ``value`` and the
full tokenised encoding in ``value_json``.

Why no new table: `system_metadata` exists precisely for system-level state whose
shape grows as the product does (`doc/database.md` §2). Adding a column per new
setting would mean a migration for every operational knob, and a per-camera prompt
is one more knob. The key prefix gives namespacing, and a `UNIQUE(key)` constraint
gives upsert semantics. If prompts later grow relationships (versions, audit
trail), a dedicated table plus a migration is the right next step - the Alembic
machinery from Stage 1 is in place for exactly that.

The repository is the only place that knows the storage shape; the API and the
dispatcher deal in `CameraPrompt` objects (`doc/conventions.md` §4: the repository
pattern, business logic never touches the engine directly).
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.storage.models import SystemMetadata
from src.vision.prompt import CameraPrompt, PromptEncodingError

__all__ = [
    "PROMPT_KEY_PREFIX",
    "delete_prompt",
    "list_prompts",
    "load_prompt",
    "prompt_key",
    "save_prompt",
    "validate_camera_id",
]

logger = logging.getLogger(__name__)

PROMPT_KEY_PREFIX = "camera_prompt:"

#: `system_metadata.key` is `varchar(128)`; subtract the prefix so a too-long id
#: is rejected here rather than raising a database error on insert.
MAX_CAMERA_ID_LENGTH = 128 - len(PROMPT_KEY_PREFIX)


def validate_camera_id(camera_id: str) -> str:
    """Normalise and validate a camera identifier.

    Raises:
        PromptEncodingError: for an empty or over-long id, or one containing
            whitespace (which would make the key ambiguous to read back).
    """
    if not camera_id or not camera_id.strip():
        raise PromptEncodingError("camera_id must not be empty")
    cleaned = camera_id.strip()
    if len(cleaned) > MAX_CAMERA_ID_LENGTH:
        raise PromptEncodingError(
            f"camera_id must be at most {MAX_CAMERA_ID_LENGTH} characters"
        )
    if any(character.isspace() for character in cleaned):
        raise PromptEncodingError("camera_id must not contain whitespace")
    return cleaned


def prompt_key(camera_id: str) -> str:
    """Storage key for one camera's prompt."""
    return f"{PROMPT_KEY_PREFIX}{validate_camera_id(camera_id)}"


def _camera_id_from_key(key: str) -> str:
    return key[len(PROMPT_KEY_PREFIX) :]


def save_prompt(session: Session, prompt: CameraPrompt) -> CameraPrompt:
    """Insert or update the prompt for ``prompt.camera_id``.

    Upsert via select-then-update rather than a dialect-specific
    ``INSERT ... ON CONFLICT``: the read path already exists, and this stays
    obvious to a reader. Concurrent writers are serialised by the unique
    constraint, and the last write wins - acceptable for a human-driven setting.
    """
    camera_id = validate_camera_id(prompt.camera_id)
    key = prompt_key(camera_id)

    row = session.scalar(select(SystemMetadata).where(SystemMetadata.key == key))
    payload = prompt.to_dict()

    if row is None:
        row = SystemMetadata(
            key=key,
            value=prompt.prompt_text,
            value_json=payload,
            last_heartbeat=None,
        )
        session.add(row)
    else:
        row.value = prompt.prompt_text
        row.value_json = payload

    session.flush()
    logger.info("stored prompt for camera %s: %r", camera_id, prompt.prompt_text)
    return prompt


def load_prompt(session: Session, camera_id: str) -> Optional[CameraPrompt]:
    """Return the stored prompt for *camera_id*, or ``None`` if unset."""
    key = prompt_key(camera_id)
    row = session.scalar(select(SystemMetadata).where(SystemMetadata.key == key))
    if row is None or row.value_json is None:
        return None
    prompt = CameraPrompt.from_dict(row.value_json)
    # Older/partial rows may lack the id; the key is authoritative.
    if not prompt.camera_id:
        prompt = CameraPrompt(
            labels=prompt.labels,
            prompt_text=prompt.prompt_text,
            input_ids=prompt.input_ids,
            attention_mask=prompt.attention_mask,
            tokenizer=prompt.tokenizer,
            max_length=prompt.max_length,
            camera_id=camera_id,
        )
    return prompt


def delete_prompt(session: Session, camera_id: str) -> bool:
    """Remove the prompt for *camera_id*; returns whether a row was deleted."""
    key = prompt_key(camera_id)
    row = session.scalar(select(SystemMetadata).where(SystemMetadata.key == key))
    if row is None:
        return False
    session.delete(row)
    session.flush()
    logger.info("deleted prompt for camera %s", camera_id)
    return True


def list_prompts(session: Session) -> list[CameraPrompt]:
    """Return every configured camera prompt, ordered by camera id."""
    rows = session.scalars(
        select(SystemMetadata)
        .where(SystemMetadata.key.like(f"{PROMPT_KEY_PREFIX}%"))
        .order_by(SystemMetadata.key)
    ).all()

    prompts: list[CameraPrompt] = []
    for row in rows:
        if row.value_json is None:
            continue
        try:
            prompts.append(load_prompt(session, _camera_id_from_key(row.key)) or None)  # type: ignore[arg-type]
        except PromptEncodingError:
            logger.warning("ignoring unreadable prompt row %s", row.key)
    return [prompt for prompt in prompts if prompt is not None]
