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
"""How a prompt label becomes the stored `object_type`.

Pure, and separate from the repository, for two reasons: the rule is the part worth testing and
it needs no database, and until migration d5f8b2a1c9e3 this same rule was the bug - an enum
that could only hold four values turned every real prompt into ``clutter``.
"""

from __future__ import annotations

__all__ = ["UNKNOWN_OBJECT_TYPE", "normalise_object_type"]

#: Written when a detection carries no usable label at all. Deliberately not "clutter": a row
#: that admits it does not know is more useful than one that pretends to classify.
UNKNOWN_OBJECT_TYPE = "unknown"


def normalise_object_type(label: str) -> str:
    """Normalise a prompt label for storage.

    Lower-cased and whitespace-collapsed so "Metal  Sheet" and "metal sheet" group together in
    a query. The exact text an operator typed stays in the `label` column, so normalisation
    costs nothing in fidelity.

    An empty label becomes :data:`UNKNOWN_OBJECT_TYPE` rather than an empty string: the column
    is NOT NULL, and a blank value would be indistinguishable from a bug.
    """
    cleaned = " ".join((label or "").split()).lower()
    return cleaned or UNKNOWN_OBJECT_TYPE
