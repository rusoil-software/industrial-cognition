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
# date   : 2026-Apr-23
# ==============================================================================

from __future__ import annotations

__all__ = ["AuthError"]


class AuthError(Exception):
    """A session is missing, malformed, expired or signed with the wrong key.

    Deliberately one class, and deliberately without detail in its message: telling an
    unauthenticated caller *which* part of a token failed helps only an attacker. The
    router translates it into a bare 401.
    """
