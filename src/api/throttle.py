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
"""Throttle repeated failed logins.

The console is protected by one shared deployment token. Without a throttle that token can be
guessed at network speed: a 32-character random token is not brute-forceable, but a human-chosen
one - and every deployment that copies the example - is. The throttle turns "unlimited attempts"
into "five per minute and then a five-minute pause", which is enough to make guessing pointless
without getting in the operator's way after one typo.

Deliberately simple and **in-process**:

* a sliding window of failures per key (client address) and a lockout on top of it;
* success clears the window, so a legitimate user who mistypes once is unaffected;
* state is per process. With several API replicas each has its own counters, so the effective
  limit multiplies by the replica count. That is a real limitation, stated rather than hidden;
  moving it to Redis is the next step if the console is ever exposed beyond a trusted network.

`clock` is injectable so the tests do not sleep.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List

__all__ = ["AttemptThrottle"]

DEFAULT_MAX_FAILURES = 5
DEFAULT_WINDOW_SECONDS = 60.0
DEFAULT_BLOCK_SECONDS = 300.0


@dataclass
class AttemptThrottle:
    """Sliding-window failure counter with a lockout."""

    max_failures: int = DEFAULT_MAX_FAILURES
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    block_seconds: float = DEFAULT_BLOCK_SECONDS
    clock: Callable[[], float] = time.monotonic

    _failures: Dict[str, List[float]] = field(default_factory=dict, repr=False)
    _blocked_until: Dict[str, float] = field(default_factory=dict, repr=False)

    def retry_after(self, key: str) -> float:
        """Seconds until *key* may try again; ``0`` when it may try now.

        Expired state is dropped here rather than by a timer, so a long-idle process does not
        accumulate keys for addresses that stopped calling.
        """
        now = self.clock()

        blocked_until = self._blocked_until.get(key)
        if blocked_until is not None:
            if blocked_until > now:
                return blocked_until - now
            # The lockout has expired: forget it and start with a clean window.
            self._blocked_until.pop(key, None)
            self._failures.pop(key, None)
            return 0.0

        failures = self._failures.get(key)
        if failures:
            cutoff = now - self.window_seconds
            failures = [moment for moment in failures if moment >= cutoff]
            if failures:
                self._failures[key] = failures
            else:
                self._failures.pop(key, None)

        return 0.0

    def allow(self, key: str) -> bool:
        """Whether a login attempt from *key* may proceed."""
        return self.retry_after(key) == 0.0

    def record_failure(self, key: str) -> float:
        """Record a failed attempt; returns the lockout in seconds (0 if not locked)."""
        now = self.clock()
        failures = [moment for moment in self._failures.get(key, []) if moment >= now - self.window_seconds]
        failures.append(now)

        if len(failures) >= self.max_failures:
            self._blocked_until[key] = now + self.block_seconds
            self._failures.pop(key, None)
            return self.block_seconds

        self._failures[key] = failures
        return 0.0

    def record_success(self, key: str) -> None:
        """A correct token clears the window - one typo must not count against a user."""
        self._failures.pop(key, None)
        self._blocked_until.pop(key, None)

    def snapshot(self) -> dict:
        return {
            "max_failures": self.max_failures,
            "window_seconds": self.window_seconds,
            "block_seconds": self.block_seconds,
            "tracked_keys": len(set(self._failures) | set(self._blocked_until)),
        }
