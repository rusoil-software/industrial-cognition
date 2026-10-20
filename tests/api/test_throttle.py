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
"""Tests for the login throttle.

Injected clock, no sleeping, no HTTP: the behaviour under test is "how many wrong guesses, and
how long is the pause".
"""

from __future__ import annotations

from src.api.throttle import AttemptThrottle


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make(**overrides) -> tuple[AttemptThrottle, Clock]:
    clock = Clock()
    settings = {"max_failures": 3, "window_seconds": 60.0, "block_seconds": 300.0}
    settings.update(overrides)
    return AttemptThrottle(clock=clock, **settings), clock


def test_a_fresh_key_is_allowed() -> None:
    throttle, _ = make()
    assert throttle.allow("10.0.0.1") is True
    assert throttle.retry_after("10.0.0.1") == 0.0


def test_failures_below_the_limit_do_not_block() -> None:
    throttle, _ = make()
    throttle.record_failure("10.0.0.1")
    throttle.record_failure("10.0.0.1")

    assert throttle.allow("10.0.0.1") is True


def test_the_limit_blocks_for_the_configured_time() -> None:
    throttle, clock = make()
    for _ in range(3):
        throttle.record_failure("10.0.0.1")

    assert throttle.allow("10.0.0.1") is False
    assert throttle.retry_after("10.0.0.1") == 300.0

    clock.advance(299)
    assert throttle.allow("10.0.0.1") is False
    clock.advance(2)
    assert throttle.allow("10.0.0.1") is True, "the lockout must expire by itself"


def test_a_success_clears_the_window() -> None:
    """One typo must not count against a legitimate operator."""
    throttle, _ = make()
    throttle.record_failure("10.0.0.1")
    throttle.record_failure("10.0.0.1")
    throttle.record_success("10.0.0.1")

    for _ in range(3):
        throttle.record_failure("10.0.0.1")
    assert throttle.allow("10.0.0.1") is False

    throttle.record_success("10.0.0.1")
    assert throttle.allow("10.0.0.1") is True


def test_failures_age_out_of_the_window() -> None:
    throttle, clock = make()
    throttle.record_failure("10.0.0.1")
    throttle.record_failure("10.0.0.1")

    clock.advance(61)  # both failures are outside the 60 s window
    assert throttle.allow("10.0.0.1") is True

    throttle.record_failure("10.0.0.1")
    assert throttle.allow("10.0.0.1") is True, "the old failures must not count"


def test_keys_are_independent() -> None:
    """One noisy neighbour must not lock the operator out."""
    throttle, _ = make()
    for _ in range(3):
        throttle.record_failure("10.0.0.1")

    assert throttle.allow("10.0.0.1") is False
    assert throttle.allow("10.0.0.2") is True


def test_record_failure_reports_the_lockout_it_triggered() -> None:
    throttle, _ = make()
    assert throttle.record_failure("k") == 0.0
    assert throttle.record_failure("k") == 0.0
    assert throttle.record_failure("k") == 300.0


def test_snapshot_reports_the_configuration() -> None:
    throttle, _ = make()
    throttle.record_failure("a")
    throttle.record_failure("b")

    snapshot = throttle.snapshot()
    assert snapshot["max_failures"] == 3
    assert snapshot["tracked_keys"] == 2
