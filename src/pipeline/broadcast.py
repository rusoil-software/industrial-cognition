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
"""Fan-out of frame results to real-time consumers (WebSocket clients).

The property that matters: **a slow client must never slow the pipeline**. At 60
FPS a WebSocket peer on a congested network can fall behind by hundreds of
results a second. If publishing blocked, or if each subscriber buffered without
limit, one lagging dashboard would either stall detection or exhaust memory.

So each subscription owns a **bounded** buffer and a publish that would overflow
drops the *oldest* entry, counting the loss. That matches the pipeline's own
policy: a stale detection is worthless, a fresh one is worth more than two stale
ones. The dropped counter is exposed so "the client is behind" is visible rather
than silent.

Deliberately free of asyncio: the runner publishes from a worker thread and the
WebSocket handler consumes from the event loop, so a plain thread-safe queue is the
right primitive. The async adapter lives in `src/api/realtime.py`.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Optional

__all__ = ["ResultBroadcaster", "Subscription"]

logger = logging.getLogger(__name__)


@dataclass
class Subscription:
    """One consumer's bounded buffer of results."""

    max_queue: int = 120
    _queue: Deque[dict[str, Any]] = field(init=False, repr=False)
    dropped: int = 0
    delivered: int = 0
    closed: bool = False

    def __post_init__(self) -> None:
        self._queue = deque()

    def push(self, item: dict[str, Any]) -> None:
        """Append, dropping the oldest item when full. Never blocks."""
        if self.closed:
            return
        if len(self._queue) >= self.max_queue:
            self._queue.popleft()
            self.dropped += 1
        self._queue.append(item)

    def poll(self) -> Optional[dict[str, Any]]:
        """Return the oldest buffered item, or ``None``."""
        if not self._queue:
            return None
        self.delivered += 1
        return self._queue.popleft()

    @property
    def depth(self) -> int:
        return len(self._queue)

    def close(self) -> None:
        self.closed = True
        self._queue.clear()


class ResultBroadcaster:
    """Thread-safe fan-out of results to any number of subscribers."""

    def __init__(self, *, max_queue_per_subscriber: int = 120) -> None:
        if max_queue_per_subscriber < 1:
            raise ValueError("max_queue_per_subscriber must be >= 1")
        self._max_queue = max_queue_per_subscriber
        self._subscriptions: list[Subscription] = []
        self._lock = threading.Lock()
        self._published = 0

    def subscribe(self) -> Subscription:
        """Register a consumer; the caller must :meth:`unsubscribe` it."""
        subscription = Subscription(max_queue=self._max_queue)
        with self._lock:
            self._subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        with self._lock:
            if subscription in self._subscriptions:
                self._subscriptions.remove(subscription)
        subscription.close()

    def publish(self, result: dict[str, Any]) -> None:
        """Deliver *result* to every subscriber without blocking.

        Copies nothing and never raises: publishing happens on the capture thread,
        where an exception would kill the loop.
        """
        with self._lock:
            self._published += 1
            subscriptions = list(self._subscriptions)
        for subscription in subscriptions:
            subscription.push(result)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscriptions)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            subscriptions = list(self._subscriptions)
            published = self._published
        return {
            "subscribers": len(subscriptions),
            "published": published,
            "dropped_per_subscriber": [s.dropped for s in subscriptions],
            "depth_per_subscriber": [s.depth for s in subscriptions],
        }
