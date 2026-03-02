"""Thread-safe event queues for the ingestion pipeline.

EventQueue     — FIFO queue with backpressure and throughput metrics.
                 Wraps queue.Queue (thread-safe) rather than collections.deque
                 (not thread-safe without an explicit lock).

Interview talking points:
- queue.Queue is thread-safe; collections.deque is NOT without a lock.
- push() returning False is explicit backpressure — equivalent to Kafka's
  max.block.ms producer config. The producer can alert or slow down.
- metrics() mirrors Prometheus gauges: queue_depth, peak_depth, drop_count.
- Kafka analogy: a single-partition topic with a fixed consumer-lag budget.
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

from .models import MatchEvent


class EventQueue:
    """Thread-safe FIFO queue for MatchEvent objects.

    Wraps queue.Queue to add:
    - max_size enforcement with explicit backpressure (push returns False if full)
    - throughput metrics: total_pushed, total_popped, peak_depth, drop_count

    Kafka analogy: a single-partition Kafka topic with a fixed consumer-lag
    budget (max_size). When the consumer lags, new events are rejected
    (backpressure) rather than silently blocked.

    Example:
        q = EventQueue(max_size=1000)
        accepted = q.push(event)   # False if full → backpressure
        ev = q.pop(timeout_ms=10)  # MatchEvent or None on timeout
        print(q.metrics())
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._q = queue.Queue(maxsize=max_size)
        self._total_pushed: int = 0
        self._total_popped: int = 0
        self._peak_depth: int = 0
        self._lock = threading.Lock()  # guards metric counters only; queue.Queue is self-safe

    def push(self, event: MatchEvent) -> bool:
        """Push an event. Returns False (backpressure signal) if the queue is full.

        In production: equivalent to a Kafka producer send() timing out at
        max.block.ms — the caller knows the consumer is lagging.
        """
        try:
            self._q.put_nowait(event)
            with self._lock:
                self._total_pushed += 1
                depth = self._q.qsize()
                if depth > self._peak_depth:
                    self._peak_depth = depth
            return True
        except queue.Full:
            return False

    def pop(self, timeout_ms: float = 10.0) -> Optional[MatchEvent]:
        """Pop an event. Returns None if the queue is empty after timeout_ms.

        In production: a Kafka consumer poll() with a configurable timeout.
        """
        try:
            event = self._q.get(timeout=timeout_ms / 1000.0)
            with self._lock:
                self._total_popped += 1
            return event
        except queue.Empty:
            return None

    def depth(self) -> int:
        """Current number of events waiting in the queue."""
        return self._q.qsize()

    def is_empty(self) -> bool:
        return self._q.empty()

    def metrics(self) -> dict:
        """Snapshot of queue health — expose as Prometheus gauges in production.

        Keys:
            total_pushed:  cumulative events accepted
            total_popped:  cumulative events consumed
            current_depth: events currently waiting
            peak_depth:    high-water mark since creation
            drop_count:    events rejected due to backpressure
        """
        with self._lock:
            return {
                "total_pushed":  self._total_pushed,
                "total_popped":  self._total_popped,
                "current_depth": self._q.qsize(),
                "peak_depth":    self._peak_depth,
                "drop_count":    self._total_pushed - self._total_popped - self._q.qsize(),
            }
