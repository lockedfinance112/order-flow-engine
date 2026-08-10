class ReplayClock:
    """Deterministic simulation clock derived strictly from replayed events."""
    def __init__(self):
        self._current_time_ms = 0

    def set_time(self, timestamp_ms: int):
        self._current_time_ms = timestamp_ms

    @property
    def now_ms(self) -> int:
        return self._current_time_ms

    @property
    def now_seconds(self) -> float:
        return self._current_time_ms / 1000.0
