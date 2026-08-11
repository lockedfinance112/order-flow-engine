from dataclasses import asdict, dataclass

from .models import canonical_hash


@dataclass(frozen=True)
class LiquidityClassificationPolicy:
    confirmation_window_ms: int = 60_000
    confirmation_hold_ms: int = 3_000
    minimum_confirming_trades: int = 3
    reclaim_buffer_bps: float = 1.0
    acceptance_buffer_bps: float = 1.0
    reorder_tolerance_ms: int = 2_000
    collision_window_ms: int = 2_000
    collision_level_tolerance_bps: float = 0.5
    canonical_price_decimal_places: int = 8
    maximum_supported_detection_lag_ms: int = 90_000
    buffer_safety_margin_ms: int = 28_000
    market_buffer_retention_ms: int = 180_000
    max_active_events_per_symbol: int = 128
    recent_final_events_per_symbol: int = 1_000
    recorder_queue_max_items: int = 10_000
    model_version: str = "1C.1-v1"

    def validate(self) -> "LiquidityClassificationPolicy":
        required = (
            self.confirmation_window_ms
            + self.reorder_tolerance_ms
            + self.maximum_supported_detection_lag_ms
            + self.buffer_safety_margin_ms
        )
        if self.market_buffer_retention_ms < required:
            raise ValueError("market buffer retention is below the frozen minimum")
        if self.minimum_confirming_trades < 1 or self.confirmation_hold_ms < 0:
            raise ValueError("confirmation constraints must be non-negative")
        return self

    @property
    def policy_hash(self) -> str:
        return canonical_hash(asdict(self))
