from typing import List, Tuple

class LiquidityVacuumDetector:
    """
    Detects low cumulative depth in the direction the price is moving.
    Flags 'UPSIDE_VACUUM' or 'DOWNSIDE_VACUUM' if liquidity is thin.
    """
    def __init__(self, vacuum_threshold_pct: float = 0.25):
        self.vacuum_threshold_pct = vacuum_threshold_pct

    def check_vacuum(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], mid_price: float, median_depth_usd: float) -> Tuple[bool, bool]:
        """
        Returns (upside_vacuum_active, downside_vacuum_active).
        """
        if median_depth_usd <= 0.0 or mid_price <= 0.0:
            return False, False

        # Accumulate notional depth within 15 basis points (0.15%) of mid price
        up_limit = mid_price * 1.0015
        dn_limit = mid_price * 0.9985

        bid_depth_near = sum(p * q for p, q in bids if dn_limit <= p <= mid_price)
        ask_depth_near = sum(p * q for p, q in asks if mid_price <= p <= up_limit)

        # Check if near depth is significantly below median threshold
        upside_vacuum = ask_depth_near < (median_depth_usd * self.vacuum_threshold_pct)
        downside_vacuum = bid_depth_near < (median_depth_usd * self.vacuum_threshold_pct)

        return upside_vacuum, downside_vacuum
