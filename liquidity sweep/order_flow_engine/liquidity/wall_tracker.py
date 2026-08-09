import time
from typing import Dict, List, Optional

class PersistentWall:
    """
    Tracks a large resting order level at a specific price bucket over time.
    """
    def __init__(self, price: float, side: str, initial_notional: float, timestamp: float):
        self.price = price
        self.side = side  # "BID" or "ASK"
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.maximum_notional = initial_notional
        self.current_notional = initial_notional
        self.average_notional = initial_notional
        self.number_of_updates = 1
        
        self.executed_volume_at_level = 0.0
        self.quantity_added = 0.0
        self.quantity_removed = 0.0
        self.replenishment_count = 0

    def update(self, notional: float, timestamp: float):
        self.last_seen = timestamp
        if notional > self.maximum_notional:
            self.maximum_notional = notional
        
        # Track adds/removes
        diff = notional - self.current_notional
        if diff > 0:
            self.quantity_added += diff
            # If it was previously depleted and rises significantly again, count as replenishment
            if self.current_notional < self.maximum_notional * 0.2 and diff > self.maximum_notional * 0.3:
                self.replenishment_count += 1
        else:
            self.quantity_removed += abs(diff)

        self.current_notional = notional
        self.number_of_updates += 1
        self.average_notional = ((self.average_notional * (self.number_of_updates - 1)) + notional) / self.number_of_updates

    def get_lifetime_ms(self, current_time: float) -> float:
        return (current_time - self.first_seen) * 1000.0

    def get_score(self, mid_price: float, median_depth_usdt: float, current_time: float) -> float:
        """
        wall_score = size_score * persistence_score * proximity_score * stability_score
        """
        if self.current_notional <= 0.0 or median_depth_usdt <= 0.0 or mid_price <= 0.0:
            return 0.0

        # 1. Size Score (ratio to recent median depth)
        size_ratio = self.current_notional / median_depth_usdt
        size_score = min(size_ratio, 5.0)  # cap at 5x depth

        # 2. Persistence Score (based on age in seconds, logs up to 1.0 at 60s)
        age_seconds = current_time - self.first_seen
        persistence_score = min(age_seconds / 60.0, 1.0)

        # 3. Proximity Score (inverse of distance from mid price)
        distance_pct = abs(self.price - mid_price) / mid_price
        # Close to mid means proximity score is near 1.0. Fades to 0.0 at 2% distance.
        proximity_score = max(0.0, 1.0 - (distance_pct / 0.02))

        # 4. Stability Score (lower updates variance implies more stable resting wall)
        # Ratio of average to maximum notional
        stability_score = self.average_notional / self.maximum_notional if self.maximum_notional > 0 else 1.0

        return size_score * persistence_score * proximity_score * stability_score


class WallTracker:
    """
    Manages active persistent walls, updates their state, and purges dead walls.
    """
    def __init__(self, wall_threshold_mult: float = 3.0, min_wall_usd: float = 250000.0):
        self.wall_threshold_mult = wall_threshold_mult
        self.min_wall_usd = min_wall_usd
        self.active_walls: Dict[float, PersistentWall] = {}  # price -> PersistentWall
        self.archived_walls: List[dict] = []

    def update_walls(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], mid_price: float, median_depth_usdt: float, timestamp: float):
        """
        Identifies and updates walls from current bids/asks.
        bids/asks are list of (price, qty)
        """
        # Determine wall threshold
        threshold = max(self.min_wall_usd, median_depth_usdt * self.wall_threshold_mult)
        
        seen_prices = set()

        # Update bids
        for price, qty in bids:
            notional = price * qty
            if notional >= threshold:
                seen_prices.add(price)
                if price in self.active_walls:
                    self.active_walls[price].update(notional, timestamp)
                else:
                    self.active_walls[price] = PersistentWall(price, "BID", notional, timestamp)

        # Update asks
        for price, qty in asks:
            notional = price * qty
            if notional >= threshold:
                seen_prices.add(price)
                if price in self.active_walls:
                    self.active_walls[price].update(notional, timestamp)
                else:
                    self.active_walls[price] = PersistentWall(price, "ASK", notional, timestamp)

        # Archive/remove any walls that disappear from the book
        dead_prices = [p for p in self.active_walls if p not in seen_prices]
        for p in dead_prices:
            wall = self.active_walls.pop(p)
            self.archived_walls.append({
                "price": wall.price,
                "side": wall.side,
                "first_seen": wall.first_seen,
                "last_seen": timestamp,
                "max_notional": wall.maximum_notional,
                "average_notional": wall.average_notional,
                "lifetime_ms": wall.get_lifetime_ms(timestamp)
            })
            if len(self.archived_walls) > 500:
                self.archived_walls.pop(0)

    def get_active_walls(self, mid_price: float, median_depth_usdt: float) -> List[dict]:
        now = time.time()
        res = []
        for wall in self.active_walls.values():
            score = wall.get_score(mid_price, median_depth_usdt, now)
            res.append({
                "price": wall.price,
                "side": wall.side,
                "current_notional": wall.current_notional,
                "lifetime_ms": wall.get_lifetime_ms(now),
                "score": score,
                "maximum_notional": wall.maximum_notional,
                "replenishment_count": wall.replenishment_count
            })
        # Return sorted by score descending
        return sorted(res, key=lambda x: x["score"], reverse=True)
