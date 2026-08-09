from typing import List, Tuple

class WallDetector:
    """
    Evaluates order book depth configurations and detects candidate walls.
    """
    def __init__(self, multiplier: float = 3.0, absolute_min_usd: float = 250000.0):
        self.multiplier = multiplier
        self.absolute_min_usd = absolute_min_usd

    def get_threshold(self, median_depth_usd: float) -> float:
        """Returns the USD threshold above which a book level is considered a wall candidate."""
        return max(self.absolute_min_usd, median_depth_usd * self.multiplier)

    def find_candidates(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], threshold: float) -> List[Tuple[float, float, str]]:
        """
        Scans bids and asks for any level exceeding the threshold.
        Returns: list of (price, quantity, side)
        """
        candidates = []
        for p, q in bids:
            if p * q >= threshold:
                candidates.append((p, q, "BID"))
        for p, q in asks:
            if p * q >= threshold:
                candidates.append((p, q, "ASK"))
        return candidates
