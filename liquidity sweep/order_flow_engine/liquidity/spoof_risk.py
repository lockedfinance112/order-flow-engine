class SpoofRiskDetector:
    """
    Identifies candidate spoofing behavior ('SPOOF_RISK') when exceptionally large
    resting orders are canceled (pulled) quickly as the price approaches,
    without executing any volume.
    """
    def __init__(self, min_spoof_usd: float = 300000.0, max_lifetime_seconds: float = 10.0):
        self.min_spoof_usd = min_spoof_usd
        self.max_lifetime_seconds = max_lifetime_seconds

    def is_spoof_attempt(self, wall_side: str, wall_lifetime_sec: float, pulled_notional: float, executed_notional: float) -> bool:
        """
        Flags True if a large wall was pulled quickly with little or no execution.
        """
        if pulled_notional < self.min_spoof_usd:
            return False
        
        # Canceled quickly as price approaches
        is_short_lived = wall_lifetime_sec <= self.max_lifetime_seconds
        has_low_execution = executed_notional < (pulled_notional * 0.05) # less than 5% execution

        return is_short_lived and has_low_execution
