class IcebergDetector:
    """
    Infers potential iceberg orders ('ICEBERG_SUSPECTED') when execution size
    significantly exceeds the displayed order book size at a specific price level.
    """
    def __init__(self, display_multiplier: float = 2.5):
        self.display_multiplier = display_multiplier
        # Tracks executed volume per price level: {price: executed_notional}
        self.executed_volume: dict[float, float] = {}

    def record_trade(self, price: float, notional: float):
        """Records trade execution at a price level."""
        self.executed_volume[price] = self.executed_volume.get(price, 0.0) + notional

    def check_iceberg(self, price: float, displayed_notional: float) -> bool:
        """
        Returns True if executed volume exceeds displayed size by display_multiplier.
        """
        if displayed_notional <= 0.0 or price not in self.executed_volume:
            return False
            
        executed = self.executed_volume[price]
        if executed > displayed_notional * self.display_multiplier:
            return True
        return False

    def clear(self):
        self.executed_volume.clear()
