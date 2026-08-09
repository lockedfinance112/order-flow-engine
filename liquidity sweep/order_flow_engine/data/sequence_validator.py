class SequenceValidator:
    """
    Validates sequence numbers (U, u, pu) for Binance order book updates.
    Ensures that u is monotonically increasing and there are no sequence gaps.
    """
    def __init__(self):
        self.last_u = None

    def validate_and_update(self, U: int, u: int, pu: int) -> str:
        """
        Validates the sequence of the incoming event (U, u, pu).
        Returns one of:
          - "OK": Sequence is correct and u is updated.
          - "DUPLICATE": The event was already processed (u <= last_u).
          - "OUT_OF_ORDER": The event is out of order or older.
          - "GAP": There is a sequence gap (pu != last_u).
        """
        if self.last_u is None:
            # First update being applied (bootstrapped after snapshot)
            self.last_u = u
            return "OK"

        if u <= self.last_u:
            return "DUPLICATE"

        if pu != self.last_u:
            # Check if this might be out-of-order or a gap
            if pu < self.last_u:
                return "OUT_OF_ORDER"
            else:
                return "GAP"

        self.last_u = u
        return "OK"

    def reset(self):
        self.last_u = None
