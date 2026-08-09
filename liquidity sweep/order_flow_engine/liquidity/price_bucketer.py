class PriceBucketer:
    """
    Groups raw price levels into discrete bins/buckets based on a configurable size.
    """
    def __init__(self, bucket_size: float = 1.0):
        self.bucket_size = bucket_size

    def bucket_price(self, price: float) -> float:
        """Returns the bucket price for a given price."""
        if self.bucket_size <= 0.0:
            return price
        return round(price / self.bucket_size) * self.bucket_size
