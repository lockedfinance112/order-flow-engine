from typing import Dict

REGIME_PERMISSION_MATRIX: Dict[str, Dict[str, str]] = {
    "TREND_UP": {
        "long_momentum": "ALLOW",
        "short_momentum": "BLOCK",
        "breakout_long": "ALLOW",
        "breakout_short": "BLOCK",
        "mean_reversion": "REDUCE",
        "sweep_reversal": "REDUCE"
    },
    "TREND_DOWN": {
        "long_momentum": "BLOCK",
        "short_momentum": "ALLOW",
        "breakout_long": "BLOCK",
        "breakout_short": "ALLOW",
        "mean_reversion": "REDUCE",
        "sweep_reversal": "REDUCE"
    },
    "RANGE": {
        "long_momentum": "REDUCE",
        "short_momentum": "REDUCE",
        "breakout_long": "WATCH",
        "breakout_short": "WATCH",
        "mean_reversion": "ALLOW",
        "sweep_reversal": "ALLOW"
    },
    "BREAKOUT_UP": {
        "long_momentum": "ALLOW",
        "short_momentum": "BLOCK",
        "breakout_long": "ALLOW",
        "breakout_short": "BLOCK",
        "mean_reversion": "BLOCK",
        "sweep_reversal": "BLOCK"
    },
    "BREAKOUT_DOWN": {
        "long_momentum": "BLOCK",
        "short_momentum": "ALLOW",
        "breakout_long": "BLOCK",
        "breakout_short": "ALLOW",
        "mean_reversion": "BLOCK",
        "sweep_reversal": "BLOCK"
    },
    "TRANSITION": {
        "long_momentum": "REDUCE",
        "short_momentum": "REDUCE",
        "breakout_long": "REDUCE",
        "breakout_short": "REDUCE",
        "mean_reversion": "REDUCE",
        "sweep_reversal": "REDUCE"
    },
    "UNKNOWN": {
        "long_momentum": "BLOCK",
        "short_momentum": "BLOCK",
        "breakout_long": "BLOCK",
        "breakout_short": "BLOCK",
        "mean_reversion": "BLOCK",
        "sweep_reversal": "BLOCK"
    }
}

def permissions_for(regime: str) -> Dict[str, str]:
    return REGIME_PERMISSION_MATRIX.get(regime, REGIME_PERMISSION_MATRIX["UNKNOWN"])
