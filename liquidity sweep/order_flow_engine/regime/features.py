import math
from typing import List, Dict, Any, Optional
from regime.models import MarketBar

def compute_ema(prices: List[float], period: int) -> List[Optional[float]]:
    if not prices:
        return []
    ema = [None] * len(prices)
    if len(prices) < period:
        return ema
    
    # First EMA value is SMA
    sma = sum(prices[:period]) / period
    ema[period - 1] = sma
    multiplier = 2.0 / (period + 1)
    
    for i in range(period, len(prices)):
        prev = ema[i - 1]
        if prev is not None:
            ema[i] = (prices[i] - prev) * multiplier + prev
    return ema

def compute_atr(bars: List[MarketBar], period: int = 14) -> List[Optional[float]]:
    if not bars:
        return []
    atr = [None] * len(bars)
    if len(bars) < period:
        return atr
        
    tr = []
    for i in range(len(bars)):
        high = bars[i].high
        low = bars[i].low
        if i == 0:
            tr.append(high - low)
        else:
            prev_close = bars[i - 1].close
            tr.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
            
    # Wilder smoothing: initial is average, then Wilder smoothed
    initial_atr = sum(tr[:period]) / period
    atr[period - 1] = initial_atr
    
    for i in range(period, len(bars)):
        prev = atr[i - 1]
        if prev is not None:
            atr[i] = (prev * (period - 1) + tr[i]) / period
    return atr

def compute_adx(bars: List[MarketBar], period: int = 14) -> List[Optional[float]]:
    n = len(bars)
    adx = [None] * n
    if n < 2 * period:
        return adx
        
    tr = [0.0] * n
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    
    for i in range(1, n):
        h_diff = bars[i].high - bars[i - 1].high
        l_diff = bars[i - 1].low - bars[i].low
        
        if h_diff > l_diff and h_diff > 0:
            plus_dm[i] = h_diff
        if l_diff > h_diff and l_diff > 0:
            minus_dm[i] = l_diff
            
        high = bars[i].high
        low = bars[i].low
        prev_close = bars[i - 1].close
        tr[i] = max(high - low, abs(high - prev_close), abs(low - prev_close))
        
    str_val = [0.0] * n
    splus_dm = [0.0] * n
    sminus_dm = [0.0] * n
    
    str_val[period] = sum(tr[1:period + 1])
    splus_dm[period] = sum(plus_dm[1:period + 1])
    sminus_dm[period] = sum(minus_dm[1:period + 1])
    
    for i in range(period + 1, n):
        str_val[i] = str_val[i - 1] - (str_val[i - 1] / period) + tr[i]
        splus_dm[i] = splus_dm[i - 1] - (splus_dm[i - 1] / period) + plus_dm[i]
        sminus_dm[i] = sminus_dm[i - 1] - (sminus_dm[i - 1] / period) + minus_dm[i]
        
    dx = [0.0] * n
    for i in range(period, n):
        tr_i = str_val[i]
        if tr_i == 0.0:
            plus_di = 0.0
            minus_di = 0.0
        else:
            plus_di = (splus_dm[i] / tr_i) * 100.0
            minus_di = (sminus_dm[i] / tr_i) * 100.0
            
        sum_di = plus_di + minus_di
        diff_di = abs(plus_di - minus_di)
        dx[i] = (diff_di / sum_di * 100.0) if sum_di != 0.0 else 0.0
        
    adx_start = 2 * period - 1
    if n >= adx_start + 1:
        adx[adx_start] = sum(dx[period:adx_start + 1]) / period
        for i in range(adx_start + 1, n):
            prev = adx[i - 1]
            if prev is not None:
                adx[i] = (prev * (period - 1) + dx[i]) / period
    return adx

def compute_features(bars: List[MarketBar], timeframe: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    n = len(bars)
    if n == 0:
        return []
        
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [b.base_volume for b in bars]
    
    ema20 = compute_ema(closes, 20)
    ema50 = compute_ema(closes, 50)
    atr14 = compute_atr(bars, 14)
    adx14 = compute_adx(bars, 14)
    
    features_list = []
    
    for i in range(n):
        c_val = closes[i]
        h_val = highs[i]
        l_val = lows[i]
        vol_val = volumes[i]
        
        # 1. EMA gap in bps
        ema_gap_bps = None
        if ema20[i] is not None and ema50[i] is not None and ema50[i] != 0.0:
            ema_gap_bps = ((ema20[i] - ema50[i]) / ema50[i]) * 10000.0
            
        # 2. EMA slope (in ATR units per bar)
        ema_slope_atr = None
        if i >= 1 and ema20[i] is not None and ema20[i - 1] is not None and atr14[i] is not None and atr14[i] > 0.0:
            ema_slope_atr = (ema20[i] - ema20[i - 1]) / atr14[i]
            
        # 3. ATR as bps of price
        atr_bps = None
        if atr14[i] is not None and c_val != 0.0:
            atr_bps = (atr14[i] / c_val) * 10000.0
            
        # 4. Kaufman Efficiency Ratio 20
        er20 = None
        if i >= 20:
            net_change = abs(closes[i] - closes[i - 20])
            sum_changes = sum(abs(closes[j] - closes[j - 1]) for j in range(i - 19, i + 1))
            er20 = (net_change / sum_changes) if sum_changes != 0.0 else 0.0
            
        # 5. Realized Volatility 20 (Std of log returns)
        realized_vol20 = None
        if i >= 20:
            log_returns = []
            for j in range(i - 19, i + 1):
                if closes[j - 1] != 0.0:
                    log_returns.append(math.log(closes[j] / closes[j - 1]))
            if len(log_returns) > 1:
                mean_lr = sum(log_returns) / len(log_returns)
                var_lr = sum((x - mean_lr) ** 2 for x in log_returns) / (len(log_returns) - 1)
                realized_vol20 = math.sqrt(var_lr)
            else:
                realized_vol20 = 0.0
                
        # 6. Donchian Channel 20 (excludes current bar i for breakout reference)
        donchian_high20 = None
        donchian_low20 = None
        donchian_pos = None
        breakout_dist = None
        
        if i >= 21:
            donchian_high20 = max(highs[i - 20:i])
            donchian_low20 = min(lows[i - 20:i])
            donchian_range = donchian_high20 - donchian_low20
            
            if donchian_range != 0.0:
                donchian_pos = (c_val - donchian_low20) / donchian_range
            else:
                donchian_pos = 0.5
                
            atr_val = atr14[i] if atr14[i] is not None and atr14[i] > 0.0 else 1.0
            if c_val > donchian_high20:
                breakout_dist = (c_val - donchian_high20) / atr_val
            elif c_val < donchian_low20:
                breakout_dist = (c_val - donchian_low20) / atr_val
            else:
                breakout_dist = 0.0
                
        # 7. Rolling range width in bps
        rolling_range_width_bps = None
        if i >= 20:
            window_high = max(highs[i - 19:i + 1])
            window_low = min(lows[i - 19:i + 1])
            if window_low != 0.0:
                rolling_range_width_bps = ((window_high - window_low) / window_low) * 10000.0
                
        # 8. Volume moving average & z-score against previous 20 bars
        vol_ma = None
        vol_zscore = None
        if i >= 20:
            prev_volumes = volumes[i - 20:i]
            mean_vol = sum(prev_volumes) / 20.0
            vol_ma = mean_vol
            var_vol = sum((v - mean_vol) ** 2 for v in prev_volumes) / 19.0
            std_vol = math.sqrt(var_vol)
            if std_vol > 0.0:
                vol_zscore = (vol_val - mean_vol) / std_vol
            else:
                vol_zscore = 0.0
                
        # 9. Return over 1, 5, 20 bars
        ret_1 = None
        ret_5 = None
        ret_20 = None
        if i >= 1 and closes[i - 1] != 0.0:
            ret_1 = (c_val - closes[i - 1]) / closes[i - 1]
        if i >= 5 and closes[i - 5] != 0.0:
            ret_5 = (c_val - closes[i - 5]) / closes[i - 5]
        if i >= 20 and closes[i - 20] != 0.0:
            ret_20 = (c_val - closes[i - 20]) / closes[i - 20]
            
        feat = {
            "ema20": ema20[i],
            "ema50": ema50[i],
            "ema_gap_bps": ema_gap_bps,
            "ema_slope_atr": ema_slope_atr,
            "atr14": atr14[i],
            "atr_bps": atr_bps,
            "adx14": adx14[i],
            "er20": er20,
            "realized_vol20": realized_vol20,
            "donchian_high20": donchian_high20,
            "donchian_low20": donchian_low20,
            "donchian_pos": donchian_pos,
            "breakout_dist": breakout_dist,
            "rolling_range_width_bps": rolling_range_width_bps,
            "vol_ma": vol_ma,
            "vol_zscore": vol_zscore,
            "ret_1": ret_1,
            "ret_5": ret_5,
            "ret_20": ret_20
        }
        features_list.append(feat)
    return features_list
