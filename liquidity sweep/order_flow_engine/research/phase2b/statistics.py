import numpy as np

def float_val(v):
    if v in (None, "", "None", "N/A", "MISSING"):
        return None
    try:
        return float(v)
    except ValueError:
        return None

def calculate_stats(returns: list, mfes: list = None, maes: list = None) -> dict:
    n = len(returns)
    if n == 0:
        return {
            "count": 0, "win_rate": 0.0, "avg_return": 0.0, "median_return": 0.0,
            "avg_winner": 0.0, "avg_loser": 0.0, "win_loss_ratio": 0.0,
            "expectancy": 0.0, "profit_factor": 0.0, "mfe_mean": 0.0, "mae_mean": 0.0
        }
        
    positives = [r for r in returns if r > 0]
    negatives = [r for r in returns if r < 0]
    
    win_rate = len(positives) / n
    avg_return = float(np.mean(returns))
    median_return = float(np.median(returns))
    
    avg_winner = float(np.mean(positives)) if positives else 0.0
    avg_loser = float(np.mean(negatives)) if negatives else 0.0
    
    win_loss_ratio = (avg_winner / abs(avg_loser)) if avg_loser != 0 else 0.0
    expectancy = (win_rate * avg_winner) - ((1 - win_rate) * abs(avg_loser))
    
    sum_pos = sum(positives)
    sum_neg = abs(sum(negatives))
    profit_factor = (sum_pos / sum_neg) if sum_neg != 0 else (float('inf') if sum_pos > 0 else 1.0)
    
    stats = {
        "count": n,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "median_return": median_return,
        "avg_winner": avg_winner,
        "avg_loser": avg_loser,
        "win_loss_ratio": win_loss_ratio,
        "expectancy": expectancy,
        "profit_factor": profit_factor
    }
    
    if mfes:
        stats["mfe_mean"] = float(np.mean(mfes))
        stats["mfe_median"] = float(np.median(mfes))
    else:
        stats["mfe_mean"] = 0.0
        stats["mfe_median"] = 0.0
        
    if maes:
        stats["mae_mean"] = float(np.mean(maes))
        stats["mae_median"] = float(np.median(maes))
    else:
        stats["mae_mean"] = 0.0
        stats["mae_median"] = 0.0
        
    return stats

def bootstrap_ci(returns: list, iterations=5000, seed=42) -> dict:
    n = len(returns)
    if n < 20:
        return {
            "status": "INSUFFICIENT_SAMPLE_FOR_BOOTSTRAP",
            "win_rate": None,
            "mean_return": None,
            "median_return": None,
            "expectancy": None
        }
        
    rng = np.random.default_rng(seed)
    win_rates, means, medians, expectancies = [], [], [], []
    
    for _ in range(iterations):
        sample = rng.choice(returns, size=n, replace=True)
        pos = [s for s in sample if s > 0]
        neg = [s for s in sample if s < 0]
        
        wr = len(pos) / n
        win_rates.append(wr)
        means.append(np.mean(sample))
        medians.append(np.median(sample))
        
        avg_w = np.mean(pos) if pos else 0.0
        avg_l = np.mean(neg) if neg else 0.0
        expectancies.append((wr * avg_w) - ((1 - wr) * abs(avg_l)))
        
    return {
        "status": "OK",
        "win_rate": (float(np.percentile(win_rates, 2.5)), float(np.percentile(win_rates, 97.5))),
        "mean_return": (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))),
        "median_return": (float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))),
        "expectancy": (float(np.percentile(expectancies, 2.5)), float(np.percentile(expectancies, 97.5)))
    }
