from typing import List, Dict, Any, Tuple

class ConfusionMatrixCalculator:
    """Computes confusion metrics and agreement rates against ex-post reference labels."""
    @staticmethod
    def map_prediction(pred: str) -> str:
        p = pred.upper()
        if p in ("TREND_UP", "BREAKOUT_UP"):
            return "UP_DIRECTIONAL"
        elif p in ("TREND_DOWN", "BREAKOUT_DOWN"):
            return "DOWN_DIRECTIONAL"
        elif p == "RANGE":
            return "RANGE"
        elif p == "TRANSITION":
            return "TRANSITION"
        return "UNKNOWN"

    @classmethod
    def calculate_matrix(
        cls,
        timeline: List[Dict[str, Any]],
        ref_labels: List[str]
    ) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Any]]:
        
        matrix = {}
        classes = ["UP_DIRECTIONAL", "DOWN_DIRECTIONAL", "RANGE", "TRANSITION", "UNKNOWN"]
        for c in classes:
            matrix[c] = {r: 0 for r in ["UP_DIRECTIONAL", "DOWN_DIRECTIONAL", "RANGE", "AMBIGUOUS"]}
            
        correct = 0
        total = 0
        
        for t, ref in zip(timeline, ref_labels):
            if ref == "UNLABELLED":
                continue
            pred = cls.map_prediction(t.get("primary_regime", "UNKNOWN"))
            matrix[pred][ref] = matrix[pred].get(ref, 0) + 1
            total += 1
            if pred == ref:
                correct += 1
                
        agreement = correct / total if total > 0 else 0.0
        
        # Calculate class level Precision, Recall, F1 for the main classes (UP, DOWN, RANGE)
        class_metrics = {}
        for c in ["UP_DIRECTIONAL", "DOWN_DIRECTIONAL", "RANGE"]:
            tp = matrix[c].get(c, 0)
            fp = sum(matrix[c].values()) - tp
            fn = sum(matrix[other].get(c, 0) for other in classes) - tp
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            
            class_metrics[c] = {
                "precision": prec,
                "recall": rec,
                "f1": f1
            }
            
        # Balanced agreement (average of recall across main classes)
        recalls = [class_metrics[c]["recall"] for c in ["UP_DIRECTIONAL", "DOWN_DIRECTIONAL", "RANGE"]]
        balanced_agreement = sum(recalls) / len(recalls) if recalls else 0.0
        
        summary = {
            "agreement_rate": agreement,
            "balanced_agreement": balanced_agreement,
            "class_metrics": class_metrics,
            "total_samples": total
        }
        return matrix, summary
