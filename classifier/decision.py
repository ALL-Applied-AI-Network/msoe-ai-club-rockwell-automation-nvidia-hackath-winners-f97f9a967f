"""Separate classification from acceptance; ambiguous vials require review."""
import math


def decide(classes, probabilities, min_conf=0.5, pass_min_conf=0.5):
    if not 0 <= min_conf <= 1 or not 0 <= pass_min_conf <= 1:
        raise ValueError("confidence thresholds must be in [0, 1]")
    if len(classes) != len(probabilities) or not probabilities:
        raise ValueError("classes and probabilities must be nonempty and match")
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in probabilities):
        raise ValueError("invalid probability")
    if not math.isclose(sum(probabilities), 1, abs_tol=1e-5):
        raise ValueError("probabilities must sum to one")
    k = max(range(len(classes)), key=lambda i: probabilities[i])
    raw, confidence = classes[k], probabilities[k]
    threshold = max(min_conf, pass_min_conf) if raw == "pass" else min_conf
    return (raw if confidence >= threshold else "uncertain"), raw, confidence
