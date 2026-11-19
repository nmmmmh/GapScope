"""Command-line interfaces for the public GapScope code package."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _quantile(probability: float, weights, log_mu, log_sigma) -> float:
    def cdf(value: float) -> float:
        z = math.log(value)
        return sum(
            weight * 0.5 * (1.0 + math.erf((z - mu) / (sigma * math.sqrt(2.0))))
            for weight, mu, sigma in zip(weights, log_mu, log_sigma)
        )

    low, high = 1e-12, max(1.0, sum(
        weight * math.exp(mu + 0.5 * sigma * sigma)
        for weight, mu, sigma in zip(weights, log_mu, log_sigma)
    ))
    while cdf(high) < probability:
        high *= 2.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if cdf(middle) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def predict_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True,
                        help="JSON with precomputed composition183 and optional space_group/structure160")
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="External directory containing the 15 FROZEN checkpoints")
    parser.add_argument("--model", choices=("formula", "multimodal"), default="formula")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, help="Write JSON here; stdout is used when omitted")
    args = parser.parse_args()

    from inference import predict_features

    data = json.loads(args.features.read_text(encoding="utf-8-sig"))
    result = predict_features(data, args.model, args.device, args.checkpoint_dir)
    assets = result.pop("checkpoints")
    result["checkpoint_manifest"] = [
        {key: asset[key] for key in ("seed", "fold_id", "sha256")}
        for asset in assets
    ]
    probabilities = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    result["quantiles_ev"] = {
        f"q{int(probability * 100):02d}": _quantile(
            probability, result["weights"], result["log_mu"], result["log_sigma"]
        )
        for probability in probabilities
    }
    result["median_ev"] = result["quantiles_ev"]["q50"]
    result["interval_90_ev"] = [result["quantiles_ev"]["q05"], result["quantiles_ev"]["q95"]]
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


def score_main() -> None:
    parser = argparse.ArgumentParser(description="Score one positive-gap observation against a lognormal mixture.")
    parser.add_argument("--input", type=Path, required=True,
                        help="JSON with gap_ev, weights, log_mu, and log_sigma")
    parser.add_argument("--output", type=Path, help="Write JSON here; stdout is used when omitted")
    args = parser.parse_args()

    import numpy as np
    from density_score import score

    data = json.loads(args.input.read_text(encoding="utf-8-sig"))
    parts = tuple(np.asarray(data[key], dtype=float) for key in ("weights", "log_mu", "log_sigma"))
    result = score(float(data["gap_ev"]), parts)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    predict_main()
