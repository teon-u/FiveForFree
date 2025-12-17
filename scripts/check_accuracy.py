#!/usr/bin/env python3
"""Check model accuracy and prediction history."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Disable loguru before importing anything else
from loguru import logger
logger.disable("src")

from src.models.model_manager import ModelManager
from config.settings import settings

def main():
    mm = ModelManager()
    loaded = mm.load_all_models()
    print(f"Loaded {loaded} models")
    print(f"BACKTEST_HOURS: {settings.BACKTEST_HOURS}")
    print()

    stats = {"100%": [], "90-99%": [], "70-89%": [], "50-69%": [], "<50%": [], "0%": []}

    for ticker in mm.get_tickers():
        if ticker not in mm._models:
            continue
        for model_type in mm._models[ticker]:
            for target in mm._models[ticker][model_type]:
                model = mm._models[ticker][model_type][target]

                accuracy = model.get_recent_accuracy(hours=settings.BACKTEST_HOURS)
                accuracy_pct = accuracy * 100
                pred_stats = model.get_prediction_stats(hours=settings.BACKTEST_HOURS)
                total = pred_stats["total_predictions"]

                key = f"{ticker}_{model_type}_{target}"

                if accuracy_pct == 100:
                    stats["100%"].append((key, total, accuracy_pct))
                elif accuracy_pct >= 90:
                    stats["90-99%"].append((key, total, accuracy_pct))
                elif accuracy_pct >= 70:
                    stats["70-89%"].append((key, total, accuracy_pct))
                elif accuracy_pct >= 50:
                    stats["50-69%"].append((key, total, accuracy_pct))
                elif accuracy_pct > 0:
                    stats["<50%"].append((key, total, accuracy_pct))
                else:
                    stats["0%"].append((key, total, accuracy_pct))

    print("=" * 60)
    print("ACCURACY DISTRIBUTION")
    print("=" * 60)
    for category, items in stats.items():
        print(f"{category}: {len(items)} models")

    print()
    print("=" * 60)
    print("100% ACCURACY MODELS (top 15)")
    print("=" * 60)
    for key, total, acc in sorted(stats["100%"], key=lambda x: -x[1])[:15]:
        print(f"  {key}: {acc:.1f}% ({total} predictions with outcomes)")

    print()
    print("=" * 60)
    print("0% ACCURACY MODELS (top 15)")
    print("=" * 60)
    for key, total, acc in sorted(stats["0%"], key=lambda x: -x[1])[:15]:
        print(f"  {key}: {total} predictions with outcomes")

    print()
    print("=" * 60)
    print("RAW PREDICTION HISTORY CHECK")
    print("=" * 60)

    sample_ticker = mm.get_tickers()[0]
    if sample_ticker in mm._models and "lightgbm" in mm._models[sample_ticker]:
        model = mm._models[sample_ticker]["lightgbm"]["up"]
        history = model.prediction_history
        print(f"{sample_ticker} lightgbm up:")
        print(f"  Total in history: {len(history)}")

        outcomes = [p.get("actual_outcome") for p in history]
        none_count = sum(1 for o in outcomes if o is None)
        true_count = sum(1 for o in outcomes if o is True)
        false_count = sum(1 for o in outcomes if o is False)
        print(f"  actual_outcome=None: {none_count}")
        print(f"  actual_outcome=True: {true_count}")
        print(f"  actual_outcome=False: {false_count}")

        print()
        print("  Sample entries (first 5):")
        for p in history[:5]:
            ts = p.get("timestamp", "N/A")
            prob = p.get("probability", 0)
            outcome = p.get("actual_outcome")
            print(f"    ts={ts}, prob={prob:.3f}, actual_outcome={outcome}")


if __name__ == "__main__":
    main()
