#!/usr/bin/env python3
"""Run SHAP analysis on trained models.

Usage:
    python scripts/run_shap_analysis.py
    python scripts/run_shap_analysis.py --ticker MRVL
    python scripts/run_shap_analysis.py --all
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

# Enable logging for debugging
logger.remove()
logger.add(sys.stderr, level="INFO")

from src.models.model_manager import ModelManager
from src.processor.feature_engineer import FeatureEngineer
from src.utils.shap_analyzer import SHAPAnalyzer, analyze_all_models


def main():
    parser = argparse.ArgumentParser(description="Run SHAP analysis on models")
    parser.add_argument('--ticker', type=str, help='Specific ticker to analyze')
    parser.add_argument('--all', action='store_true', help='Analyze all tickers')
    parser.add_argument('--output', type=str, default='Docs/shap_analysis',
                        help='Output directory')

    args = parser.parse_args()

    print("=" * 60)
    print("SHAP Analysis - Model Interpretability")
    print("=" * 60)

    # Load models
    print("\nLoading models...")
    model_manager = ModelManager()
    loaded = model_manager.load_all_models()
    print(f"Loaded {loaded} models from {len(model_manager.get_tickers())} tickers")

    # Create feature engineer
    feature_engineer = FeatureEngineer()

    if args.ticker:
        # Single ticker analysis
        print(f"\nAnalyzing {args.ticker}...")
        tickers = [args.ticker.upper()]
    else:
        # All tickers (limited to top performers)
        tickers = ['MRVL', 'INTC', 'LCID', 'LULU', 'NVDA']
        print(f"\nAnalyzing top tickers: {', '.join(tickers)}")

    # Run analysis
    print("\nRunning SHAP analysis (this may take a few minutes)...")

    try:
        summary = analyze_all_models(
            model_manager=model_manager,
            feature_engineer=feature_engineer,
            output_dir=args.output,
            tickers=tickers,
            model_types=['lightgbm', 'xgboost'],
            targets=['up', 'down']
        )

        print("\n" + "=" * 60)
        print("SHAP ANALYSIS RESULTS")
        print("=" * 60)

        print(f"\nModels analyzed: {summary['n_models_analyzed']}")
        print(f"Tickers analyzed: {summary['n_tickers']}")

        print("\n" + "-" * 40)
        print("TOP 10 GLOBAL FEATURE IMPORTANCE")
        print("-" * 40)

        for i, feat in enumerate(summary['top_10_features'], 1):
            print(f"  {i}. {feat}")

        print("\n" + "-" * 40)
        print("DETAILED FEATURE IMPORTANCE")
        print("-" * 40)

        for feat in summary.get('global_feature_importance', [])[:15]:
            mean_imp = feat.get('mean_importance', 0)
            std_imp = feat.get('std_importance', 0)
            feat_name = feat.get('feature', 'Unknown')
            print(f"  {feat_name:30s} | Importance: {mean_imp:.4f} (+/- {std_imp:.4f})")

        print(f"\nResults saved to: {args.output}/")
        print("  - shap_analysis_summary.json")
        print("  - global_feature_importance.png")

    except ImportError as e:
        print(f"\nError: {e}")
        print("Please install required packages:")
        print("  pip install shap matplotlib")
        return 1

    except Exception as e:
        logger.exception("Analysis failed")
        print(f"\nError during analysis: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
