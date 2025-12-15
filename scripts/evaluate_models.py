#!/usr/bin/env python3
"""Evaluate trained model performance."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sqlalchemy import select

from config.settings import settings
from src.utils.database import get_db, MinuteBar as DBMinuteBar
from src.models.model_manager import ModelManager
from src.processor.feature_engineer import FeatureEngineer
from src.processor.label_generator import LabelGenerator


def main():
    # Load models
    model_manager = ModelManager()
    loaded_count = model_manager.load_all_models()
    print(f"Loaded {loaded_count} models from {model_manager.get_summary()['total_tickers']} tickers")

    # Get all tickers
    tickers = model_manager.get_tickers()
    print(f"Evaluating {len(tickers)} tickers...")

    # Initialize processors
    feature_engineer = FeatureEngineer()
    label_generator = LabelGenerator(
        target_percent=settings.TARGET_PERCENT,
        prediction_horizon_minutes=settings.PREDICTION_HORIZON_MINUTES,
    )

    all_results = []
    label_stats = []

    for i, ticker in enumerate(tickers, 1):
        try:
            # Load data
            cutoff_date = datetime.now() - timedelta(days=60)
            with get_db() as db:
                stmt = (
                    select(DBMinuteBar)
                    .where(DBMinuteBar.symbol == ticker)
                    .where(DBMinuteBar.timestamp >= cutoff_date)
                    .order_by(DBMinuteBar.timestamp)
                )
                bars = db.execute(stmt).scalars().all()

                df = pd.DataFrame(
                    [
                        {
                            "timestamp": bar.timestamp,
                            "open": bar.open,
                            "high": bar.high,
                            "low": bar.low,
                            "close": bar.close,
                            "volume": bar.volume,
                            "vwap": bar.vwap,
                        }
                        for bar in bars
                    ]
                )

            if len(df) < 500:
                continue

            # Generate features
            features_df = feature_engineer.compute_features(df)
            feature_names = feature_engineer.get_feature_names()

            # Generate labels
            labels_up = []
            labels_down = []
            for idx in range(len(df) - settings.PREDICTION_HORIZON_MINUTES - 1):
                entry_time = df.iloc[idx]["timestamp"]
                entry_price = df.iloc[idx]["close"]
                labels = label_generator.generate_labels(df, entry_time, entry_price)
                labels_up.append(labels["label_up"])
                labels_down.append(labels["label_down"])

            # Align features and labels
            X = features_df[feature_names].values[: len(labels_up)]
            y_up = np.array(labels_up)
            y_down = np.array(labels_down)

            # Remove NaN rows
            valid_indices = ~np.isnan(X).any(axis=1)
            X = X[valid_indices]
            y_up = y_up[valid_indices]
            y_down = y_down[valid_indices]

            # Record label stats
            label_stats.append(
                {
                    "ticker": ticker,
                    "samples": len(X),
                    "up_rate": y_up.mean(),
                    "down_rate": y_down.mean(),
                }
            )

            # Use last 20% for testing
            test_size = int(len(X) * 0.2)
            if test_size < 50:
                continue

            X_test = X[-test_size:]
            y_up_test = y_up[-test_size:]
            y_down_test = y_down[-test_size:]

            # Evaluate models for this ticker
            for target in ["up", "down"]:
                y_test = y_up_test if target == "up" else y_down_test
                models = model_manager.get_all_models(ticker, target)

                for model_type, model in models.items():
                    if not model.is_trained:
                        continue

                    try:
                        probs = model.predict_proba(X_test)
                        preds = (probs >= 0.5).astype(int)

                        acc = accuracy_score(y_test, preds)
                        prec = precision_score(y_test, preds, zero_division=0)
                        rec = recall_score(y_test, preds, zero_division=0)
                        f1 = f1_score(y_test, preds, zero_division=0)

                        try:
                            auc = (
                                roc_auc_score(y_test, probs)
                                if len(np.unique(y_test)) > 1
                                else 0.5
                            )
                        except:
                            auc = 0.5

                        all_results.append(
                            {
                                "ticker": ticker,
                                "target": target,
                                "model_type": model_type,
                                "accuracy": acc,
                                "precision": prec,
                                "recall": rec,
                                "f1": f1,
                                "auc": auc,
                                "positive_rate": y_test.mean(),
                            }
                        )
                    except:
                        pass

            if i % 10 == 0:
                print(f"  Processed {i}/{len(tickers)} tickers")

        except Exception as e:
            pass

    print(f"\nEvaluated {len(set(r['ticker'] for r in all_results))} tickers")

    # Summary statistics
    df_results = pd.DataFrame(all_results)
    df_labels = pd.DataFrame(label_stats)

    print("\n" + "=" * 80)
    print("LABEL DISTRIBUTION SUMMARY")
    print("=" * 80)
    print(f"Average positive rate (up):   {df_labels['up_rate'].mean()*100:.1f}%")
    print(f"Average positive rate (down): {df_labels['down_rate'].mean()*100:.1f}%")
    print(f"Max positive rate (up):       {df_labels['up_rate'].max()*100:.1f}%")
    print(f"Max positive rate (down):     {df_labels['down_rate'].max()*100:.1f}%")

    print("\n" + "=" * 80)
    print("AVERAGE PERFORMANCE BY MODEL TYPE")
    print("=" * 80)
    avg_by_model = df_results.groupby("model_type")[
        ["accuracy", "precision", "recall", "f1", "auc"]
    ].mean()
    print(avg_by_model.round(3).to_string())

    print("\n" + "=" * 80)
    print("AVERAGE PERFORMANCE BY TARGET")
    print("=" * 80)
    avg_by_target = df_results.groupby("target")[
        ["accuracy", "precision", "recall", "f1", "auc"]
    ].mean()
    print(avg_by_target.round(3).to_string())

    print("\n" + "=" * 80)
    print("GRAND AVERAGE (ALL MODELS)")
    print("=" * 80)
    print(
        f"Accuracy:  {df_results['accuracy'].mean():.3f} (std: {df_results['accuracy'].std():.3f})"
    )
    print(
        f"Precision: {df_results['precision'].mean():.3f} (std: {df_results['precision'].std():.3f})"
    )
    print(
        f"Recall:    {df_results['recall'].mean():.3f} (std: {df_results['recall'].std():.3f})"
    )
    print(
        f"F1 Score:  {df_results['f1'].mean():.3f} (std: {df_results['f1'].std():.3f})"
    )
    print(
        f"AUC-ROC:   {df_results['auc'].mean():.3f} (std: {df_results['auc'].std():.3f})"
    )

    # Best performing model-ticker combinations
    print("\n" + "=" * 80)
    print("TOP 10 MODEL-TICKER COMBINATIONS (by F1 Score)")
    print("=" * 80)
    top_f1 = df_results.nlargest(10, "f1")[
        ["ticker", "target", "model_type", "accuracy", "precision", "recall", "f1", "auc"]
    ]
    print(top_f1.to_string(index=False))

    print("\n" + "=" * 80)
    print("TOP 10 MODEL-TICKER COMBINATIONS (by AUC)")
    print("=" * 80)
    top_auc = df_results.nlargest(10, "auc")[
        ["ticker", "target", "model_type", "accuracy", "precision", "recall", "f1", "auc"]
    ]
    print(top_auc.to_string(index=False))

    # Models with meaningful predictions (F1 > 0)
    meaningful = df_results[df_results["f1"] > 0]
    print(
        f"\n{len(meaningful)}/{len(df_results)} models ({100*len(meaningful)/len(df_results):.1f}%) making meaningful positive predictions"
    )
    if len(meaningful) > 0:
        print(f"Average F1 for meaningful models: {meaningful['f1'].mean():.3f}")
        print(f"Average AUC for meaningful models: {meaningful['auc'].mean():.3f}")


if __name__ == "__main__":
    main()
