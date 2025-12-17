#!/usr/bin/env python3
"""Compare model performance including ensemble."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sqlalchemy import select

from config.settings import settings
from src.utils.database import get_db, MinuteBar as DBMinuteBar
from src.models.model_manager import ModelManager
from src.processor.feature_engineer import FeatureEngineer
from src.processor.label_generator import LabelGenerator


def main():
    # Initialize
    mm = ModelManager()
    mm.load_all_models()
    fe = FeatureEngineer()
    lg = LabelGenerator(
        target_percent=settings.TARGET_PERCENT,
        prediction_horizon_minutes=settings.PREDICTION_HORIZON_MINUTES,
    )

    tickers = ['BIIB', 'COIN', 'HOOD', 'NVDA', 'AAPL']
    target = 'up'
    model_types = ['xgboost', 'lightgbm', 'lstm', 'transformer', 'ensemble']

    print('=' * 80)
    print('FINAL ENSEMBLE PERFORMANCE COMPARISON')
    print('(4 Base Models: XGBoost, LightGBM, LSTM, Transformer + Ensemble)')
    print('=' * 80)

    all_results = []

    for ticker in tickers:
        print(f'\n--- {ticker} ---')

        # Load data
        cutoff_date = datetime.now() - timedelta(days=30)
        with get_db() as db:
            stmt = (
                select(DBMinuteBar)
                .where(DBMinuteBar.symbol == ticker)
                .where(DBMinuteBar.timestamp >= cutoff_date)
                .order_by(DBMinuteBar.timestamp)
            )
            bars = db.execute(stmt).scalars().all()

            df = pd.DataFrame([
                {
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'vwap': bar.vwap,
                }
                for bar in bars
            ])

        if len(df) < 500:
            print(f'  Insufficient data: {len(df)} bars')
            continue

        # Generate features
        features_df = fe.compute_features(df)
        feature_names = fe.get_feature_names()

        # Generate labels
        labels_up = []
        for idx in range(len(df) - settings.PREDICTION_HORIZON_MINUTES - 1):
            entry_time = df.iloc[idx]['timestamp']
            entry_price = df.iloc[idx]['close']
            labels = lg.generate_labels(df, entry_time, entry_price)
            labels_up.append(labels['label_up'])

        X = features_df[feature_names].values[:len(labels_up)]
        y = np.array(labels_up)

        # Remove NaN rows
        valid_indices = ~np.isnan(X).any(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]

        # Use last 20% for testing
        test_size = int(len(X) * 0.2)
        X_test = X[-test_size:]
        y_test = y[-test_size:]

        print(f'  Test samples: {len(X_test)}, Positive rate: {y_test.mean():.1%}')

        # Evaluate all models
        for model_type in model_types:
            try:
                _, model = mm.get_or_create_model(ticker, model_type, target)

                if model is None or not model.is_trained:
                    continue

                probs = model.predict_proba(X_test)

                # Handle different output formats
                if isinstance(probs, np.ndarray):
                    if len(probs.shape) > 1 and probs.shape[1] > 1:
                        probs = probs[:, 1]
                    probs = probs.flatten()

                # Skip if NaN
                if np.isnan(probs).any():
                    continue

                # Align lengths
                min_len = min(len(probs), len(y_test))
                probs = probs[-min_len:]
                y_aligned = y_test[-min_len:]

                preds = (probs >= 0.5).astype(int)

                prec = precision_score(y_aligned, preds, zero_division=0)
                rec = recall_score(y_aligned, preds, zero_division=0)
                f1 = f1_score(y_aligned, preds, zero_division=0)

                try:
                    auc = roc_auc_score(y_aligned, probs) if len(np.unique(y_aligned)) > 1 else 0.5
                except:
                    auc = 0.5

                all_results.append({
                    'ticker': ticker,
                    'model_type': model_type,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1,
                    'auc': auc,
                    'predictions': int(preds.sum()),
                    'samples': len(y_aligned)
                })

                print(f'  {model_type:12s}: Prec={prec:.1%} Rec={rec:.1%} F1={f1:.3f} AUC={auc:.3f} Preds={preds.sum()}')

            except Exception as e:
                print(f'  {model_type:12s}: Error - {str(e)[:40]}')

    # Summary
    if all_results:
        df_results = pd.DataFrame(all_results)

        print('\n' + '=' * 80)
        print('SUMMARY BY MODEL TYPE (Average across 5 tickers)')
        print('=' * 80)

        avg = df_results.groupby('model_type')[['precision', 'recall', 'f1', 'auc']].mean()
        count = df_results.groupby('model_type').size()

        print('\nModel        Precision     Recall         F1        AUC  Tickers')
        print('-' * 62)
        for m in model_types:
            if m in avg.index:
                r = avg.loc[m]
                c = count.loc[m]
                print(f"{m:<12} {r['precision']*100:>8.1f}%  {r['recall']*100:>8.1f}%  {r['f1']:>8.3f}  {r['auc']:>8.3f}  {c:>6}")

        # Per-ticker comparison
        print('\n' + '=' * 80)
        print('ENSEMBLE vs BEST BASE MODEL (Per Ticker)')
        print('=' * 80)

        for ticker in tickers:
            ticker_data = df_results[df_results['ticker'] == ticker]
            if len(ticker_data) == 0:
                continue

            base_models = ticker_data[ticker_data['model_type'].isin(['xgboost', 'lightgbm', 'lstm', 'transformer'])]
            ensemble = ticker_data[ticker_data['model_type'] == 'ensemble']

            if len(base_models) > 0 and len(ensemble) > 0:
                best_base = base_models.loc[base_models['precision'].idxmax()]
                ens = ensemble.iloc[0]

                diff = ens['precision'] - best_base['precision']
                winner = 'ENSEMBLE' if diff > 0 else best_base['model_type'].upper()

                sign = '+' if diff >= 0 else ''
                print(f"{ticker}: Ensemble={ens['precision']*100:.1f}% vs Best({best_base['model_type']})={best_base['precision']*100:.1f}% -> {winner} ({sign}{diff*100:.1f}%)")

        # Profitability check
        print('\n' + '=' * 80)
        print('PROFITABILITY CHECK (3:1 R/R Breakeven: 30% Precision)')
        print('=' * 80)

        profitable = df_results[df_results['precision'] >= 0.30]
        print(f"Models above breakeven: {len(profitable)}/{len(df_results)} ({100*len(profitable)/len(df_results):.1f}%)")

        if len(profitable) > 0:
            print('\nProfitable combinations (sorted by precision):')
            for _, row in profitable.sort_values('precision', ascending=False).iterrows():
                print(f"  {row['ticker']:5s} {row['model_type']:12s}: {row['precision']*100:.1f}% precision ({row['predictions']} predictions)")

        print('\n' + '=' * 80)
        print('KEY INSIGHTS')
        print('=' * 80)

        # Best overall model
        best_overall = df_results.loc[df_results['precision'].idxmax()]
        print(f"Best overall: {best_overall['ticker']} {best_overall['model_type']} with {best_overall['precision']*100:.1f}% precision")

        # Ensemble wins
        ensemble_wins = 0
        for ticker in tickers:
            ticker_data = df_results[df_results['ticker'] == ticker]
            base_best = ticker_data[ticker_data['model_type'].isin(['xgboost', 'lightgbm', 'lstm', 'transformer'])]['precision'].max()
            ens_prec = ticker_data[ticker_data['model_type'] == 'ensemble']['precision'].values
            if len(ens_prec) > 0 and ens_prec[0] > base_best:
                ensemble_wins += 1

        print(f"Ensemble outperforms base models: {ensemble_wins}/{len(tickers)} tickers")


if __name__ == '__main__':
    main()
