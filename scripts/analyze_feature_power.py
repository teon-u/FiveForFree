#!/usr/bin/env python3
"""
Feature Predictive Power Analysis

Analyzes each feature's ability to predict the target labels:
1. Correlation analysis (linear relationship)
2. Mutual Information (non-linear relationship)
3. Feature importance from trained models
4. Train vs Validation stability
5. Overfitting indicator detection
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select

from config.settings import settings
from src.utils.database import get_db, MinuteBar as DBMinuteBar
from src.models.model_manager import ModelManager
from src.processor.feature_engineer import FeatureEngineer
from src.processor.label_generator import LabelGenerator


def load_all_data(tickers: list, days: int = 60) -> tuple:
    """Load and combine data from all tickers."""
    print(f"Loading data for {len(tickers)} tickers...")

    feature_engineer = FeatureEngineer()
    label_generator = LabelGenerator(
        target_percent=settings.TARGET_PERCENT,
        prediction_horizon_minutes=settings.PREDICTION_HORIZON_MINUTES,
    )

    all_features = []
    all_labels_up = []
    all_labels_down = []

    cutoff_date = datetime.now() - timedelta(days=days)

    for ticker in tickers:
        try:
            with get_db() as db:
                stmt = (
                    select(DBMinuteBar)
                    .where(DBMinuteBar.symbol == ticker)
                    .where(DBMinuteBar.timestamp >= cutoff_date)
                    .order_by(DBMinuteBar.timestamp)
                )
                bars = db.execute(stmt).scalars().all()

                if len(bars) < 500:
                    continue

                df = pd.DataFrame([
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
                ])

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

            # Align
            X = features_df[feature_names].values[:len(labels_up)]
            y_up = np.array(labels_up)
            y_down = np.array(labels_down)

            # Remove NaN
            valid_mask = ~np.isnan(X).any(axis=1)
            X = X[valid_mask]
            y_up = y_up[valid_mask]
            y_down = y_down[valid_mask]

            if len(X) > 100:
                all_features.append(X)
                all_labels_up.append(y_up)
                all_labels_down.append(y_down)

        except Exception as e:
            continue

    if not all_features:
        return None, None, None, None

    X_all = np.vstack(all_features)
    y_up_all = np.concatenate(all_labels_up)
    y_down_all = np.concatenate(all_labels_down)

    print(f"Total samples: {len(X_all):,}")
    print(f"Label up rate: {y_up_all.mean()*100:.1f}%")
    print(f"Label down rate: {y_down_all.mean()*100:.1f}%")

    return X_all, y_up_all, y_down_all, feature_engineer.get_feature_names()


def analyze_correlations(X: np.ndarray, y: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Calculate correlation between each feature and target."""
    correlations = []

    for i, name in enumerate(feature_names):
        feature = X[:, i]

        # Pearson correlation
        valid_mask = ~np.isnan(feature) & ~np.isinf(feature)
        if valid_mask.sum() > 100:
            corr, p_value = stats.pearsonr(feature[valid_mask], y[valid_mask])
        else:
            corr, p_value = 0, 1

        # Point-biserial correlation (for binary target)
        if valid_mask.sum() > 100:
            pb_corr, pb_p = stats.pointbiserialr(y[valid_mask], feature[valid_mask])
        else:
            pb_corr, pb_p = 0, 1

        correlations.append({
            'feature': name,
            'pearson_corr': corr,
            'pearson_p': p_value,
            'pointbiserial_corr': pb_corr,
            'pointbiserial_p': pb_p,
            'abs_corr': abs(corr)
        })

    return pd.DataFrame(correlations).sort_values('abs_corr', ascending=False)


def analyze_mutual_information(X: np.ndarray, y: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Calculate mutual information between features and target."""
    # Handle NaN/Inf
    X_clean = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Calculate MI
    mi_scores = mutual_info_classif(X_scaled, y, random_state=42, n_neighbors=5)

    mi_df = pd.DataFrame({
        'feature': feature_names,
        'mutual_info': mi_scores
    }).sort_values('mutual_info', ascending=False)

    return mi_df


def analyze_train_val_stability(X: np.ndarray, y: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Check if feature-target relationship is stable across train/val."""
    n_samples = len(X)
    n_train = int(n_samples * 0.8)

    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    stability_results = []

    for i, name in enumerate(feature_names):
        feat_train = X_train[:, i]
        feat_val = X_val[:, i]

        # Correlation in train
        valid_train = ~np.isnan(feat_train) & ~np.isinf(feat_train)
        if valid_train.sum() > 50:
            corr_train, _ = stats.pearsonr(feat_train[valid_train], y_train[valid_train])
        else:
            corr_train = 0

        # Correlation in val
        valid_val = ~np.isnan(feat_val) & ~np.isinf(feat_val)
        if valid_val.sum() > 50:
            corr_val, _ = stats.pearsonr(feat_val[valid_val], y_val[valid_val])
        else:
            corr_val = 0

        # Stability score (similar correlation = stable)
        stability = 1 - abs(corr_train - corr_val)

        # Sign flip detection (dangerous!)
        sign_flip = (corr_train * corr_val) < 0

        stability_results.append({
            'feature': name,
            'corr_train': corr_train,
            'corr_val': corr_val,
            'corr_diff': corr_train - corr_val,
            'stability': stability,
            'sign_flip': sign_flip
        })

    return pd.DataFrame(stability_results).sort_values('stability', ascending=True)


def get_model_feature_importance(model_manager: ModelManager, feature_names: list) -> pd.DataFrame:
    """Extract feature importance from trained tree models."""
    importance_scores = {name: [] for name in feature_names}

    tickers = model_manager.get_tickers()

    for ticker in tickers:
        for model_type in ['xgboost', 'lightgbm']:
            try:
                if ticker not in model_manager._models:
                    continue
                if model_type not in model_manager._models[ticker]:
                    continue
                if 'up' not in model_manager._models[ticker][model_type]:
                    continue

                model = model_manager._models[ticker][model_type]['up']

                if not model.is_trained or model._model is None:
                    continue

                # Get feature importance
                if hasattr(model._model, 'feature_importances_'):
                    importances = model._model.feature_importances_

                    if len(importances) == len(feature_names):
                        for i, name in enumerate(feature_names):
                            importance_scores[name].append(importances[i])
            except:
                continue

    # Calculate mean importance
    importance_df = pd.DataFrame([
        {
            'feature': name,
            'mean_importance': np.mean(scores) if scores else 0,
            'std_importance': np.std(scores) if scores else 0,
            'n_models': len(scores)
        }
        for name, scores in importance_scores.items()
    ]).sort_values('mean_importance', ascending=False)

    return importance_df


def identify_problematic_features(corr_df: pd.DataFrame, mi_df: pd.DataFrame,
                                   stability_df: pd.DataFrame, importance_df: pd.DataFrame) -> dict:
    """Identify features that might be causing problems."""

    # Merge all analyses
    analysis = corr_df.merge(mi_df, on='feature')
    analysis = analysis.merge(stability_df, on='feature')
    analysis = analysis.merge(importance_df, on='feature')

    problematic = {
        'no_predictive_power': [],
        'unstable': [],
        'sign_flip': [],
        'high_importance_low_correlation': [],
        'potential_leakage': []
    }

    for _, row in analysis.iterrows():
        feature = row['feature']

        # No predictive power (low correlation AND low MI)
        if abs(row['pearson_corr']) < 0.01 and row['mutual_info'] < 0.001:
            problematic['no_predictive_power'].append(feature)

        # Unstable (correlation changes significantly between train/val)
        if abs(row['corr_diff']) > 0.1:
            problematic['unstable'].append(feature)

        # Sign flip (correlation direction changes)
        if row['sign_flip']:
            problematic['sign_flip'].append(feature)

        # High importance but low correlation (might be overfitting to noise)
        if row['mean_importance'] > 0.02 and abs(row['pearson_corr']) < 0.02:
            problematic['high_importance_low_correlation'].append(feature)

        # Suspiciously high correlation (potential leakage)
        if abs(row['pearson_corr']) > 0.3:
            problematic['potential_leakage'].append(feature)

    return problematic, analysis


def main():
    print("=" * 70)
    print("FEATURE PREDICTIVE POWER ANALYSIS")
    print("=" * 70)

    # Load models
    print("\n[1/6] Loading trained models...")
    model_manager = ModelManager()
    model_manager.load_all_models()
    tickers = model_manager.get_tickers()
    print(f"Found {len(tickers)} tickers with models")

    # Load data
    print("\n[2/6] Loading and preparing data...")
    X, y_up, y_down, feature_names = load_all_data(tickers[:30], days=60)  # Use subset for speed

    if X is None:
        print("Failed to load data")
        return

    print(f"Features: {len(feature_names)}")

    # Correlation analysis
    print("\n[3/6] Analyzing correlations...")
    corr_df = analyze_correlations(X, y_up, feature_names)

    # Mutual information
    print("\n[4/6] Calculating mutual information...")
    mi_df = analyze_mutual_information(X, y_up, feature_names)

    # Train/Val stability
    print("\n[5/6] Checking train/validation stability...")
    stability_df = analyze_train_val_stability(X, y_up, feature_names)

    # Model feature importance
    print("\n[6/6] Extracting model feature importance...")
    importance_df = get_model_feature_importance(model_manager, feature_names)

    # Identify problematic features
    problematic, full_analysis = identify_problematic_features(
        corr_df, mi_df, stability_df, importance_df
    )

    # ==================== RESULTS ====================
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)

    # Top correlated features
    print("\n--- TOP 10 CORRELATED FEATURES (with label_up) ---")
    print(f"{'Feature':<30} {'Correlation':>12} {'P-value':>12}")
    print("-" * 56)
    for _, row in corr_df.head(10).iterrows():
        sig = "***" if row['pearson_p'] < 0.001 else "**" if row['pearson_p'] < 0.01 else "*" if row['pearson_p'] < 0.05 else ""
        print(f"{row['feature']:<30} {row['pearson_corr']:>+12.4f} {row['pearson_p']:>10.2e} {sig}")

    # Top MI features
    print("\n--- TOP 10 MUTUAL INFORMATION FEATURES ---")
    print(f"{'Feature':<30} {'MI Score':>12}")
    print("-" * 44)
    for _, row in mi_df.head(10).iterrows():
        print(f"{row['feature']:<30} {row['mutual_info']:>12.4f}")

    # Top model importance
    print("\n--- TOP 10 MODEL IMPORTANCE FEATURES ---")
    print(f"{'Feature':<30} {'Importance':>12} {'Std':>10}")
    print("-" * 54)
    for _, row in importance_df.head(10).iterrows():
        print(f"{row['feature']:<30} {row['mean_importance']:>12.4f} {row['std_importance']:>10.4f}")

    # Unstable features
    print("\n--- UNSTABLE FEATURES (Train/Val correlation differs) ---")
    unstable = stability_df[abs(stability_df['corr_diff']) > 0.05].head(10)
    print(f"{'Feature':<30} {'Train Corr':>12} {'Val Corr':>12} {'Diff':>10}")
    print("-" * 66)
    for _, row in unstable.iterrows():
        print(f"{row['feature']:<30} {row['corr_train']:>+12.4f} {row['corr_val']:>+12.4f} {row['corr_diff']:>+10.4f}")

    # Sign flip features (DANGEROUS)
    sign_flips = stability_df[stability_df['sign_flip'] == True]
    if len(sign_flips) > 0:
        print(f"\n--- SIGN FLIP FEATURES (Correlation reverses!) ---")
        print(f"{'Feature':<30} {'Train Corr':>12} {'Val Corr':>12}")
        print("-" * 56)
        for _, row in sign_flips.iterrows():
            print(f"{row['feature']:<30} {row['corr_train']:>+12.4f} {row['corr_val']:>+12.4f}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_corr = corr_df['abs_corr'].mean()
    max_corr = corr_df['abs_corr'].max()
    avg_mi = mi_df['mutual_info'].mean()

    print(f"\nOverall Feature Quality:")
    print(f"  Average |correlation|:     {avg_corr:.4f}")
    print(f"  Maximum |correlation|:     {max_corr:.4f}")
    print(f"  Average Mutual Info:       {avg_mi:.4f}")

    print(f"\nProblematic Features:")
    print(f"  No predictive power:       {len(problematic['no_predictive_power'])}")
    print(f"  Unstable (train/val):      {len(problematic['unstable'])}")
    print(f"  Sign flip:                 {len(problematic['sign_flip'])}")
    print(f"  High importance, low corr: {len(problematic['high_importance_low_correlation'])}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if max_corr < 0.05:
        print("\n[CRITICAL] Maximum correlation is very low (<0.05)")
        print("  -> Features have almost NO linear predictive power")
        print("  -> Model is likely fitting to noise")
    elif max_corr < 0.1:
        print("\n[WARNING] Maximum correlation is low (<0.1)")
        print("  -> Features have weak predictive power")
        print("  -> Need stronger features or different approach")
    else:
        print(f"\n[OK] Maximum correlation: {max_corr:.4f}")

    if len(problematic['sign_flip']) > 5:
        print(f"\n[CRITICAL] {len(problematic['sign_flip'])} features have sign flip!")
        print("  -> Feature-target relationship is unstable")
        print("  -> These features likely cause overfitting")

    if len(problematic['high_importance_low_correlation']) > 5:
        print(f"\n[WARNING] {len(problematic['high_importance_low_correlation'])} features: high importance but low correlation")
        print("  -> Model may be overfitting to noise in these features")

    # Save full analysis
    output_path = project_root / "data" / "feature_analysis.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_analysis.to_csv(output_path, index=False)
    print(f"\n\nFull analysis saved to: {output_path}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
