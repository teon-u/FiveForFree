"""SHAP Analysis Module for Model Interpretability.

Provides:
- Feature importance analysis using SHAP values
- Per-prediction contribution analysis
- Visualization utilities for model interpretation
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

import numpy as np
import pandas as pd
from loguru import logger

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP not installed. Run: pip install shap")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class SHAPAnalyzer:
    """SHAP-based model interpretation analyzer."""

    def __init__(self, model_manager=None):
        """
        Initialize SHAP Analyzer.

        Args:
            model_manager: Optional ModelManager instance for loading models
        """
        if not HAS_SHAP:
            raise ImportError("SHAP library is required. Install with: pip install shap")

        self.model_manager = model_manager
        self._explainers: Dict[str, shap.Explainer] = {}
        self._shap_values_cache: Dict[str, np.ndarray] = {}

    def create_explainer(
        self,
        model,
        model_type: str,
        background_data: Optional[np.ndarray] = None
    ) -> shap.Explainer:
        """
        Create appropriate SHAP explainer for model type.

        Args:
            model: Trained model instance
            model_type: Type of model (lightgbm, xgboost, etc.)
            background_data: Background dataset for KernelExplainer

        Returns:
            SHAP Explainer instance
        """
        if model_type in ['lightgbm', 'xgboost']:
            # Use TreeExplainer for tree-based models (fast and exact)
            # Access internal model via _model attribute
            internal_model = getattr(model, '_model', None)
            if internal_model is None:
                raise ValueError(f"Model has no _model attribute")
            explainer = shap.TreeExplainer(internal_model)
        elif model_type in ['lstm', 'transformer']:
            # Use DeepExplainer or GradientExplainer for neural networks
            if background_data is None:
                raise ValueError("Background data required for neural network explainer")
            # For now, use KernelExplainer as fallback
            explainer = shap.KernelExplainer(
                model.predict_proba,
                shap.sample(background_data, 100)
            )
        else:
            # Fallback to KernelExplainer
            if background_data is None:
                raise ValueError("Background data required for KernelExplainer")
            explainer = shap.KernelExplainer(
                model.predict_proba,
                shap.sample(background_data, 100)
            )

        return explainer

    def compute_shap_values(
        self,
        model,
        model_type: str,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        cache_key: Optional[str] = None
    ) -> Tuple[np.ndarray, shap.Explainer]:
        """
        Compute SHAP values for given data.

        Args:
            model: Trained model
            model_type: Type of model
            X: Input features
            feature_names: Optional feature names
            cache_key: Optional key for caching results

        Returns:
            Tuple of (shap_values, explainer)
        """
        # Check cache
        if cache_key and cache_key in self._shap_values_cache:
            return self._shap_values_cache[cache_key], self._explainers.get(cache_key)

        # Create explainer
        if cache_key and cache_key in self._explainers:
            explainer = self._explainers[cache_key]
        else:
            explainer = self.create_explainer(model, model_type, X)
            if cache_key:
                self._explainers[cache_key] = explainer

        # Compute SHAP values
        shap_values = explainer.shap_values(X)

        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take positive class

        # Cache results
        if cache_key:
            self._shap_values_cache[cache_key] = shap_values

        return shap_values, explainer

    def get_feature_importance(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        method: str = 'mean_abs'
    ) -> pd.DataFrame:
        """
        Calculate feature importance from SHAP values.

        Args:
            shap_values: SHAP values array
            feature_names: List of feature names
            method: Aggregation method ('mean_abs', 'mean', 'max')

        Returns:
            DataFrame with feature importance scores
        """
        if method == 'mean_abs':
            importance = np.abs(shap_values).mean(axis=0)
        elif method == 'mean':
            importance = shap_values.mean(axis=0)
        elif method == 'max':
            importance = np.abs(shap_values).max(axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })

        return df.sort_values('importance', ascending=False).reset_index(drop=True)

    def explain_prediction(
        self,
        shap_values: np.ndarray,
        idx: int,
        feature_names: List[str],
        feature_values: np.ndarray,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.

        Args:
            shap_values: SHAP values array
            idx: Index of prediction to explain
            feature_names: Feature names
            feature_values: Feature values for the prediction
            top_k: Number of top features to include

        Returns:
            Dictionary with explanation details
        """
        pred_shap = shap_values[idx]
        pred_features = feature_values[idx] if len(feature_values.shape) > 1 else feature_values

        # Create DataFrame of contributions
        contributions = pd.DataFrame({
            'feature': feature_names,
            'value': pred_features,
            'contribution': pred_shap,
            'abs_contribution': np.abs(pred_shap)
        })

        # Sort by absolute contribution
        contributions = contributions.sort_values('abs_contribution', ascending=False)

        # Get top positive and negative contributors
        positive = contributions[contributions['contribution'] > 0].head(top_k)
        negative = contributions[contributions['contribution'] < 0].head(top_k)

        return {
            'top_positive': positive[['feature', 'value', 'contribution']].to_dict('records'),
            'top_negative': negative[['feature', 'value', 'contribution']].to_dict('records'),
            'base_value': float(shap_values.mean()) if hasattr(shap_values, 'mean') else 0.0,
            'total_contribution': float(pred_shap.sum()),
        }

    def analyze_ticker_model(
        self,
        ticker: str,
        model_type: str,
        target: str,
        X: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Full SHAP analysis for a ticker's model.

        Args:
            ticker: Ticker symbol
            model_type: Model type (lightgbm, xgboost)
            target: Target (up, down)
            X: Feature data
            feature_names: Feature names

        Returns:
            Complete analysis results
        """
        if self.model_manager is None:
            raise ValueError("ModelManager required for ticker analysis")

        # Get model from _models dict
        ticker_upper = ticker.upper()
        if ticker_upper not in self.model_manager._models:
            return {'error': f'No models for ticker {ticker}'}
        if model_type not in self.model_manager._models[ticker_upper]:
            return {'error': f'No {model_type} models for {ticker}'}
        if target not in self.model_manager._models[ticker_upper][model_type]:
            return {'error': f'No {target} model for {ticker}/{model_type}'}

        model = self.model_manager._models[ticker_upper][model_type][target]
        if model is None or not model.is_trained:
            return {'error': f'Model not available for {ticker}/{model_type}/{target}'}

        # Compute SHAP values
        cache_key = f"{ticker}_{model_type}_{target}"
        shap_values, explainer = self.compute_shap_values(
            model, model_type, X, feature_names, cache_key
        )

        # Get feature importance
        importance_df = self.get_feature_importance(shap_values, feature_names)

        return {
            'ticker': ticker,
            'model_type': model_type,
            'target': target,
            'n_samples': len(X),
            'feature_importance': importance_df.head(20).to_dict('records'),
            'top_features': importance_df['feature'].head(10).tolist(),
            'shap_values_shape': shap_values.shape,
        }

    def generate_summary_plot(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        output_path: str,
        max_display: int = 20,
        title: str = "SHAP Feature Importance"
    ) -> str:
        """
        Generate and save SHAP summary plot.

        Args:
            shap_values: SHAP values
            X: Feature data
            feature_names: Feature names
            output_path: Path to save plot
            max_display: Maximum features to display
            title: Plot title

        Returns:
            Path to saved plot
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required for visualization")

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def generate_bar_plot(
        self,
        importance_df: pd.DataFrame,
        output_path: str,
        top_k: int = 20,
        title: str = "Feature Importance"
    ) -> str:
        """
        Generate feature importance bar plot.

        Args:
            importance_df: DataFrame with feature importance
            output_path: Path to save plot
            top_k: Number of features to show
            title: Plot title

        Returns:
            Path to saved plot
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required for visualization")

        top_features = importance_df.head(top_k)

        plt.figure(figsize=(10, 8))
        plt.barh(
            range(len(top_features)),
            top_features['importance'].values[::-1],
            color='steelblue'
        )
        plt.yticks(
            range(len(top_features)),
            top_features['feature'].values[::-1]
        )
        plt.xlabel('Mean |SHAP Value|')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def generate_waterfall_plot(
        self,
        shap_values: np.ndarray,
        idx: int,
        feature_names: List[str],
        output_path: str,
        max_display: int = 10
    ) -> str:
        """
        Generate waterfall plot for a single prediction.

        Args:
            shap_values: SHAP values array
            idx: Index of prediction
            feature_names: Feature names
            output_path: Path to save plot
            max_display: Max features to show

        Returns:
            Path to saved plot
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required for visualization")

        plt.figure(figsize=(10, 6))

        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values[idx],
            feature_names=feature_names
        )

        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def save_analysis_report(
        self,
        analysis_results: Dict[str, Any],
        output_path: str
    ) -> str:
        """
        Save analysis results to JSON file.

        Args:
            analysis_results: Analysis results dictionary
            output_path: Path to save JSON

        Returns:
            Path to saved file
        """
        # Convert numpy types to Python types
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            return obj

        results = convert_types(analysis_results)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        return output_path


def analyze_all_models(
    model_manager,
    feature_engineer,
    output_dir: str = "Docs/shap_analysis",
    tickers: Optional[List[str]] = None,
    model_types: List[str] = ['lightgbm', 'xgboost'],
    targets: List[str] = ['up', 'down']
) -> Dict[str, Any]:
    """
    Run SHAP analysis on all models.

    Args:
        model_manager: ModelManager instance
        feature_engineer: FeatureEngineer instance
        output_dir: Directory for output files
        tickers: List of tickers (None for all)
        model_types: Model types to analyze
        targets: Targets to analyze

    Returns:
        Summary of all analyses
    """
    from datetime import datetime, timedelta
    from sqlalchemy import select
    from src.utils.database import get_db, MinuteBar as DBMinuteBar

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    analyzer = SHAPAnalyzer(model_manager)

    if tickers is None:
        tickers = model_manager.get_tickers()

    all_results = []
    feature_importance_global = {}

    for ticker in tickers[:10]:  # Limit to 10 for speed
        try:
            print(f"  Processing {ticker}...")
            # Load data
            cutoff = datetime.now() - timedelta(days=30)
            with get_db() as db:
                stmt = (
                    select(DBMinuteBar)
                    .where(DBMinuteBar.symbol == ticker)
                    .where(DBMinuteBar.timestamp >= cutoff)
                    .order_by(DBMinuteBar.timestamp)
                )
                bars = db.execute(stmt).scalars().all()

                if len(bars) < 500:
                    print(f"    Skipping {ticker}: only {len(bars)} bars")
                    continue

                df = pd.DataFrame([{
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "vwap": bar.vwap,
                } for bar in bars])

            # Generate features
            features_df = feature_engineer.compute_features(df)
            feature_names = feature_engineer.get_feature_names()
            X = features_df[feature_names].values

            # Remove NaN rows
            valid_mask = ~np.isnan(X).any(axis=1)
            X = X[valid_mask]

            if len(X) < 100:
                continue

            # Sample for speed
            if len(X) > 500:
                indices = np.random.choice(len(X), 500, replace=False)
                X = X[indices]

            for model_type in model_types:
                for target in targets:
                    try:
                        print(f"    Analyzing {model_type}/{target}...")
                        result = analyzer.analyze_ticker_model(
                            ticker, model_type, target, X, feature_names
                        )
                        if 'error' not in result:
                            print(f"      Success! Features: {len(result.get('feature_importance', []))}")
                            all_results.append(result)

                            # Aggregate feature importance
                            for feat in result['feature_importance']:
                                key = feat['feature']
                                if key not in feature_importance_global:
                                    feature_importance_global[key] = []
                                feature_importance_global[key].append(feat['importance'])
                        else:
                            print(f"      Error: {result.get('error')}")

                    except Exception as e:
                        print(f"      Exception: {e}")

        except Exception as e:
            logger.warning(f"Failed to load data for {ticker}: {e}")

    # Calculate global feature importance
    global_importance = []
    for feat, values in feature_importance_global.items():
        global_importance.append({
            'feature': feat,
            'mean_importance': np.mean(values),
            'std_importance': np.std(values),
            'count': len(values)
        })

    global_importance_df = pd.DataFrame(global_importance)

    if len(global_importance_df) > 0:
        global_importance_df = global_importance_df.sort_values('mean_importance', ascending=False)
        top_features = global_importance_df['feature'].head(10).tolist()
        global_importance_records = global_importance_df.head(30).to_dict('records')
    else:
        top_features = []
        global_importance_records = []

    # Save results
    summary = {
        'n_models_analyzed': len(all_results),
        'n_tickers': len(set(r['ticker'] for r in all_results)) if all_results else 0,
        'global_feature_importance': global_importance_records,
        'top_10_features': top_features,
        'per_model_results': all_results
    }

    # Save to file
    analyzer.save_analysis_report(summary, str(output_path / "shap_analysis_summary.json"))

    # Generate global importance plot
    if HAS_MATPLOTLIB and len(global_importance_df) > 0:
        analyzer.generate_bar_plot(
            global_importance_df,
            str(output_path / "global_feature_importance.png"),
            title="Global Feature Importance (All Models)"
        )

    logger.info(f"SHAP analysis complete. Results saved to {output_dir}")

    return summary
