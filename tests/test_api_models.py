"""Tests for model API endpoints."""
import pytest
from datetime import datetime


class TestModelPerformanceEndpoints:
    """Test suite for /api/models endpoints."""

    def test_get_model_performance_success(self, test_client):
        """Test getting model performance for valid ticker."""
        response = test_client.get("/api/models/AAPL")

        assert response.status_code == 200
        data = response.json()

        assert data['ticker'] == 'AAPL'
        assert 'up_models' in data
        assert 'down_models' in data
        assert 'best_up_model' in data
        assert 'best_down_model' in data
        assert 'timestamp' in data

    def test_get_model_performance_normalizes_ticker(self, test_client):
        """Test that ticker is normalized to uppercase."""
        response = test_client.get("/api/models/aapl")

        assert response.status_code == 200
        data = response.json()
        assert data['ticker'] == 'AAPL'

    def test_get_model_performance_not_found(self, test_client):
        """Test 404 for non-existent ticker."""
        response = test_client.get("/api/models/NONEXISTENT")

        assert response.status_code == 404

    def test_get_model_performance_model_metrics_structure(self, test_client):
        """Test that model metrics have correct structure."""
        response = test_client.get("/api/models/AAPL")
        data = response.json()

        if data['up_models']:
            model = data['up_models'][0]
            assert 'model_type' in model
            assert 'target' in model
            assert 'hit_rate_50h' in model
            assert 'total_predictions' in model
            assert 'is_trained' in model

    def test_get_model_performance_models_sorted_by_hit_rate(self, test_client):
        """Test that models are sorted by hit rate descending."""
        response = test_client.get("/api/models/AAPL")
        data = response.json()

        up_models = data['up_models']
        if len(up_models) >= 2:
            for i in range(len(up_models) - 1):
                assert up_models[i]['hit_rate_50h'] >= up_models[i+1]['hit_rate_50h']

    def test_get_all_models_performance_success(self, test_client):
        """Test getting all models performance."""
        response = test_client.get("/api/models")

        assert response.status_code == 200
        data = response.json()

        assert 'ticker_performances' in data
        assert 'summary' in data
        assert 'timestamp' in data

    def test_get_all_models_summary_structure(self, test_client):
        """Test that summary has correct structure."""
        response = test_client.get("/api/models")
        data = response.json()

        summary = data['summary']
        assert 'total_tickers' in summary
        assert 'total_models' in summary
        assert 'trained_models' in summary
        assert 'untrained_models' in summary

    def test_get_model_prediction_stats_success(self, test_client):
        """Test getting prediction stats for ticker."""
        response = test_client.get("/api/models/AAPL/stats")

        assert response.status_code == 200
        data = response.json()

        assert data['ticker'] == 'AAPL'
        assert 'statistics' in data

    def test_get_overall_summary_success(self, test_client):
        """Test getting overall summary."""
        response = test_client.get("/api/models/summary/overall")

        assert response.status_code == 200
        data = response.json()

        assert 'total_tickers' in data
        assert 'total_models' in data
        assert 'trained_models' in data
        assert 'average_hit_rate_50h' in data
        assert 'timestamp' in data


class TestModelOverviewEndpoint:
    """Test suite for /api/models/{ticker}/overview endpoint."""

    def test_get_model_overview_success(self, test_client):
        """Test getting model overview."""
        response = test_client.get("/api/models/AAPL/overview")

        assert response.status_code == 200
        data = response.json()

        assert data['ticker'] == 'AAPL'
        assert 'prediction' in data
        assert 'ranking' in data
        assert 'quick_stats' in data
        assert 'risk_indicators' in data
        assert 'timestamp' in data

    def test_model_overview_prediction_structure(self, test_client):
        """Test prediction structure in overview."""
        response = test_client.get("/api/models/AAPL/overview")
        data = response.json()

        prediction = data['prediction']
        assert 'direction' in prediction
        assert 'probability' in prediction
        assert 'best_model' in prediction
        assert 'risk_level' in prediction
        assert prediction['direction'] in ['up', 'down']
        assert prediction['risk_level'] in ['low', 'moderate', 'high']

    def test_model_overview_quick_stats_structure(self, test_client):
        """Test quick stats structure in overview."""
        response = test_client.get("/api/models/AAPL/overview")
        data = response.json()

        stats = data['quick_stats']
        assert 'accuracy' in stats
        assert 'win_rate' in stats
        assert 'avg_return' in stats
        assert 'sharpe' in stats

    def test_model_overview_risk_indicators_structure(self, test_client):
        """Test risk indicators structure in overview."""
        response = test_client.get("/api/models/AAPL/overview")
        data = response.json()

        risk = data['risk_indicators']
        assert 'false_positive_rate' in risk
        assert 'model_agreement' in risk
        assert 'performance_trend' in risk
        assert risk['performance_trend'] in ['improving', 'declining', 'stable', 'unknown']


class TestModelPerformanceDetailsEndpoint:
    """Test suite for /api/models/{ticker}/performance endpoint."""

    def test_get_performance_details_success(self, test_client):
        """Test getting detailed performance metrics."""
        response = test_client.get("/api/models/AAPL/performance")

        assert response.status_code == 200
        data = response.json()

        assert data['ticker'] == 'AAPL'
        assert 'confusion_matrix' in data
        assert 'metrics' in data
        assert 'roc_curve' in data
        assert 'pr_curve' in data
        assert 'time_series' in data
        assert 'calibration' in data

    def test_performance_confusion_matrix_structure(self, test_client):
        """Test confusion matrix structure."""
        response = test_client.get("/api/models/AAPL/performance")
        data = response.json()

        cm = data['confusion_matrix']
        assert 'tp' in cm
        assert 'fp' in cm
        assert 'tn' in cm
        assert 'fn' in cm

    def test_performance_metrics_structure(self, test_client):
        """Test metrics structure."""
        response = test_client.get("/api/models/AAPL/performance")
        data = response.json()

        metrics = data['metrics']
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'fp_rate' in metrics
        assert 'fn_rate' in metrics

    def test_performance_roc_curve_structure(self, test_client):
        """Test ROC curve data structure."""
        response = test_client.get("/api/models/AAPL/performance")
        data = response.json()

        roc = data['roc_curve']
        assert 'fpr' in roc
        assert 'tpr' in roc
        assert 'auc' in roc

    def test_performance_metrics_in_valid_range(self, test_client):
        """Test that metrics are in valid range [0, 1]."""
        response = test_client.get("/api/models/AAPL/performance")
        data = response.json()

        metrics = data['metrics']
        for key, value in metrics.items():
            assert 0 <= value <= 1, f"{key} should be between 0 and 1"


class TestEnsembleAnalysisEndpoint:
    """Test suite for /api/models/{ticker}/ensemble endpoint."""

    def test_get_ensemble_analysis_success(self, test_client):
        """Test getting ensemble analysis."""
        response = test_client.get("/api/models/AAPL/ensemble")

        assert response.status_code == 200
        data = response.json()

        assert data['ticker'] == 'AAPL'
        assert 'meta_learner' in data
        assert 'base_models' in data
        assert 'current_agreement' in data
        assert 'ensemble_vs_base' in data

    def test_ensemble_meta_learner_structure(self, test_client):
        """Test meta learner structure."""
        response = test_client.get("/api/models/AAPL/ensemble")
        data = response.json()

        meta = data['meta_learner']
        assert 'type' in meta
        assert 'weights' in meta
        assert 'trained_base_models' in meta
        assert 'total_base_models' in meta

    def test_ensemble_base_models_structure(self, test_client):
        """Test base models structure."""
        response = test_client.get("/api/models/AAPL/ensemble")
        data = response.json()

        base_models = data['base_models']
        assert isinstance(base_models, list)

        if base_models:
            model = base_models[0]
            assert 'type' in model
            assert 'accuracy' in model
            assert 'is_trained' in model

    def test_ensemble_vs_base_comparison(self, test_client):
        """Test ensemble vs base model comparison."""
        response = test_client.get("/api/models/AAPL/ensemble")
        data = response.json()

        comparison = data['ensemble_vs_base']
        assert 'ensemble_accuracy' in comparison
        assert 'best_base_accuracy' in comparison
        assert 'avg_base_accuracy' in comparison
        assert 'improvement' in comparison


class TestFinancialAnalysisEndpoint:
    """Test suite for /api/models/{ticker}/financial endpoint."""

    def test_get_financial_analysis_success(self, test_client):
        """Test getting financial analysis."""
        response = test_client.get("/api/models/AAPL/financial")

        assert response.status_code == 200
        data = response.json()

        assert data['ticker'] == 'AAPL'
        assert 'equity_curve' in data
        assert 'risk_metrics' in data
        assert 'trade_metrics' in data
        assert 'trade_distribution' in data
        assert 'recent_trades' in data

    def test_financial_with_custom_hours(self, test_client):
        """Test financial analysis with custom hours parameter."""
        response = test_client.get("/api/models/AAPL/financial?hours=100")

        assert response.status_code == 200

    def test_financial_risk_metrics_structure(self, test_client):
        """Test risk metrics structure."""
        response = test_client.get("/api/models/AAPL/financial")
        data = response.json()

        risk = data['risk_metrics']
        assert 'sharpe_ratio' in risk
        assert 'sortino_ratio' in risk
        assert 'max_drawdown' in risk
        assert 'calmar_ratio' in risk

    def test_financial_trade_metrics_structure(self, test_client):
        """Test trade metrics structure."""
        response = test_client.get("/api/models/AAPL/financial")
        data = response.json()

        trade = data['trade_metrics']
        assert 'total_trades' in trade
        assert 'win_rate' in trade
        assert 'total_return' in trade
        assert 'avg_return' in trade
        assert 'profit_factor' in trade

    def test_financial_equity_curve_structure(self, test_client):
        """Test equity curve structure."""
        response = test_client.get("/api/models/AAPL/financial")
        data = response.json()

        equity_curve = data['equity_curve']
        assert isinstance(equity_curve, list)

        if equity_curve:
            point = equity_curve[0]
            assert 'timestamp' in point
            assert 'equity' in point

    def test_financial_not_found_for_invalid_ticker(self, test_client):
        """Test 404 for non-existent ticker."""
        response = test_client.get("/api/models/NONEXISTENT/financial")

        assert response.status_code == 404
