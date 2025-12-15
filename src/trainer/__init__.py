"""
Training modules for NASDAQ prediction system.

This package provides GPU-accelerated training capabilities:

GPU Training:
    - GPUParallelTrainer: Parallel training orchestrator for RTX 5080
    - ThreadPoolExecutor for tree models (XGBoost, LightGBM)
    - Sequential training for neural models (LSTM, Transformer)

Incremental Learning:
    - IncrementalTrainer: Online learning for tree-based models
    - Efficient updates with new data
    - Maintains model performance over time
"""

from src.trainer.gpu_trainer import GPUParallelTrainer
from src.trainer.incremental import IncrementalTrainer

__all__ = [
    'GPUParallelTrainer',
    'IncrementalTrainer',
]

__version__ = '1.0.0'
