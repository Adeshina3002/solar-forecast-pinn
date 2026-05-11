"""Solar forecasting with physics-informed neural networks."""

from solar_forecast.preprocess import (
    PreprocessConfig,
    build_and_cache,
    build_dataset,
    load_dataset,
)

__all__ = [
    "PreprocessConfig",
    "build_and_cache",
    "build_dataset",
    "load_dataset",
]
__version__ = "0.1.0"
