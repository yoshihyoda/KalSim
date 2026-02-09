"""Tests for plotting utilities."""

from __future__ import annotations

import threading

import pandas as pd

from src.visualizer import plot_results


def test_plot_results_works_in_background_thread(tmp_path):
    """Plotting should succeed when invoked from a non-main thread."""
    analysis_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=4, freq="6h"),
            "sentiment_score": [0.1, -0.2, 0.05, 0.3],
            "sentiment_ma": [0.1, -0.05, -0.02, 0.04],
            "kw_gme": [2, 1, 0, 3],
        }
    )
    output_path = tmp_path / "thread_plot.png"

    errors: list[Exception] = []

    def run_plot() -> None:
        try:
            plot_results(analysis_df, output_path=output_path, show_plot=False)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    worker = threading.Thread(target=run_plot)
    worker.start()
    worker.join()

    assert not errors
    assert output_path.exists()
