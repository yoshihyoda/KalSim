"""Visualizer module for plotting analysis results.

Creates visualizations of sentiment trends and keyword frequencies.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .config import SENTIMENT_PLOT_FILE, RESULTS_DIR

logger = logging.getLogger(__name__)


class VisualizerError(Exception):
    """Base exception for visualizer errors."""
    pass


def plot_results(
    analysis_df: pd.DataFrame,
    output_path: Path = SENTIMENT_PLOT_FILE,
    show_plot: bool = False,
) -> Path:
    """Generate and save visualization of analysis results.
    
    Creates a two-panel figure with sentiment trends and keyword frequencies.
    
    Args:
        analysis_df: DataFrame from Predictor.analyze().
        output_path: Path to save the output image.
        show_plot: Whether to display the plot interactively.
        
    Returns:
        Path to the saved plot file.
        
    Raises:
        VisualizerError: If plotting or saving fails.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[1.2, 1])
        fig.suptitle(
            "Simons' Heir: GameStop Simulation Analysis",
            fontsize=16,
            fontweight="bold",
        )
        
        _plot_sentiment_panel(axes[0], analysis_df)
        _plot_keyword_panel(axes[1], analysis_df)
        
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        logger.info(f"Plot saved to {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to create plot: {e}")
        raise VisualizerError(f"Plotting failed: {e}") from e


def _plot_sentiment_panel(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Plot sentiment trends on the given axes.
    
    Args:
        ax: Matplotlib axes to plot on.
        df: Analysis DataFrame with sentiment data.
    """
    if "timestamp" not in df.columns or "sentiment_score" not in df.columns:
        ax.text(0.5, 0.5, "No sentiment data available", ha="center", va="center")
        return
    
    timestamps = df["timestamp"]
    sentiment = df["sentiment_score"]
    
    ax.fill_between(
        timestamps,
        sentiment,
        0,
        where=(sentiment >= 0),
        interpolate=True,
        color="#2ecc71",
        alpha=0.3,
        label="Positive",
    )
    ax.fill_between(
        timestamps,
        sentiment,
        0,
        where=(sentiment < 0),
        interpolate=True,
        color="#e74c3c",
        alpha=0.3,
        label="Negative",
    )
    
    ax.plot(
        timestamps,
        sentiment,
        color="#3498db",
        linewidth=1.5,
        alpha=0.7,
        label="Raw Sentiment",
    )
    
    if "sentiment_ma" in df.columns:
        ax.plot(
            timestamps,
            df["sentiment_ma"],
            color="#2c3e50",
            linewidth=2.5,
            label="Moving Average",
        )
    
    ax.axhline(y=0, color="#7f8c8d", linestyle="--", linewidth=0.8, alpha=0.7)
    
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Sentiment Score", fontsize=11)
    ax.set_title("Community Sentiment Over Time", fontsize=13, fontweight="bold")
    
    ax.set_ylim(-1.1, 1.1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:00"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    
    if "market_price" in df.columns:
        ax2 = ax.twinx()
        price_color = "#f1c40f"  # Golden color for price
        ax2.plot(
            timestamps,
            df["market_price"],
            color=price_color,
            linewidth=2.0,
            linestyle="-.",
            marker=".",
            markersize=8,
            alpha=0.8,
            label="GameStop Price ($)",
        )
        ax2.set_ylabel("Stock Price ($)", fontsize=11, color=price_color)
        ax2.tick_params(axis="y", labelcolor=price_color)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)


def _plot_keyword_panel(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Plot keyword frequency trends on the given axes.
    
    Args:
        ax: Matplotlib axes to plot on.
        df: Analysis DataFrame with keyword data.
    """
    keyword_cols = [col for col in df.columns if col.startswith("kw_")]
    
    if not keyword_cols or "timestamp" not in df.columns:
        ax.text(0.5, 0.5, "No keyword data available", ha="center", va="center")
        return
    
    keyword_totals = {col: df[col].sum() for col in keyword_cols}
    top_keywords = sorted(keyword_totals.items(), key=lambda x: x[1], reverse=True)[:6]
    top_cols = [kw[0] for kw in top_keywords]
    
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    
    timestamps = df["timestamp"]
    bottom = np.zeros(len(df))
    
    for i, col in enumerate(top_cols):
        values = df[col].values
        label = col.replace("kw_", "").replace("_", " ").title()
        ax.bar(
            timestamps,
            values,
            bottom=bottom,
            width=0.035,
            label=label,
            color=colors[i % len(colors)],
            alpha=0.8,
        )
        bottom += values
    
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Keyword Mentions", fontsize=11)
    ax.set_title("Tracked Keyword Frequency", fontsize=13, fontweight="bold")
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:00"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis="y", linestyle="-", linewidth=0.5)


def plot_price_sentiment_correlation(
    analysis_df: pd.DataFrame,
    price_history: list[float],
    output_path: Path | None = None,
) -> Path:
    """Plot price and sentiment correlation.
    
    Args:
        analysis_df: Analysis DataFrame with sentiment data.
        price_history: List of price values.
        output_path: Optional custom output path.
        
    Returns:
        Path to saved plot.
    """
    output_path = output_path or RESULTS_DIR / "price_sentiment_correlation.png"
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    timestamps = analysis_df["timestamp"] if "timestamp" in analysis_df.columns else range(len(analysis_df))
    
    color1 = "#2ecc71"
    ax1.set_xlabel("Time", fontsize=11)
    ax1.set_ylabel("Stock Price ($)", color=color1, fontsize=11)
    
    price_subset = price_history[:len(analysis_df)] if len(price_history) > len(analysis_df) else price_history
    while len(price_subset) < len(analysis_df):
        price_subset.append(price_subset[-1] if price_subset else 20.0)
    
    ax1.plot(
        timestamps[:len(price_subset)],
        price_subset,
        color=color1,
        linewidth=2,
        label="GME Price",
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.fill_between(
        timestamps[:len(price_subset)],
        price_subset,
        alpha=0.1,
        color=color1,
    )
    
    if "sentiment_ma" in analysis_df.columns:
        ax2 = ax1.twinx()
        color2 = "#3498db"
        ax2.set_ylabel("Sentiment (MA)", color=color2, fontsize=11)
        ax2.plot(
            timestamps,
            analysis_df["sentiment_ma"],
            color=color2,
            linewidth=2,
            linestyle="--",
            label="Sentiment MA",
        )
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(-1.1, 1.1)
    
    ax1.set_title(
        "Price vs Community Sentiment Correlation",
        fontsize=14,
        fontweight="bold",
    )
    
    if isinstance(timestamps.iloc[0] if hasattr(timestamps, 'iloc') else timestamps[0], pd.Timestamp):
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:00"))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    if "sentiment_ma" in analysis_df.columns:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    else:
        ax1.legend(loc="upper left")
    
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    logger.info(f"Correlation plot saved to {output_path}")
    return output_path


def generate_summary_report(
    analysis_df: pd.DataFrame,
    summary_stats: dict[str, Any],
    output_path: Path | None = None,
) -> Path:
    """Generate a text summary report.
    
    Args:
        analysis_df: Analysis DataFrame.
        summary_stats: Summary statistics dictionary.
        output_path: Optional custom output path.
        
    Returns:
        Path to saved report.
    """
    output_path = output_path or RESULTS_DIR / "analysis_report.txt"
    
    report_lines = [
        "=" * 60,
        "SIMONS' HEIR - SIMULATION ANALYSIS REPORT",
        "=" * 60,
        "",
        "OVERVIEW",
        "-" * 40,
        f"Time Periods Analyzed: {summary_stats.get('time_periods', 'N/A')}",
        f"Total Tweets Generated: {summary_stats.get('total_tweets', 'N/A')}",
        f"Total Actions Recorded: {summary_stats.get('total_actions', 'N/A')}",
        "",
        "SENTIMENT ANALYSIS",
        "-" * 40,
        f"Average Sentiment: {summary_stats.get('avg_sentiment', 0):.4f}",
        f"Maximum Sentiment: {summary_stats.get('max_sentiment', 0):.4f}",
        f"Minimum Sentiment: {summary_stats.get('min_sentiment', 0):.4f}",
        f"Sentiment Std Dev: {summary_stats.get('sentiment_std', 0):.4f}",
        f"Peak Activity: {summary_stats.get('peak_activity_hour', 'N/A')}",
        "",
        "KEYWORD TRACKING",
        "-" * 40,
    ]
    
    keyword_totals = summary_stats.get("keyword_totals", {})
    for keyword, count in sorted(keyword_totals.items(), key=lambda x: x[1], reverse=True):
        report_lines.append(f"  {keyword}: {count}")
    
    report_lines.extend([
        "",
        "=" * 60,
        "END OF REPORT",
        "=" * 60,
    ])
    
    report_text = "\n".join(report_lines)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    logger.info(f"Report saved to {output_path}")
    return output_path
