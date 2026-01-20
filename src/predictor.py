"""Predictor module for sentiment analysis and trend detection.

Analyzes simulation logs to extract collective sentiment patterns.
"""

import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    POSITIVE_KEYWORDS,
    NEGATIVE_KEYWORDS,
    TRACKED_KEYWORDS,
    SIMULATION_LOG_FILE,
)

logger = logging.getLogger(__name__)


class PredictorError(Exception):
    """Base exception for predictor errors."""
    pass


class Predictor:
    """Analyzes simulation data for sentiment and trend patterns.
    
    Performs keyword-based sentiment analysis and tracks specific
    keyword frequencies over time.
    
    Attributes:
        positive_keywords: List of positive sentiment indicators.
        negative_keywords: List of negative sentiment indicators.
        tracked_keywords: Keywords to track frequency over time.
    """
    
    def __init__(
        self,
        positive_keywords: list[str] | None = None,
        negative_keywords: list[str] | None = None,
        tracked_keywords: list[str] | None = None,
    ) -> None:
        """Initialize the Predictor.
        
        Args:
            positive_keywords: Custom positive keywords (uses default if None).
            negative_keywords: Custom negative keywords (uses default if None).
            tracked_keywords: Custom tracked keywords (uses default if None).
        """
        self.positive_keywords = positive_keywords or POSITIVE_KEYWORDS
        self.negative_keywords = negative_keywords or NEGATIVE_KEYWORDS
        self.tracked_keywords = tracked_keywords or TRACKED_KEYWORDS
        
        self._positive_pattern = self._compile_pattern(self.positive_keywords)
        self._negative_pattern = self._compile_pattern(self.negative_keywords)
        self._tracked_patterns = {
            kw: self._compile_pattern([kw]) for kw in self.tracked_keywords
        }
    
    @staticmethod
    def _compile_pattern(keywords: list[str]) -> re.Pattern[str]:
        """Compile keywords into a regex pattern.
        
        Args:
            keywords: List of keywords to compile.
            
        Returns:
            Compiled regex pattern.
        """
        escaped = [re.escape(kw) for kw in keywords]
        pattern = r"\b(" + "|".join(escaped) + r")\b"
        return re.compile(pattern, re.IGNORECASE)
    
    def analyze(
        self,
        simulation_log: list[dict[str, Any]] | None = None,
        log_file: Path = SIMULATION_LOG_FILE,
    ) -> pd.DataFrame:
        """Analyze simulation log for sentiment and keyword trends.
        
        Args:
            simulation_log: List of action dictionaries, or None to load from file.
            log_file: Path to simulation log file if simulation_log is None.
            
        Returns:
            DataFrame with time-series sentiment and keyword data.
            
        Raises:
            PredictorError: If log loading or analysis fails.
        """
        if simulation_log is None:
            simulation_log, metadata = self._load_log(log_file)
        else:
            metadata = {}
        
        logger.info(f"Analyzing {len(simulation_log)} actions")
        
        hourly_data = self._aggregate_by_hour(simulation_log)
        
        analysis_df = self._build_analysis_dataframe(hourly_data)
        
        analysis_df = self._add_derived_metrics(analysis_df)
        
        logger.info(
            f"Analysis complete: {len(analysis_df)} time periods, "
            f"avg sentiment: {analysis_df['sentiment_score'].mean():.3f}"
        )
        
        return analysis_df
    
    def _load_log(self, log_file: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Load simulation log from file.
        
        Args:
            log_file: Path to the log file.
            
        Returns:
            Tuple of (actions list, metadata dict).
            
        Raises:
            PredictorError: If file loading fails.
        """
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            actions = data.get("actions", data) if isinstance(data, dict) else data
            metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
            
            return actions, metadata
        except (IOError, json.JSONDecodeError) as e:
            raise PredictorError(f"Failed to load log file: {e}") from e
    
    def _aggregate_by_hour(
        self,
        actions: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Aggregate actions by hour.
        
        Args:
            actions: List of action dictionaries.
            
        Returns:
            Dictionary mapping hour keys to aggregated data.
        """
        hourly: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "tweets": [],
                "action_counts": defaultdict(int),
                "sentiment_sum": 0.0,
                "sentiment_count": 0,
                "price_sum": 0.0,
                "price_count": 0,
                "keyword_counts": defaultdict(int),
            }
        )
        
        for action in actions:
            timestamp_str = action.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                hour_key = timestamp.strftime("%Y-%m-%d %H:00")
            except (ValueError, TypeError):
                continue
            
            action_type = action.get("action_type", "UNKNOWN")
            content = action.get("content", "")
            
            # Extract market price
            market_data = action.get("observation", {}).get("market", {})
            price = market_data.get("price")
            if price is not None:
                hourly[hour_key]["price_sum"] += float(price)
                hourly[hour_key]["price_count"] += 1
            
            hourly[hour_key]["action_counts"][action_type] += 1
            
            if action_type == "TWEET" and content:
                hourly[hour_key]["tweets"].append(content)
                
                sentiment = self._calculate_sentiment(content)
                hourly[hour_key]["sentiment_sum"] += sentiment
                hourly[hour_key]["sentiment_count"] += 1
                
                keyword_counts = self._count_keywords(content)
                for kw, count in keyword_counts.items():
                    hourly[hour_key]["keyword_counts"][kw] += count
        
        return hourly
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score for a piece of text.
        
        Args:
            text: Text to analyze.
            
        Returns:
            Sentiment score between -1.0 (negative) and 1.0 (positive).
        """
        positive_matches = len(self._positive_pattern.findall(text))
        negative_matches = len(self._negative_pattern.findall(text))
        
        total = positive_matches + negative_matches
        if total == 0:
            return 0.0
        
        score = (positive_matches - negative_matches) / total
        return max(-1.0, min(1.0, score))
    
    def _count_keywords(self, text: str) -> dict[str, int]:
        """Count occurrences of tracked keywords in text.
        
        Args:
            text: Text to analyze.
            
        Returns:
            Dictionary mapping keywords to their counts.
        """
        counts = {}
        for keyword, pattern in self._tracked_patterns.items():
            matches = pattern.findall(text)
            if matches:
                counts[keyword] = len(matches)
        return counts
    
    def _build_analysis_dataframe(
        self,
        hourly_data: dict[str, dict[str, Any]],
    ) -> pd.DataFrame:
        """Build analysis DataFrame from hourly aggregated data.
        
        Args:
            hourly_data: Dictionary of hourly aggregated data.
            
        Returns:
            DataFrame with analysis results.
        """
        rows = []
        
        for hour_key in sorted(hourly_data.keys()):
            data = hourly_data[hour_key]
            
            sentiment_count = data["sentiment_count"]
            avg_sentiment = (
                data["sentiment_sum"] / sentiment_count
                if sentiment_count > 0
                else 0.0
            )
            
            price_count = data["price_count"]
            avg_price = (
                data["price_sum"] / price_count
                if price_count > 0
                else 0.0
            )
            
            row = {
                "timestamp": datetime.strptime(hour_key, "%Y-%m-%d %H:00"),
                "tweet_count": data["action_counts"].get("TWEET", 0),
                "hold_count": data["action_counts"].get("HOLD", 0),
                "lurk_count": data["action_counts"].get("LURK", 0),
                "total_actions": sum(data["action_counts"].values()),
                "sentiment_score": round(avg_sentiment, 4),
                "sentiment_samples": sentiment_count,
                "market_price": round(avg_price, 2),
            }
            
            for keyword in self.tracked_keywords:
                safe_key = keyword.replace("$", "").replace(" ", "_").lower()
                row[f"kw_{safe_key}"] = data["keyword_counts"].get(keyword, 0)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _add_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived metrics to the analysis DataFrame.
        
        Args:
            df: Input DataFrame with base metrics.
            
        Returns:
            DataFrame with additional derived metrics.
        """
        if len(df) < 2:
            df["sentiment_ma"] = df["sentiment_score"]
            df["sentiment_change"] = 0.0
            df["activity_ma"] = df["tweet_count"].astype(float)
            return df
        
        window = min(6, len(df))
        df["sentiment_ma"] = df["sentiment_score"].rolling(
            window=window, min_periods=1
        ).mean()
        
        df["sentiment_change"] = df["sentiment_score"].diff().fillna(0)
        
        df["activity_ma"] = df["tweet_count"].rolling(
            window=window, min_periods=1
        ).mean()
        
        keyword_cols = [col for col in df.columns if col.startswith("kw_")]
        if keyword_cols:
            df["total_keyword_mentions"] = df[keyword_cols].sum(axis=1)
        
        return df
    
    def get_summary_stats(self, analysis_df: pd.DataFrame) -> dict[str, Any]:
        """Generate summary statistics from analysis.
        
        Args:
            analysis_df: DataFrame from analyze() method.
            
        Returns:
            Dictionary of summary statistics.
        """
        keyword_cols = [col for col in analysis_df.columns if col.startswith("kw_")]
        keyword_totals = {
            col.replace("kw_", ""): int(analysis_df[col].sum())
            for col in keyword_cols
        }
        
        return {
            "time_periods": len(analysis_df),
            "total_tweets": int(analysis_df["tweet_count"].sum()),
            "total_actions": int(analysis_df["total_actions"].sum()),
            "avg_sentiment": float(analysis_df["sentiment_score"].mean()),
            "max_sentiment": float(analysis_df["sentiment_score"].max()),
            "min_sentiment": float(analysis_df["sentiment_score"].min()),
            "sentiment_std": float(analysis_df["sentiment_score"].std()),
            "peak_activity_hour": (
                analysis_df.loc[analysis_df["tweet_count"].idxmax(), "timestamp"].isoformat()
                if len(analysis_df) > 0
                else None
            ),
            "keyword_totals": keyword_totals,
        }
