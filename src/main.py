#!/usr/bin/env python3
"""Main entry point for Simons' Heir simulation.

Orchestrates the simulation pipeline: setup, execution, analysis, and visualization.
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import (
    SIMULATION_DAYS,
    AGENTS_COUNT,
    PERSONA_FILE,
    TWEETS_FILE,
    SIMULATION_LOG_FILE,
    SENTIMENT_PLOT_FILE,
    ensure_directories,
)
from .simulation import Simulation
from .predictor import Predictor
from .visualizer import plot_results, generate_summary_report


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application.
    
    Args:
        verbose: If True, set DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="simons_heir",
        description="Simons' Heir: Social simulation platform for collective sentiment analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--agents", "-a",
        type=int,
        default=AGENTS_COUNT,
        help="Number of agents in the simulation",
    )
    
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=SIMULATION_DAYS,
        help="Number of days to simulate",
    )
    
    parser.add_argument(
        "--persona-file",
        type=Path,
        default=PERSONA_FILE,
        help="Path to persona JSON file",
    )
    
    parser.add_argument(
        "--tweets-file",
        type=Path,
        default=TWEETS_FILE,
        help="Path to seed tweets CSV file",
    )
    
    parser.add_argument(
        "--output-log",
        type=Path,
        default=SIMULATION_LOG_FILE,
        help="Path for simulation log output",
    )
    
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=SENTIMENT_PLOT_FILE,
        help="Path for sentiment plot output",
    )
    
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="Skip simulation and only run analysis on existing log",
    )
    
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip analysis and visualization",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Run in mock mode (no LLM connection)",
    )
    
    return parser.parse_args()


def run_simulation(args: argparse.Namespace, logger: logging.Logger) -> list[dict]:
    """Execute the simulation phase.
    
    Args:
        args: Parsed command-line arguments.
        logger: Logger instance.
        
    Returns:
        List of action dictionaries from simulation.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: SIMULATION")
    logger.info("=" * 60)
    
    sim = Simulation(
        days=args.days,
        agent_count=args.agents,
        persona_file=args.persona_file,
        tweets_file=args.tweets_file,
        mock_llm=args.mock_llm,
    )
    
    sim.setup()
    
    results = sim.run()
    
    logger.info(f"Simulation complete: {len(results)} actions recorded")
    
    return results


def run_analysis(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Execute the analysis and visualization phase.
    
    Args:
        args: Parsed command-line arguments.
        logger: Logger instance.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: ANALYSIS")
    logger.info("=" * 60)
    
    predictor = Predictor()
    
    analysis_df = predictor.analyze(log_file=args.output_log)
    
    summary_stats = predictor.get_summary_stats(analysis_df)
    
    logger.info("Analysis Summary:")
    logger.info(f"  Time periods: {summary_stats['time_periods']}")
    logger.info(f"  Total tweets: {summary_stats['total_tweets']}")
    logger.info(f"  Avg sentiment: {summary_stats['avg_sentiment']:.4f}")
    logger.info(f"  Peak activity: {summary_stats['peak_activity_hour']}")
    
    logger.info("=" * 60)
    logger.info("PHASE 3: VISUALIZATION")
    logger.info("=" * 60)
    
    plot_path = plot_results(analysis_df, args.output_plot)
    logger.info(f"Sentiment plot saved: {plot_path}")
    
    report_path = generate_summary_report(analysis_df, summary_stats)
    logger.info(f"Analysis report saved: {report_path}")


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    args = parse_arguments()
    
    setup_logging(args.verbose)
    logger = logging.getLogger("simons_heir")
    
    logger.info("=" * 60)
    logger.info("SIMONS' HEIR - Social Simulation Platform")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  Agents: {args.agents}")
    logger.info(f"  Days: {args.days}")
    logger.info(f"  Persona file: {args.persona_file}")
    logger.info(f"  Output log: {args.output_log}")
    logger.info(f"  Output plot: {args.output_plot}")
    
    try:
        ensure_directories()
        
        if not args.skip_simulation:
            run_simulation(args, logger)
        else:
            logger.info("Skipping simulation (--skip-simulation flag)")
        
        if not args.skip_analysis:
            if not args.output_log.exists():
                logger.error(f"Log file not found: {args.output_log}")
                logger.error("Run simulation first or check the file path.")
                return 1
            run_analysis(args, logger)
        else:
            logger.info("Skipping analysis (--skip-analysis flag)")
        
        logger.info("=" * 60)
        logger.info("ALL PHASES COMPLETE")
        logger.info("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
