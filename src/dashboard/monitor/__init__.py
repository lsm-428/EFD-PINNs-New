"""Dashboard monitor module for training log analysis.

This module provides non-invasive incremental log parsing capability.

Usage:
    from src.dashboard.monitor import LogParser, parse_training_log

    parser = LogParser()
    records = parser.parse("outputs/train/pinn_*/training.log")

    # Or use the simple function
    records = parse_training_log("outputs/train/pinn_*/training.log")
"""

__all__ = [
    "LOSS_COLORS",
    "PATTERN_LR",
    "PATTERN_MAIN",
    "TAG_MAP",
    "LogParser",
    "analyze_rmse_per_voltage",
    "analyze_volume_trend",
    "find_log_path",
    "generate_html_report",
    "parse_training_log",
    "plot_learning_curve",
    "plot_loss_components",
    "save_csv",
    "summarize_tail",
]


def __getattr__(name: str):
    if name == "LogParser":
        from .log_parser import LogParser

        return LogParser
    if name in (
        "TAG_MAP",
        "PATTERN_MAIN",
        "PATTERN_LR",
        "find_log_path",
        "parse_training_log",
        "save_csv",
    ):
        from .log_parsing import (
            PATTERN_LR,
            PATTERN_MAIN,
            TAG_MAP,
            find_log_path,
            parse_training_log,
            save_csv,
        )

        return locals()[name]
    if name in ("summarize_tail", "analyze_volume_trend", "analyze_rmse_per_voltage"):
        from .performance_metrics import (
            analyze_rmse_per_voltage,
            analyze_volume_trend,
            summarize_tail,
        )

        return locals()[name]
    if name in (
        "plot_loss_components",
        "plot_learning_curve",
        "generate_html_report",
        "LOSS_COLORS",
    ):
        from .visualization_output import (
            LOSS_COLORS,
            generate_html_report,
            plot_learning_curve,
            plot_loss_components,
        )

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
