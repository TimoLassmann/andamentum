"""CLI entry point for mosaic-figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mosaic-figures",
        description="Publication-quality scientific figure generation",
    )
    sub = parser.add_subparsers(dest="command")

    # plot subcommand
    plot = sub.add_parser("plot", help="Generate a figure from a data file")
    plot.add_argument("data_file", help="Path to CSV or JSON data file")
    plot.add_argument(
        "--kind",
        default="auto",
        help="Plot type (bar, line, scatter, box, violin, histogram, heatmap, strip, swarm, auto)",
    )
    plot.add_argument("--x", default=None, help="X-axis column name")
    plot.add_argument("--y", default=None, nargs="+", help="Y-axis column name(s)")
    plot.add_argument("--group", default=None, help="Grouping column")
    plot.add_argument("--error", default=None, help="Error column")
    plot.add_argument("--error-type", default=None, help="Error type (sem, sd, ci95)")
    plot.add_argument("--title", default=None, help="Figure title")
    plot.add_argument("--x-label", default=None, help="X-axis label")
    plot.add_argument("--y-label", default=None, help="Y-axis label")
    plot.add_argument(
        "--style",
        default="npg",
        help="Color palette (npg, nejm, lancet, jama, aaas, d3, okabe_ito)",
    )
    plot.add_argument(
        "--journal",
        default="default",
        help="Journal preset (default, nature, science, cell, plos)",
    )
    plot.add_argument(
        "--mode",
        default="publication",
        choices=["publication", "showcase"],
        help="Output mode",
    )
    plot.add_argument(
        "--width", default="single", help="Width (single, 1.5, double, or inches)"
    )
    plot.add_argument("--dpi", type=int, default=300, help="Resolution")
    plot.add_argument("-o", "--output", default="figure.pdf", help="Output file path")

    # palettes subcommand
    sub.add_parser("palettes", help="List available color palettes")

    # journals subcommand
    sub.add_parser("journals", help="List available journal format presets")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "palettes":
        from .palettes import PALETTES

        for name, colors in PALETTES.items():
            print(
                f"  {name:12s} ({len(colors)} colors): {', '.join(colors[:5])}{'...' if len(colors) > 5 else ''}"
            )
        return

    if args.command == "journals":
        from .standards import list_presets

        for name, desc in list_presets().items():
            print(f"  {name:12s} {desc}")
        return

    if args.command == "plot":
        _run_plot(args)
        return


def _run_plot(args: argparse.Namespace) -> None:
    from .render import figure

    data_path = Path(args.data_file)
    if not data_path.exists():
        print(f"Error: File not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # Load data
    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        data: dict | list | str = data_path.read_text()
    elif suffix == ".json":
        with open(data_path) as f:
            data = json.load(f)
    else:
        print(
            f"Error: Unsupported file type: {suffix}. Use .csv or .json",
            file=sys.stderr,
        )
        sys.exit(1)

    # Parse width
    width: str | float = args.width
    try:
        width = float(args.width)
    except ValueError:
        pass

    # Parse y columns
    y = args.y
    if y and len(y) == 1:
        y = y[0]

    result = figure(
        data,
        kind=args.kind,
        x=args.x,
        y=y,
        group=args.group,
        error=args.error,
        error_type=args.error_type,
        title=args.title,
        x_label=args.x_label,
        y_label=args.y_label,
        style=args.style,
        journal=args.journal,
        mode=args.mode,
        width=width,
        dpi=args.dpi,
        output=args.output,
    )

    print(f"Saved: {result.path}")
    print(f"Kind:  {result.kind}")
    print(f'Size:  {result.width_inches}" × {result.height_inches}"')
    if result.advisor_notes:
        print("Warnings:")
        for note in result.advisor_notes:
            print(f"  - {note}")
    print(f"\nLegend:\n  {result.legend}")
