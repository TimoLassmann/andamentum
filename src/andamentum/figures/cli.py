"""CLI entry point for andamentum-figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="andamentum-figures",
        description="Publication-quality scientific figure rendering — deterministic plotting of your data with journal-matched sizing.",
    )
    from andamentum import __version__ as _ver

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s (andamentum {_ver})",
    )
    sub = parser.add_subparsers(dest="command")

    # Help strings come from the registries so they cannot drift from them.
    from .palettes import ANDAMENTUM_TONE_ALIASES, ANDAMENTUM_TONES, PALETTES
    from .standards import JOURNAL_PRESETS
    from .types import PlotKind

    kinds = ", ".join(k.value for k in PlotKind)
    palettes = ", ".join(PALETTES)
    journals = ", ".join(JOURNAL_PRESETS)
    tones = ", ".join(list(ANDAMENTUM_TONES) + list(ANDAMENTUM_TONE_ALIASES))

    # plot subcommand
    plot = sub.add_parser("plot", help="Generate a figure from a data file")
    plot.add_argument("data_file", help="Path to CSV or JSON data file")
    plot.add_argument("--kind", default="auto", help=f"Plot type ({kinds})")
    plot.add_argument("--x", default=None, help="X-axis column name")
    plot.add_argument("--y", default=None, nargs="+", help="Y-axis column name(s)")
    plot.add_argument(
        "--group",
        default=None,
        help=(
            "Series column. With --kind bar: one bar per series side by side. "
            "With --kind stacked: one segment per series."
        ),
    )
    plot.add_argument(
        "--group-order",
        default=None,
        help="Series order, comma-separated. Must name every series once.",
    )
    plot.add_argument("--error", default=None, help="Error column")
    plot.add_argument("--error-type", default=None, help="Error type (sem, sd, ci95)")
    plot.add_argument("--title", default=None, help="Figure title")
    plot.add_argument("--x-label", default=None, help="X-axis label")
    plot.add_argument("--y-label", default=None, help="Y-axis label")
    plot.add_argument("--style", default="npg", help=f"Color palette ({palettes})")
    plot.add_argument(
        "--series-colors",
        default=None,
        help=(
            "Map series (or bar categories) to design-system tones by name, "
            f"e.g. wrong=rust,refused=ochre,right=olive. Tones: {tones}. "
            "No hex values."
        ),
    )
    plot.add_argument(
        "--journal", default="default", help=f"Journal preset ({journals})"
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
    plot.add_argument(
        "--horizontal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Force horizontal (--horizontal) or vertical (--no-horizontal) bars. "
            "Default: auto by label count and length."
        ),
    )
    plot.add_argument(
        "--normalize",
        action="store_true",
        help="Stacked only: segments as fractions of the category total; totals still labelled as counts.",
    )
    plot.add_argument(
        "--svg-inline",
        action="store_true",
        help=(
            "Write the bare <svg> element (no XML declaration, DOCTYPE, metadata, "
            "width or height) so a report can inline it. Output must be .svg."
        ),
    )
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

    group_order = _split_list(args.group_order)
    series_colors = _parse_series_colors(args.series_colors)

    try:
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
            horizontal=args.horizontal,
            output=args.output,
            group_order=group_order,
            normalize=args.normalize,
            series_colors=series_colors,
            svg_inline=args.svg_inline,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Saved: {result.path}")
    print(f"Kind:  {result.kind}")
    print(f'Size:  {result.width_inches}" × {result.height_inches}"')
    if result.advisor_notes:
        print("Warnings:")
        for note in result.advisor_notes:
            print(f"  - {note}")
        print(
            "\nThe renderer auto-applied one or more visual changes "
            "(see Warnings above). Mirror these in your figure caption "
            "so reviewers can see what was decided automatically.",
            file=sys.stderr,
        )
    print(f"\nLegend:\n  {result.legend}")


def _split_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return items or None


def _parse_series_colors(value: str | None) -> dict[str, str] | None:
    """Parse ``name=tone,name=tone`` into a dict. Tone names are checked by figure()."""
    if value is None:
        return None
    mapping: dict[str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            print(
                f"Error: --series-colors entry '{item}' is not name=tone",
                file=sys.stderr,
            )
            sys.exit(1)
        name, tone = item.split("=", 1)
        mapping[name.strip()] = tone.strip()
    return mapping or None
