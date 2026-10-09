"""Command-line entry point: andamentum-figures.

Usage::

    andamentum-figures [plot] data.csv --kind bar --x group --y value -o figure.pdf
    andamentum-figures palettes
    andamentum-figures journals

``plot`` is the default: when the first argument is not a command or a flag it
is taken to be the data file. It writes the figure to the file named with ``-o/--out`` (the format
follows the extension: .pdf, .png, .svg) and prints a short summary, or the
full result as JSON with ``--json``. Raster output (.png) is 300 dpi; large
type for slides is ``--journal showcase``. The Python ``figure()`` keeps every
setting (``dpi``, ``mode``, ``height``, ``log_scale``, ...).

Exit codes: 0 success; 2 usage error (a wrong or missing argument, or an
option the data cannot satisfy); 3 the data file could not be read (missing,
not .csv or .json, not valid JSON); 4 the run failed (a crash, or the figure
could not be written). A figure has no negative result, so 1 is not used.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

_COMMANDS = ("plot", "palettes", "journals")

_EXIT_CODES = """\
exit codes:
  0  success
  2  usage error (a wrong or missing argument, or an option the data cannot satisfy)
  3  the data file could not be read (missing, not .csv or .json, invalid JSON)
  4  the run failed (a crash, or the figure could not be written)
"""


def build_parser() -> argparse.ArgumentParser:
    from andamentum import __version__

    parser = argparse.ArgumentParser(
        prog="andamentum-figures",
        description="Publication-quality scientific figures. Plots only the numbers "
        "in your data file; no generative AI. Journal-matched sizing. `plot` is "
        "the default command: andamentum-figures DATA --x A --y B -o FILE.svg. "
        "Also: palettes, journals (list the choices).",
        epilog="examples:\n"
        "  andamentum-figures results.csv --kind bar --x group --y value -o figure.svg\n"
        "  andamentum-figures results.csv --x group --y value --journal showcase -o slide.png\n"
        "  andamentum-figures palettes\n"
        f"{_EXIT_CODES}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"andamentum-figures {__version__}"
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
    plot = sub.add_parser(
        "plot",
        help="draw a figure from a data file",
        description="Draw a figure from a CSV or JSON table and write it to the file named with -o.",
        epilog="examples:\n"
        "  andamentum-figures plot results.csv --kind bar --x group --y value -o figure.svg\n"
        "  andamentum-figures plot results.csv --x group --y value --error sd "
        "--error-type sd -o figure.pdf\n"
        "  andamentum-figures plot results.csv --x group --y value --journal report "
        "--style andamentum --svg-inline -o figure.svg\n"
        f"{_EXIT_CODES}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    plot.add_argument(
        "data_file", metavar="DATA", help="path to a .csv or .json data file"
    )
    plot.add_argument(
        "--kind", default="auto", help=f"chart type (default: auto): {kinds}"
    )
    plot.add_argument(
        "--x", default=None, help="column for the x axis (default: detected)"
    )
    plot.add_argument(
        "--y",
        default=None,
        nargs="+",
        metavar="COLUMN",
        help="column(s) for the y axis; give several, separated by spaces, for a wide-form table",
    )
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
        metavar="A,B,C",
        help="series order, comma-separated; must name every series once",
    )
    plot.add_argument("--error", default=None, help="column holding error-bar values")
    plot.add_argument(
        "--error-type", default=None, help="what the error bars show: sem, sd or ci95"
    )
    plot.add_argument("--title", default=None, help="figure title")
    plot.add_argument("--x-label", default=None, help="x-axis label")
    plot.add_argument("--y-label", default=None, help="y-axis label")
    plot.add_argument(
        "--style", default="npg", help=f"colour palette (default: npg): {palettes}"
    )
    plot.add_argument(
        "--series-colors",
        default=None,
        metavar="NAME=TONE,...",
        help=(
            "colour series (or bar categories) by tone name, "
            f"e.g. wrong=rust,refused=ochre,right=olive. Tones: {tones}. "
            "No hex values."
        ),
    )
    plot.add_argument(
        "--journal",
        default="default",
        help=f"journal size preset (default: default): {journals}; "
        "showcase gives large type for slides, report the andamentum report look",
    )
    plot.add_argument(
        "--width",
        default="single",
        help="figure width: single, 1.5 or double (journal column widths), or a number of inches (default: single)",
    )
    plot.add_argument(
        "--horizontal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "force horizontal (--horizontal) or vertical (--no-horizontal) bars; "
            "by default the orientation follows label count and length"
        ),
    )
    plot.add_argument(
        "--normalize",
        action="store_true",
        help="stacked only: draw segments as fractions of the category total (totals are still labelled as counts)",
    )
    plot.add_argument(
        "--svg-inline",
        action="store_true",
        help=(
            "write the bare <svg> element (no XML declaration, DOCTYPE, metadata, "
            "width or height) so a report can inline it; --out must end in .svg"
        ),
    )
    plot.add_argument(
        "-o",
        "--out",
        required=True,
        metavar="FILE",
        help="file to write the figure to; the format follows the extension (.pdf, .png, .svg)",
    )
    plot.add_argument(
        "--json",
        action="store_true",
        help="print the full result (path, legend, notes, size, ...) as JSON instead of a summary",
    )

    # palettes subcommand
    sub.add_parser(
        "palettes",
        help="list the colour palettes",
        description="List the colour palettes that --style accepts.",
        epilog="example: andamentum-figures palettes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # journals subcommand
    sub.add_parser(
        "journals",
        help="list the journal size presets",
        description="List the journal size presets that --journal accepts.",
        epilog="example: andamentum-figures journals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] not in _COMMANDS and not raw[0].startswith("-"):
        raw = ["plot", *raw]
    args = parser.parse_args(raw)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "palettes":
        from .palettes import PALETTES

        for name, colors in PALETTES.items():
            print(
                f"  {name:12s} ({len(colors)} colors): {', '.join(colors[:5])}{'...' if len(colors) > 5 else ''}"
            )
        return 0

    if args.command == "journals":
        from .standards import list_presets

        for name, desc in list_presets().items():
            print(f"  {name:12s} {desc}")
        return 0

    return _run_plot(args)


def _run_plot(args: argparse.Namespace) -> int:
    from .errors import FiguresError
    from .render import figure

    data_path = Path(args.data_file)
    suffix = data_path.suffix.lower()
    if suffix not in (".csv", ".json"):
        print(
            f"error: unsupported data file type '{suffix}'; use a .csv or .json file",
            file=sys.stderr,
        )
        return 3
    try:
        data: dict | list | str
        if suffix == ".csv":
            data = data_path.read_text(encoding="utf-8")
        else:
            data = json.loads(data_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"error: data file not found: {data_path}", file=sys.stderr)
        return 3
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"error: cannot read {data_path}: {exc}", file=sys.stderr)
        return 3

    # --width is a column-width name (single, 1.5, double) or a number of inches.
    width: str | float = args.width
    try:
        width = float(args.width)
    except ValueError:
        pass

    # One --y value is a single column; several are a wide-form table.
    y = args.y
    if y and len(y) == 1:
        y = y[0]

    group_order = _split_list(args.group_order)
    try:
        series_colors = _parse_series_colors(args.series_colors)
    except FiguresError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

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
            width=width,
            dpi=300,
            horizontal=args.horizontal,
            out=args.out,
            group_order=group_order,
            normalize=args.normalize,
            series_colors=series_colors,
            svg_inline=args.svg_inline,
        )
    except FiguresError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: cannot write {args.out}: {exc}", file=sys.stderr)
        return 4

    if args.json:
        print(result.model_dump_json(indent=2))
        return 0

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
    return 0


def _split_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return items or None


def _parse_series_colors(value: str | None) -> dict[str, str] | None:
    """Parse ``name=tone,name=tone`` into a dict. Tone names are checked by figure()."""
    from .errors import FiguresError

    if value is None:
        return None
    mapping: dict[str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise FiguresError(f"--series-colors entry '{item}' is not name=tone")
        name, tone = item.split("=", 1)
        mapping[name.strip()] = tone.strip()
    return mapping or None


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
