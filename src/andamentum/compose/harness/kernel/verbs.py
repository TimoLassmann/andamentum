"""Verb registry: the CLI is generated from it (MCP later)."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import Literal, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo


@dataclass(frozen=True)
class Verb:
    name: str
    summary: str
    params: type[BaseModel]
    handler: Callable[
        ..., int
    ]  # receives the validated params model, returns an exit code


_VERBS: dict[str, Verb] = {}


def register_verb(verb: Verb) -> Verb:
    if verb.name in _VERBS:
        raise ValueError(f"verb {verb.name!r} is already registered")
    _VERBS[verb.name] = verb
    return verb


def all_verbs() -> list[Verb]:
    return list(_VERBS.values())


def get_verb(name: str) -> Verb:
    try:
        return _VERBS[name]
    except KeyError:
        raise KeyError(f"unknown verb {name!r}") from None


def _clear_verbs() -> None:
    """Test helper."""
    _VERBS.clear()


def _unwrap(annotation: object) -> object:
    """Strip ``X | None`` down to ``X``."""
    args = [a for a in get_args(annotation) if a is not type(None)]
    if get_origin(annotation) in (Union, UnionType) and len(args) == 1:
        return args[0]
    return annotation


def _add_field(parser: argparse.ArgumentParser, name: str, field: FieldInfo) -> None:
    """Add one argument for a params-model field.

    A field with ``json_schema_extra={"positional": True}`` is positional; the
    rest are ``--flags`` (underscores become dashes). ``bool`` is a switch,
    ``list[X]`` repeats, ``Literal`` gives choices.
    """
    annotation = _unwrap(field.annotation)
    extra = field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
    positional = bool(extra.get("positional"))
    help_text = field.description
    required = field.is_required()
    default = field.get_default(call_default_factory=True)
    flag = "--" + name.replace("_", "-")

    if get_origin(annotation) is list:
        (item,) = get_args(annotation)
        if item not in (str, int, float, Path):
            raise ValueError(f"field {name!r}: unsupported list item type {item!r}")
        if positional:
            parser.add_argument(
                name, nargs="+" if required else "*", type=item, help=help_text
            )
        else:
            parser.add_argument(
                flag, dest=name, action="append", type=item, default=[], help=help_text
            )
    elif annotation is bool:
        if positional:
            raise ValueError(f"field {name!r}: a bool cannot be positional")
        parser.add_argument(
            flag, dest=name, action="store_true", default=bool(default), help=help_text
        )
    elif get_origin(annotation) is Literal:
        choices = list(get_args(annotation))
        if positional:
            parser.add_argument(
                name,
                choices=choices,
                nargs=None if required else "?",
                default=default,
                help=help_text,
            )
        else:
            parser.add_argument(
                flag,
                dest=name,
                choices=choices,
                required=required,
                default=default,
                help=help_text,
            )
    elif annotation in (str, int, float, Path):
        kind: Callable[[str], object] = annotation  # type: ignore[assignment]
        if positional:
            parser.add_argument(
                name,
                type=kind,
                nargs=None if required else "?",
                default=default,
                help=help_text,
            )
        else:
            parser.add_argument(
                flag,
                dest=name,
                type=kind,
                required=required,
                default=default,
                help=help_text,
            )
    else:
        raise ValueError(f"field {name!r}: unsupported type {annotation!r} for the CLI")


def build_subparsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Add one argparse subcommand per registered verb, from its params model.

    The parsed namespace carries ``verb`` (the verb name); pass it to
    :func:`dispatch_verb`.
    """
    for verb in all_verbs():
        parser = subparsers.add_parser(
            verb.name, help=verb.summary, description=verb.summary
        )
        for name, field in verb.params.model_fields.items():
            _add_field(parser, name, field)
        parser.set_defaults(verb=verb.name)


def dispatch_verb(args: argparse.Namespace) -> int:
    """Validate the parsed arguments against the verb's params model and run it.

    Invalid arguments print ``error: ...`` and return 2 (usage error).
    """
    verb = get_verb(args.verb)
    values = {name: getattr(args, name) for name in verb.params.model_fields}
    try:
        params = verb.params.model_validate(values)
    except ValidationError as exc:
        print(f"error: {verb.name}: {exc}", file=sys.stderr)
        return 2
    return verb.handler(params)


# Modules that define verbs. Each exposes ``VERBS: tuple[Verb, ...]``; the
# loader registers the ones not yet registered, so it is safe to call twice
# (and after a test cleared the registry).
VERB_MODULES: tuple[str, ...] = (
    "andamentum.compose.harness.check",
    "andamentum.compose.harness.bindings.bind",
    "andamentum.compose.harness.bindings.refresh",
    "andamentum.compose.harness.respond.run",
    "andamentum.compose.harness.literature.verbs",
    "andamentum.compose.harness.draft.verb",
    "andamentum.compose.harness.mcp_server",
)


def load_builtin_verbs() -> None:
    """Register every verb the modules in ``VERB_MODULES`` define."""
    for name in VERB_MODULES:
        module = importlib.import_module(name)
        for verb in module.VERBS:
            if verb.name not in _VERBS:
                register_verb(verb)
