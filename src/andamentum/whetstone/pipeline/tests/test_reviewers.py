"""Reviewer profiles: the registry, the fit predicate, and precedence.

Three guarantees are worth more than the rest of this file put together:

* ``resolve(profile=general, …)`` reproduces exactly what the pipelines
  configured before profiles existed, for every document type. This is the
  no-regression test.
* Every criterion a profile holds ``is`` a registry constant, so no profile can
  ship a second wording of a question that already has one.
* The full profile x document-type fit matrix is a golden table, so any change
  to compatibility is a visible diff rather than a silent behaviour change.
"""

from __future__ import annotations

import dataclasses
import logging
import re

import pytest

from andamentum.whetstone.pipeline import criteria as registry
from andamentum.whetstone.pipeline.checklist.bank import (
    ALL_PROBE_IDS,
    GENERAL_PROBES,
    TUTORIAL_PROBES,
    probe_bank_for,
    probe_for,
)
from andamentum.whetstone.pipeline.criteria import Criterion, criterion_set_for
from andamentum.whetstone.pipeline.editor import DEFAULT_EDITOR_CRITERIA
from andamentum.whetstone.pipeline.reviewers import (
    ALL_DOCUMENT_TYPES,
    DEFAULT_PROFILE_NAME,
    PROFILES,
    UNRESOLVED_DOCUMENT_TYPE,
    ResolvedReviewer,
    ReviewerProfile,
    fits,
    resolve,
)
from andamentum.whetstone.renderers.docx import DEFAULT_AI_AUTHOR

DOCUMENT_TYPES = (
    "academic",
    "external_communication",
    "essay",
    "tutorial",
    "creative",
    "general",
)


def _registry_criteria_by_identity() -> dict[int, str]:
    """Every module-level ``Criterion`` in ``criteria.py``, keyed by id()."""
    return {
        id(value): name
        for name, value in vars(registry).items()
        if isinstance(value, Criterion)
    }


# ── The registry itself ─────────────────────────────────────────────────────


def test_seven_profiles_in_display_order():
    assert list(PROFILES) == [
        "default",
        "claims",
        "methods",
        "framing",
        "reader",
        "adversary",
        "editor",
    ]


def test_the_default_profile_is_named_default_and_comes_first():
    """Named ``default``, not ``general``, because ``general`` is also a
    DOCUMENT TYPE — ``--reviewer general --document-type general`` was legal
    and meant two unrelated things with one word."""
    assert DEFAULT_PROFILE_NAME == "default"
    assert next(iter(PROFILES)) == DEFAULT_PROFILE_NAME


def test_profile_names_are_unique_and_usable_as_argument_and_filename():
    names = [p.name for p in PROFILES.values()]
    assert len(names) == len(set(names))
    for name in names:
        assert re.fullmatch(r"[a-z]+", name), name
        assert name == PROFILES[name].name


def test_every_profile_key_matches_its_name():
    for key, profile in PROFILES.items():
        assert key == profile.name


def test_profiles_mapping_is_read_only():
    with pytest.raises(TypeError):
        PROFILES["intruder"] = PROFILES["claims"]  # type: ignore[index]


def test_profiles_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        PROFILES["claims"].name = "other"  # type: ignore[misc]


def test_identity_description_and_brief_are_non_empty_except_the_default():
    for profile in PROFILES.values():
        assert profile.identity.strip()
        assert profile.description.strip()
        if profile.name == DEFAULT_PROFILE_NAME:
            assert profile.brief == ""
        else:
            assert profile.brief.strip()


def test_general_selects_nothing_so_every_consumer_falls_through():
    general = PROFILES[DEFAULT_PROFILE_NAME]
    assert general.criteria is None
    assert general.probes is None
    assert general.editor_criteria is None
    assert general.fits == ALL_DOCUMENT_TYPES


def test_only_the_editor_skips_a_pass():
    for profile in PROFILES.values():
        if profile.name == "editor":
            assert profile.criteria == ()
            assert profile.probes == ()
        else:
            assert profile.criteria != ()
            assert profile.probes != ()


# ── Criteria are references, not copies ─────────────────────────────────────


def test_every_profile_criterion_is_a_registry_constant():
    known = _registry_criteria_by_identity()
    for profile in PROFILES.values():
        for criterion in profile.criteria or ():
            assert id(criterion) in known, (
                f"{profile.name} holds a Criterion that is not a module-level "
                f"constant in criteria.py: {criterion.name!r}"
            )


def test_the_twelve_profile_criteria_are_all_selected_by_some_profile():
    selected = {
        id(c) for profile in PROFILES.values() for c in (profile.criteria or ())
    }
    in_a_type_set = {id(c) for t in DOCUMENT_TYPES for c in criterion_set_for(t)}
    registry_ids = set(_registry_criteria_by_identity())
    orphans = registry_ids - selected - in_a_type_set
    assert not orphans, (
        "criteria.py holds criteria that neither a document type nor a "
        "reviewer profile selects"
    )


def test_no_profile_criterion_is_also_in_that_document_type_default():
    """A profile replaces the type set, so an overlap would be two wordings.

    Not a correctness failure in itself — it is the one-wording-per-question
    rule, checked where it is cheapest to check.
    """
    for profile in PROFILES.values():
        for document_type in sorted(profile.fits):
            default_ids = {id(c) for c in criterion_set_for(document_type)}
            for criterion in profile.criteria or ():
                assert id(criterion) not in default_ids


# ── Probes are ids, resolved per document type ──────────────────────────────


def test_every_profile_probe_id_is_a_shipped_id():
    for profile in PROFILES.values():
        for probe_id in profile.probes or ():
            assert probe_id in ALL_PROBE_IDS, (
                f"{profile.name} names probe {probe_id!r}, which no bank ships"
            )


def test_every_profile_probe_resolves_on_every_type_it_fits():
    for profile in PROFILES.values():
        for document_type in sorted(profile.fits):
            for probe_id in profile.probes or ():
                probe = probe_for(probe_id, document_type=document_type)
                assert probe.id == probe_id


def test_no_profile_selects_the_open_slot():
    """The catch-all names the other core probes and says not to report those.

    Inside a narrowed bank it would be told to suppress exactly the defect
    classes the profile dropped, which is the opposite of insurance.
    """
    for profile in PROFILES.values():
        assert "open_slot" not in (profile.probes or ())


def test_probe_for_prefers_the_document_type_variant():
    academic = probe_for("unmotivated_choice", document_type="academic")
    general = probe_for("unmotivated_choice", document_type="general")
    assert academic.instruction != general.instruction
    assert academic is probe_bank_for("academic")[0]


def test_probe_for_reaches_the_cross_type_variants():
    order_dependency = probe_for("order_dependency", document_type="academic")
    assert order_dependency.id == "order_dependency"
    # It lives in the tutorial bank and nowhere else, and resolves to the
    # same object whatever type asks for it.
    assert order_dependency is probe_for("order_dependency", document_type="general")
    assert order_dependency in TUTORIAL_PROBES


def test_probe_for_falls_back_to_the_general_bank_on_an_unknown_type():
    probe = probe_for("unmotivated_choice", document_type="not-a-type")
    assert probe is GENERAL_PROBES[0]


def test_probe_for_raises_on_an_unknown_id():
    with pytest.raises(ValueError, match="unknown probe id"):
        probe_for("no_such_probe", document_type="academic")


def test_all_probe_ids_covers_every_bank():
    for document_type in DOCUMENT_TYPES:
        for probe in probe_bank_for(document_type):
            assert probe.id in ALL_PROBE_IDS


# ── Editor criteria ─────────────────────────────────────────────────────────


def test_editor_criteria_are_single_lowercase_words():
    for profile in PROFILES.values():
        for word in profile.editor_criteria or ():
            assert word and word == word.lower() and " " not in word


def test_an_empty_editor_criteria_tuple_is_refused_at_construction():
    with pytest.raises(ValueError, match="editor_criteria"):
        ReviewerProfile(
            name="broken",
            identity="Broken",
            description="…",
            brief="",
            criteria=None,
            probes=None,
            editor_criteria=(),
        )


# ── The fit matrix ──────────────────────────────────────────────────────────

# Golden table: the full profile x document-type product, written out cell by
# cell in the order of the header below. Every cell is a literal on purpose —
# a row spelled `set(DOCUMENT_TYPES)` would make a change to compatibility
# invisible in the diff, which is the only reason this table exists rather
# than a comprehension over `profile.fits`.
_GOLDEN_FITS_TYPES = (
    "academic",
    "external_communication",
    "essay",
    "tutorial",
    "creative",
    "general",
)

# fmt: off
_GOLDEN_FITS: tuple[tuple[str, bool, bool, bool, bool, bool, bool], ...] = (
    # profile        academic  external_c…  essay   tutorial  creative  general
    ("default",      True,     True,        True,   True,     True,     True),
    ("claims",       True,     True,        True,   True,     False,    True),
    ("methods",      True,     False,       False,  True,     False,    True),
    ("framing",      True,     True,        True,   True,     True,     True),
    ("reader",       True,     True,        True,   True,     True,     True),
    ("adversary",    True,     True,        True,   True,     True,     True),
    ("editor",       True,     True,        True,   True,     True,     True),
)
# fmt: on


def test_the_golden_table_covers_the_whole_product():
    """The table is only a golden table if nothing can slip out of it."""
    assert _GOLDEN_FITS_TYPES == DOCUMENT_TYPES
    assert [row[0] for row in _GOLDEN_FITS] == list(PROFILES)
    for row in _GOLDEN_FITS:
        assert len(row) == 1 + len(_GOLDEN_FITS_TYPES), row[0]


def test_fits_over_the_full_product_matches_the_golden_table():
    for name, *cells in _GOLDEN_FITS:
        profile = PROFILES[name]
        for document_type, expected in zip(_GOLDEN_FITS_TYPES, cells):
            assert fits(profile=profile, document_type=document_type) is expected, (
                f"{name} x {document_type}"
            )


def test_every_declared_type_is_a_known_document_type():
    for profile in PROFILES.values():
        assert profile.fits <= ALL_DOCUMENT_TYPES, profile.name
        assert profile.fits, profile.name


def test_fits_is_false_for_an_unrecognised_document_type():
    assert not fits(profile=PROFILES["default"], document_type="not-a-type")


# ── resolve: the no-regression case ─────────────────────────────────────────


@pytest.mark.parametrize("document_type", DOCUMENT_TYPES)
def test_general_resolves_to_todays_configuration(document_type: str):
    resolved = resolve(
        profile=PROFILES[DEFAULT_PROFILE_NAME], document_type=document_type
    )
    assert resolved.criteria == criterion_set_for(document_type)
    assert resolved.probes == probe_bank_for(document_type)
    assert resolved.editor_criteria == DEFAULT_EDITOR_CRITERIA
    assert resolved.author == DEFAULT_AI_AUTHOR
    assert resolved.brief == ""
    assert resolved.fit_warning == ""
    assert resolved.header is None


def test_general_resolves_criteria_to_the_registry_constants_themselves():
    resolved = resolve(profile=PROFILES["default"], document_type="academic")
    assert resolved.criteria is not None
    for got, expected in zip(resolved.criteria, criterion_set_for("academic")):
        assert got is expected


# ── resolve: the profile case ───────────────────────────────────────────────


def test_a_profile_supplies_its_own_criteria_and_probes():
    resolved = resolve(profile=PROFILES["methods"], document_type="academic")
    assert resolved.criteria == [
        registry.REPRODUCIBILITY,
        registry.MOTIVATED_CHOICES,
        registry.EVALUATION_ADEQUACY,
    ]
    assert [p.id for p in resolved.probes] == [
        "unmotivated_choice",
        "missing_to_act",
        "order_dependency",
    ]
    assert resolved.editor_criteria == DEFAULT_EDITOR_CRITERIA
    assert resolved.brief == PROFILES["methods"].brief


def test_probe_order_follows_the_profile_not_the_bank():
    resolved = resolve(profile=PROFILES["reader"], document_type="tutorial")
    assert [p.id for p in resolved.probes] == list(PROFILES["reader"].probes or ())


def test_the_editor_profile_resolves_to_an_empty_criterion_set():
    resolved = resolve(profile=PROFILES["editor"], document_type="academic")
    assert resolved.criteria == []
    assert resolved.probes == []
    assert resolved.editor_criteria == [
        "clarity",
        "concision",
        "grammar",
        "precision",
        "hedging",
    ]


def test_resolved_criteria_are_a_fresh_list_of_the_same_objects():
    profile = PROFILES["claims"]
    resolved = resolve(profile=profile, document_type="academic")
    assert resolved.criteria is not None
    assert resolved.criteria == list(profile.criteria or ())
    for got, expected in zip(resolved.criteria, profile.criteria or ()):
        assert got is expected
    resolved.criteria.append(registry.CONTRIBUTION)
    assert len(profile.criteria or ()) == 3


# ── resolve: precedence, field by field ─────────────────────────────────────


def test_inline_criteria_replace_the_profiles_and_leave_the_rest():
    mine = [Criterion(name="Mine", questions=["?"])]
    resolved = resolve(
        profile=PROFILES["methods"], document_type="academic", criteria=mine
    )
    assert resolved.criteria == mine
    assert [p.id for p in resolved.probes] == list(PROFILES["methods"].probes or ())


def test_guidelines_defer_the_criteria_and_beat_the_profile():
    resolved = resolve(
        profile=PROFILES["methods"],
        document_type="academic",
        guidelines_text="be kind",
    )
    assert resolved.criteria is None
    line = _line(resolved, "criteria")
    assert line.source == "supplied guidelines"


def test_inline_criteria_beat_guidelines_text():
    mine = [Criterion(name="Mine", questions=["?"])]
    resolved = resolve(
        profile=PROFILES["methods"],
        document_type="academic",
        criteria=mine,
        guidelines_text="be kind",
    )
    assert resolved.criteria == mine


def test_inline_probes_replace_the_profiles_and_leave_the_rest():
    mine = list(probe_bank_for("creative"))
    resolved = resolve(
        profile=PROFILES["methods"], document_type="academic", probes=mine
    )
    assert resolved.probes == mine
    assert resolved.criteria == list(PROFILES["methods"].criteria or ())


def test_inline_editor_criteria_replace_the_profiles():
    resolved = resolve(
        profile=PROFILES["claims"],
        document_type="academic",
        editor_criteria=["tone"],
    )
    assert resolved.editor_criteria == ["tone"]
    assert resolved.criteria == list(PROFILES["claims"].criteria or ())


def test_inline_author_replaces_the_profiles():
    resolved = resolve(
        profile=PROFILES["claims"],
        document_type="academic",
        patch_author="Reviewer 1",
    )
    assert resolved.author == "Reviewer 1"


# ── resolve: the tracked-change author ──────────────────────────────────────


def test_the_default_profile_keeps_the_package_author_verbatim():
    resolved = resolve(profile=PROFILES["default"], document_type="academic")
    assert resolved.author == DEFAULT_AI_AUTHOR


@pytest.mark.parametrize("name", [n for n in PROFILES if n != DEFAULT_PROFILE_NAME])
def test_every_specialist_author_carries_the_machine_marker(name: str):
    resolved = resolve(profile=PROFILES[name], document_type="general")
    assert resolved.author.startswith(PROFILES[name].identity)
    assert resolved.author.endswith("(andamentum-whetstone, AI)")


def test_specialist_authors_are_distinct_from_each_other():
    authors = {
        resolve(profile=p, document_type="general").author for p in PROFILES.values()
    }
    assert len(authors) == len(PROFILES)


# ── resolve: the fit warning ────────────────────────────────────────────────


def test_a_fitting_reviewer_warns_about_nothing():
    resolved = resolve(profile=PROFILES["methods"], document_type="academic")
    assert resolved.fit_warning == ""
    assert resolved.header is not None
    assert resolved.header.fit_warning == ""


def test_a_mismatched_reviewer_proceeds_with_one_warning():
    resolved = resolve(profile=PROFILES["methods"], document_type="creative")
    assert "methods reviewer" in resolved.fit_warning
    assert "creative" in resolved.fit_warning
    assert "--document-type" in resolved.fit_warning
    # It proceeds: the configuration is complete, not blank.
    assert resolved.criteria == list(PROFILES["methods"].criteria or ())
    assert resolved.header is not None
    assert resolved.header.fit_warning == resolved.fit_warning


# ── resolve: the header ─────────────────────────────────────────────────────


def _line(resolved, label: str):
    matches = [c for c in resolved.configuration if c.label == label]
    assert len(matches) == 1, label
    return matches[0]


def test_the_default_reviewer_still_reports_its_configuration():
    """No header in the report, but the command line still has to print it."""
    resolved = resolve(profile=PROFILES["default"], document_type="academic")
    assert resolved.header is None
    assert [c.label for c in resolved.configuration] == [
        "criteria",
        "probes",
        "editor criteria",
        "author",
    ]
    assert _line(resolved, "criteria").source == "document type academic"
    assert _line(resolved, "probes").source == "document type academic"
    assert _line(resolved, "author").source == "package default"


def test_the_header_and_the_resolution_report_the_same_configuration():
    resolved = resolve(profile=PROFILES["reader"], document_type="essay")
    assert resolved.header is not None
    assert resolved.header.configuration == resolved.configuration


def test_the_header_names_every_resolved_value_and_its_source():
    resolved = resolve(profile=PROFILES["claims"], document_type="academic")
    assert resolved.header is not None
    assert [c.label for c in resolved.header.configuration] == [
        "criteria",
        "probes",
        "editor criteria",
        "author",
    ]
    assert _line(resolved, "criteria").source == "reviewer claims"
    assert _line(resolved, "probes").source == "reviewer claims"
    assert _line(resolved, "editor criteria").source == "reviewer claims"
    assert _line(resolved, "author").source == "reviewer claims"
    assert "Evidence support" in _line(resolved, "criteria").value


def test_header_sources_distinguish_the_profile_from_an_inline_value():
    resolved = resolve(
        profile=PROFILES["claims"],
        document_type="academic",
        patch_author="Reviewer 1",
        editor_criteria=["tone"],
    )
    assert _line(resolved, "author").source == "supplied inline"
    assert _line(resolved, "editor criteria").source == "supplied inline"
    assert _line(resolved, "criteria").source == "reviewer claims"


def test_a_profile_that_falls_through_names_the_document_type():
    resolved = resolve(profile=PROFILES["methods"], document_type="academic")
    assert _line(resolved, "editor criteria").source == "editor default"


def test_an_empty_selection_reads_as_none_not_as_a_blank():
    resolved = resolve(profile=PROFILES["editor"], document_type="academic")
    assert _line(resolved, "criteria").value == "none"
    assert _line(resolved, "probes").value == "none"


def test_the_header_carries_the_package_version():
    from andamentum import __version__

    resolved = resolve(profile=PROFILES["reader"], document_type="essay")
    assert resolved.header is not None
    assert resolved.header.version == __version__


def test_the_header_repeats_the_profiles_own_words():
    profile = PROFILES["adversary"]
    resolved = resolve(profile=profile, document_type="essay")
    assert resolved.header is not None
    assert resolved.header.identity == profile.identity
    assert resolved.header.description == profile.description
    assert resolved.header.brief == profile.brief == resolved.brief


# ── resolve: before the document type is known ──────────────────────────────
#
# "auto" is a real value on the way in, and the CLI's default. Every value
# that needs the type has to say it does not have one rather than guess, and
# a profile's probe *ids* need it as much as the bank does: the same id names
# a differently worded probe on different types.


def test_an_undetermined_type_yields_no_criteria_and_says_so():
    resolved = resolve(
        profile=PROFILES["default"], document_type=UNRESOLVED_DOCUMENT_TYPE
    )
    assert resolved.criteria == []
    assert _line(resolved, "criteria").source == "not determined"


def test_an_undetermined_type_yields_no_probes_and_says_so():
    resolved = resolve(
        profile=PROFILES["default"], document_type=UNRESOLVED_DOCUMENT_TYPE
    )
    assert resolved.probes == []
    assert _line(resolved, "probes").source == "not determined"


@pytest.mark.parametrize(
    "name", ["claims", "methods", "framing", "reader", "adversary"]
)
def test_a_profiles_probe_ids_are_not_resolved_before_the_type_is(name: str):
    """The ids are type-relative, so resolving early answers the wrong question.

    Resolving them against ``"auto"`` would go through ``probe_bank_for``'s
    unknown-type fallback, quietly select the general bank and log a warning
    per id — on every review run under automatic classification, which is the
    command line's default.
    """
    resolved = resolve(profile=PROFILES[name], document_type=UNRESOLVED_DOCUMENT_TYPE)
    assert resolved.probes == []
    assert _line(resolved, "probes").source == "not determined"


def test_resolving_before_the_type_is_known_logs_nothing(caplog):
    with caplog.at_level(logging.WARNING):
        for profile in PROFILES.values():
            resolve(profile=profile, document_type=UNRESOLVED_DOCUMENT_TYPE)
    assert caplog.records == []


def test_the_type_is_still_honoured_once_it_is_known():
    """The guard above must not have turned profile probe selection off."""
    resolved = resolve(profile=PROFILES["reader"], document_type="academic")
    assert [p.id for p in resolved.probes] == list(PROFILES["reader"].probes or ())


def test_an_undetermined_type_warns_about_no_fit():
    """Nothing is known to be a miss when the type is not known at all."""
    resolved = resolve(
        profile=PROFILES["methods"], document_type=UNRESOLVED_DOCUMENT_TYPE
    )
    assert resolved.fit_warning == ""


# ── resolve: an empty inline editor criteria list ───────────────────────────


def test_an_empty_inline_editor_criteria_list_reads_as_not_supplied():
    """It did before this function existed, and the profile field's own
    refusal of ``()`` cites that coercion as its reason.

    The editor has no "run with no criteria" state: an empty list reaching its
    prompt is replaced there by a placeholder and it proposes edits anyway.
    So the only readings available are "the default" and "these" — and an
    empty list is not "these".
    """
    assert resolve(
        profile=PROFILES["default"], document_type="academic", editor_criteria=[]
    ).editor_criteria == list(DEFAULT_EDITOR_CRITERIA)


def test_an_empty_inline_editor_criteria_list_leaves_the_profiles_alone():
    resolved = resolve(
        profile=PROFILES["claims"], document_type="academic", editor_criteria=[]
    )
    assert resolved.editor_criteria == list(PROFILES["claims"].editor_criteria or ())
    assert _line(resolved, "editor criteria").source == "reviewer claims"


# ── the resolved configuration is keyword-only too ──────────────────────────


def test_a_resolved_reviewer_cannot_be_built_positionally():
    """Its three strings — author, brief, fit warning — are transposable.

    A swap would put the reviewer's stance in the Word margin and the author
    name into every prompt, and a type checker would see nothing.
    """
    assert all(field.kw_only for field in dataclasses.fields(ResolvedReviewer))
    with pytest.raises(TypeError):
        ResolvedReviewer([], [], [], "author", "brief", [], None, "")  # type: ignore[misc]
