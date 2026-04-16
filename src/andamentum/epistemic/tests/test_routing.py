"""Tests for question-type routing configuration."""

import pytest
from ..routing import (
    TrackActivation,
    RoutingProfile,
    ROUTING_TABLE,
    get_routing_profile,
    get_active_tracks,
)


EXPECTED_TYPES = {
    "verificatory",
    "explanatory",
    "exploratory",
    "comparative",
    "predictive",
    "compositional",
    "normative",
}


class TestRoutingTable:
    def test_has_all_seven_types(self):
        assert set(ROUTING_TABLE.keys()) == EXPECTED_TYPES

    def test_routing_profile_structure(self):
        profile = ROUTING_TABLE["verificatory"]
        assert isinstance(profile, RoutingProfile)
        assert isinstance(profile.tracks, dict)
        assert isinstance(profile.gate_thresholds, dict)

    def test_track_activation_values(self):
        valid = set(TrackActivation)
        for qt_name, profile in ROUTING_TABLE.items():
            for track_name, activation in profile.tracks.items():
                assert activation in valid, (
                    f"{qt_name}.{track_name} has invalid activation: {activation}"
                )

    def test_all_profiles_have_seven_tracks(self):
        expected_tracks = {
            "adversarial",
            "convergence",
            "deductive",
            "computational",
            "argument",
            "contrastive",
            "consistency",
        }
        for qt_name, profile in ROUTING_TABLE.items():
            assert set(profile.tracks.keys()) == expected_tracks, (
                f"{qt_name} missing tracks"
            )


class TestGetRoutingProfile:
    def test_returns_profile_for_valid_type(self):
        profile = get_routing_profile("verificatory")
        assert isinstance(profile, RoutingProfile)

    def test_raises_for_unknown_type(self):
        with pytest.raises(KeyError):
            get_routing_profile("nonexistent")


class TestGetActiveTracks:
    def test_verificatory_has_adversarial_primary(self):
        tracks = get_active_tracks("verificatory")
        assert tracks["adversarial"] == TrackActivation.PRIMARY

    def test_verificatory_skips_contrastive(self):
        tracks = get_active_tracks("verificatory")
        assert tracks["contrastive"] == TrackActivation.SKIP

    def test_exploratory_skips_most_tracks(self):
        tracks = get_active_tracks("exploratory")
        assert tracks["adversarial"] == TrackActivation.SKIP
        assert tracks["deductive"] == TrackActivation.SKIP
        assert tracks["consistency"] == TrackActivation.PRIMARY

    def test_explanatory_has_contrastive_primary(self):
        tracks = get_active_tracks("explanatory")
        assert tracks["contrastive"] == TrackActivation.PRIMARY
        assert tracks["deductive"] == TrackActivation.PRIMARY
        assert tracks["argument"] == TrackActivation.PRIMARY

    def test_predictive_emphasises_deductive_and_computational(self):
        tracks = get_active_tracks("predictive")
        assert tracks["deductive"] == TrackActivation.PRIMARY
        assert tracks["computational"] == TrackActivation.PRIMARY

    def test_normative_requires_consistency(self):
        tracks = get_active_tracks("normative")
        assert tracks["consistency"] == TrackActivation.PRIMARY
        assert tracks["deductive"] == TrackActivation.PRIMARY
        assert tracks["argument"] == TrackActivation.PRIMARY


class TestGateThresholds:
    def test_exploratory_has_lower_supported_bar(self):
        profile = get_routing_profile("exploratory")
        supported = profile.gate_thresholds.get("supported", {})
        verificatory_profile = get_routing_profile("verificatory")
        v_supported = verificatory_profile.gate_thresholds.get("supported", {})
        assert float(supported.get("min_evidence_weighted", 1.0)) <= float(  # type: ignore[arg-type]
            v_supported.get("min_evidence_weighted", 1.0)  # type: ignore[arg-type]
        )

    def test_predictive_requires_falsification_at_robust_not_supported(self):
        """Falsification is checked at ROBUST→ACTIONABLE, not at SUPPORTED.

        Predictions are generated at the ROBUST stage (Lakatos). Requiring
        them earlier creates a deadlock where claims can never advance.
        """
        profile = get_routing_profile("predictive")
        supported = profile.gate_thresholds.get("supported", {})
        robust = profile.gate_thresholds.get("robust", {})
        assert "requires_falsification_criteria" not in supported
        assert robust.get("requires_falsification_criteria") is True

    def test_normative_requires_fact_value_separation(self):
        profile = get_routing_profile("normative")
        supported = profile.gate_thresholds.get("supported", {})
        assert supported.get("requires_fact_value_separation") is True
