import pytest

from voicestress.domain.value_objects import BaselineProfile, ProsodyFeatures, SpeakerId


def _features(f0_mean=180.0, f0_std=20.0, jitter=1.0, shimmer=5.0, hnr=15.0, rate=4.0):
    return ProsodyFeatures(
        f0_mean_hz=f0_mean,
        f0_std_hz=f0_std,
        jitter_local_pct=jitter,
        shimmer_local_pct=shimmer,
        hnr_db=hnr,
        speech_rate_syll_s=rate,
    )


def _reliable_profile() -> BaselineProfile:
    return BaselineProfile(
        speaker=SpeakerId(corpus="interview", raw_id="candidate_1"),
        means={
            "f0_mean_hz": 180.0,
            "f0_std_hz": 20.0,
            "jitter_local_pct": 1.0,
            "shimmer_local_pct": 5.0,
            "hnr_db": 15.0,
            "speech_rate_syll_s": 4.0,
        },
        stds={
            "f0_mean_hz": 10.0,
            "f0_std_hz": 5.0,
            "jitter_local_pct": 0.2,
            "shimmer_local_pct": 1.0,
            "hnr_db": 2.0,
            "speech_rate_syll_s": 0.5,
        },
        n_calibration_turns=3,
    )


def test_zero_deviation_when_features_equal_baseline_mean():
    profile = _reliable_profile()
    deviation = profile.deviation(_features())
    for z in deviation.values():
        assert z == pytest.approx(0.0, abs=1e-9)


def test_deviation_sign_and_magnitude():
    profile = _reliable_profile()
    # f0_mean 200 vs baseline mean 180, std 10 -> z = +2.0
    deviation = profile.deviation(_features(f0_mean=200.0))
    assert deviation["f0_mean_z"] == pytest.approx(2.0)


def test_below_minimum_turns_is_not_reliable():
    profile = BaselineProfile(
        speaker=SpeakerId(corpus="interview", raw_id="c1"),
        means={"f0_mean_hz": 180.0},
        stds={"f0_mean_hz": 10.0},
        n_calibration_turns=1,
    )
    assert profile.is_reliable is False
    with pytest.raises(ValueError):
        profile.deviation(_features())


def test_degenerate_std_is_not_reliable():
    profile = BaselineProfile(
        speaker=SpeakerId(corpus="interview", raw_id="c1"),
        means={"f0_mean_hz": 180.0},
        stds={"f0_mean_hz": 0.0},  # zero variance in calibration window
        n_calibration_turns=5,
    )
    assert profile.is_reliable is False


def test_reliable_profile_with_enough_varied_turns():
    assert _reliable_profile().is_reliable is True
