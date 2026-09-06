from voicestress.application.baseline_service import BaselineCalibrator
from voicestress.domain.value_objects import ProsodyFeatures, SpeakerId


def _features(f0_mean: float, wobble: float = 0.0) -> ProsodyFeatures:
    """`wobble` nudges every field a little so realistic turn-to-turn variance shows up
    on all tracked features, not just f0_mean — a fixture that only varies one field
    was masking the (correct) degenerate-std guard on the other five. See the test this
    replaced, caught by test_reliable_after_enough_varied_turns failing for the right
    reason: the guard was doing its job against an unrealistic fixture."""
    return ProsodyFeatures(
        f0_mean_hz=f0_mean,
        f0_std_hz=15.0 + wobble,
        jitter_local_pct=1.0 + wobble * 0.05,
        shimmer_local_pct=5.0 + wobble * 0.2,
        hnr_db=15.0 - wobble,
        speech_rate_syll_s=4.0 + wobble * 0.1,
    )


def test_not_reliable_before_minimum_turns():
    calibrator = BaselineCalibrator(speaker=SpeakerId("interview", "c1"))
    calibrator.add_turn(_features(170.0))
    profile = calibrator.build_profile()
    assert profile.n_calibration_turns == 1
    assert profile.is_reliable is False


def test_reliable_after_enough_varied_turns():
    calibrator = BaselineCalibrator(speaker=SpeakerId("interview", "c1"))
    for f0, wobble in ((170.0, -3.0), (175.0, 1.0), (165.0, -1.0), (180.0, 3.0)):
        calibrator.add_turn(_features(f0, wobble))
    profile = calibrator.build_profile()
    assert profile.n_calibration_turns == 4
    assert profile.is_reliable is True
    # mean should land near the sample mean of the 4 values
    assert 165.0 <= profile.means["f0_mean_hz"] <= 180.0


def test_turn_with_undetected_pitch_is_skipped_not_corrupting_baseline():
    """Reproduces the real 2026-09-06 finding: Praat found zero voiced frames on
    several calibration turns of a live session. Before f0_detected existed, those
    turns' fallback zeros were silently averaged into the baseline mean/std. A skipped
    turn must not count toward n_calibration_turns or contaminate the stats."""
    calibrator = BaselineCalibrator(speaker=SpeakerId("interview", "c1"))
    calibrator.add_turn(_features(180.0))  # real turn
    calibrator.add_turn(
        ProsodyFeatures(
            f0_mean_hz=0.0, f0_std_hz=0.0, jitter_local_pct=0.0, shimmer_local_pct=0.0,
            hnr_db=-6.5, speech_rate_syll_s=10.0, f0_detected=False,
        )
    )  # Praat failure — must be skipped entirely
    calibrator.add_turn(_features(185.0, wobble=2.0))  # real turn

    profile = calibrator.build_profile()

    assert profile.n_calibration_turns == 2  # the failed turn does not count
    assert calibrator.n_turns_skipped_no_pitch == 1
    # baseline mean must reflect only the two real turns (~180-185), not be dragged
    # toward 0 by the fabricated zero
    assert 175.0 <= profile.means["f0_mean_hz"] <= 190.0


def test_identical_turns_yield_degenerate_std_and_are_not_reliable():
    """If a 'neutral' calibration window is suspiciously constant (e.g. clipped audio,
    or a bug feeding the same clip 3x), the profile must flag itself unreliable rather
    than silently producing a zero-variance baseline that would blow up z-scores."""
    calibrator = BaselineCalibrator(speaker=SpeakerId("interview", "c1"))
    for _ in range(4):
        calibrator.add_turn(_features(170.0))
    profile = calibrator.build_profile()
    assert profile.is_reliable is False
