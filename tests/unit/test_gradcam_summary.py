"""ADR-032: Grad-CAM as citable facts, not just a PNG the text model can't open."""
import numpy as np
import pytest

from voicestress.application.gradcam_summary import describe_gradcam, summarize_gradcam


def _heatmap(h=96, w=128):
    return np.zeros((h, w), dtype=np.float32)


def test_locates_peak_frequency_band():
    """A hot stripe in the upper mel bins must report a high frequency, not a low one."""
    hm = _heatmap()
    hm[70:80, :] = 1.0  # high mel bins -> high Hz
    summary = summarize_gradcam(hm, fmin_hz=50, fmax_hz=8000)
    assert summary["peak_frequency_hz"] > 2000
    # Inclusive bounds: with a uniform block the peak sits exactly at the band edge,
    # since the half-maximum walk can't extend past the block's own boundary.
    assert summary["band_hz"][0] <= summary["peak_frequency_hz"] <= summary["band_hz"][1]


def test_low_band_activation_reports_low_frequency():
    hm = _heatmap()
    hm[2:8, :] = 1.0  # lowest mel bins
    summary = summarize_gradcam(hm, fmin_hz=50, fmax_hz=8000)
    assert summary["peak_frequency_hz"] < 500


def test_locates_peak_time_position():
    hm = _heatmap()
    hm[:, 100:110] = 1.0  # late in the clip
    assert summarize_gradcam(hm)["peak_time_fraction"] > 0.7

    hm2 = _heatmap()
    hm2[:, 5:15] = 1.0  # early
    assert summarize_gradcam(hm2)["peak_time_fraction"] < 0.2


def test_detects_attention_landing_on_padding():
    """The diagnostic that matters most: a short clip padded to 1s, where the model
    looked at the fabricated silence rather than the speech (ADR-025's artifact, now
    measurable rather than inferred)."""
    hm = _heatmap()
    hm[:, 90:] = 1.0  # all attention in the last ~30% of the clip
    summary = summarize_gradcam(hm, real_audio_fraction=0.3)  # only first 30% is real audio
    assert summary["attention_on_padding"] > 0.9


def test_no_padding_reported_when_clip_is_all_real_audio():
    hm = _heatmap()
    hm[:, 90:] = 1.0
    summary = summarize_gradcam(hm, real_audio_fraction=1.0)
    assert summary["attention_on_padding"] == 0.0


def test_concentration_distinguishes_focused_from_diffuse():
    focused = _heatmap()
    focused[40:45, 60:65] = 1.0
    diffuse = np.full((96, 128), 0.5, dtype=np.float32)

    assert summarize_gradcam(focused)["concentration"] > summarize_gradcam(diffuse)["concentration"]


def test_rejects_malformed_heatmap():
    with pytest.raises(ValueError):
        summarize_gradcam(np.array([]))
    with pytest.raises(ValueError):
        summarize_gradcam(np.zeros((10,)))


# --- description rendering ----------------------------------------------------------


def test_description_mentions_frequency_and_position():
    hm = _heatmap()
    hm[70:80, 5:15] = 1.0
    text = describe_gradcam(summarize_gradcam(hm))
    assert "Hz" in text
    assert "early in" in text


def test_description_warns_loudly_when_attention_is_on_padding():
    hm = _heatmap()
    hm[:, 90:] = 1.0
    text = describe_gradcam(summarize_gradcam(hm, real_audio_fraction=0.3))
    assert "WARNING" in text
    assert "zero-padded silence" in text
    assert "artifact" in text


def test_description_omits_padding_warning_when_not_applicable():
    hm = _heatmap()
    hm[40:50, 40:50] = 1.0
    text = describe_gradcam(summarize_gradcam(hm, real_audio_fraction=1.0))
    assert "WARNING" not in text
