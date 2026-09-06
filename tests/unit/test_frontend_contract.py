"""Static guards on the dashboard frontend.

These exist because the two frontend regressions that actually reached the reviewer were
both invisible to every Python test: assets served stale because a cache-busting version
was out of step (Appendix A row 32), and controls that were unreachable because the
markup and the script had drifted apart. Neither needs a browser to catch — the contract
between index.html, app.js and style.css is checkable as text.

See ADR-036, which names the hand-bumped `?v=` as a known failure mode; this closes it.
"""
import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[2] / "web" / "static"
HTML = (STATIC / "index.html").read_text(encoding="utf-8")
APP_JS = (STATIC / "app.js").read_text(encoding="utf-8")
CSS = (STATIC / "style.css").read_text(encoding="utf-8")


def test_every_versioned_asset_shares_one_version():
    """A forgotten bump on one asset ships a mismatched pair — the exact staleness trap
    ADR-036 accepted as the cost of not having a build step."""
    versions = set(re.findall(r'(?:href|src)="[\w./-]+\?v=(\d+)"', HTML))
    assert versions, "no cache-busting version found on any local asset"
    assert len(versions) == 1, f"assets reference different versions: {sorted(versions)}"


@pytest.mark.parametrize("asset", ["style.css", "app.js", "markdown.js"])
def test_local_assets_are_version_queried(asset):
    assert re.search(rf'{re.escape(asset)}\?v=\d+', HTML), (
        f"{asset} is referenced without ?v= — a cached copy will survive the next edit"
    )


def test_every_element_the_script_reaches_for_exists_in_the_markup():
    """A getElementById returning null throws mid-render and leaves the panel blank —
    how the interface became unusable rather than merely ugly."""
    wanted = set(re.findall(r'getElementById\("([\w-]+)"\)', APP_JS))
    present = set(re.findall(r'id="([\w-]+)"', HTML))
    assert wanted, "no element lookups found; did app.js move?"
    assert not (wanted - present), f"app.js reads ids absent from index.html: {sorted(wanted - present)}"


def test_every_class_used_has_a_style_rule():
    used = set()
    for m in re.finditer(r'class="([^"]+)"', HTML):
        used |= set(m.group(1).split())
    for m in re.finditer(r'class="([^"$]*?)"', APP_JS):
        used |= {c for c in m.group(1).split() if c}
    for m in re.finditer(r'classList\.(?:add|toggle|remove)\(([^)]*)\)', APP_JS):
        used |= set(re.findall(r'"([\w-]+)"', m.group(1)))

    defined = set(re.findall(r'\.([A-Za-z][\w-]*)', CSS))
    assert used, "no classes found; did the markup move?"
    assert not (used - defined), f"unstyled classes: {sorted(used - defined)}"


def test_analyst_chat_ships_visible():
    """ADR-036: the interrogation is the product. Guards the container and the composer
    specifically — children may legitimately toggle `hidden` at runtime (the starter
    chips do), so a blanket search for the word would fire falsely."""
    section = re.search(r'<div class="[^"]*chat-section[^"]*"([^>]*)>', HTML)
    assert section, "no chat-section container in the markup"
    assert "hidden" not in section.group(1), "chat panel ships hidden"

    for element_id in ("chat-form", "chat-input", "chat-log"):
        tag = re.search(rf'<[^>]*id="{element_id}"([^>]*)>', HTML)
        assert tag, f"#{element_id} missing from the markup"
        assert "hidden" not in tag.group(1), f"#{element_id} ships hidden"


def test_chat_panel_keeps_a_headline_share_of_the_rail():
    """It was once `flex-shrink: 0` with no floor, which let the evidence section grow to
    fill and strand the chat in a strip at the bottom — visually a footer, not the
    feature. Pin both the growth share and the floor."""
    rule = re.search(r'\.chat-section\s*\{([^}]*)\}', CSS)
    assert rule, "no .chat-section rule"
    body = rule.group(1)

    grow = re.search(r'flex:\s*([\d.]+)', body)
    assert grow and float(grow.group(1)) >= 1, ".chat-section no longer grows with the rail"

    floor = re.search(r'min-height:\s*(\d+)px', body)
    assert floor and int(floor.group(1)) >= 280, (
        ".chat-section has no meaningful minimum height; it can collapse to a footer strip"
    )


# Leaf fields of to_evidence_dict() that are unambiguous enough to match textually.
# `label` is deliberately absent: it collides with DOM/CSS uses of the same word.
EVIDENCE_LEAVES = {
    "stress": ["score", "score_reliable", "calibrated", "duration_ms"],
    "prosody_raw": [
        "f0_detected", "f0_mean_hz", "f0_std_hz", "jitter_local_pct",
        "shimmer_local_pct", "hnr_db", "speech_rate_syll_s",
    ],
    "xai": ["gradcam_png", "spectrogram_png", "gradcam_description", "gradcam_summary"],
}


@pytest.mark.parametrize(
    "group,leaf",
    [(g, leaf) for g, leaves in EVIDENCE_LEAVES.items() for leaf in leaves],
)
def test_evidence_leaves_are_read_through_their_group(group, leaf):
    """ADR-035 made to_evidence_dict() the single evidence schema for the dashboard and
    the agent alike. Reading `turn.score` instead of `turn.stress.score` yields undefined,
    not an error — the UI renders "NaN" and no test notices. So: every access to a leaf
    must carry its group, wherever it appears."""
    for m in re.finditer(rf'\.{leaf}\b', APP_JS):
        prefix = APP_JS[max(0, m.start() - 60):m.start()]
        assert prefix.endswith(f".{group}") or prefix.endswith(f'["{group}"]'), (
            f"`.{leaf}` read without its `{group}` group near: "
            f"...{APP_JS[max(0, m.start() - 40):m.end() + 10]!r}"
        )


def test_the_group_containers_are_actually_read():
    """Guards the test above from passing vacuously if app.js stops reading evidence."""
    for group in EVIDENCE_LEAVES:
        assert f".{group}." in APP_JS, f"app.js never reads the {group} group"
