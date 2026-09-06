"""Static guards on the live-capture page (ADR-041), mirroring test_frontend_contract.py's
discipline for index.html — same rationale: the two frontend regressions that actually
reached a reviewer (Appendix A rows 32-33) were both invisible to every Python test, and
both are checkable as text without a browser. Every guard here is mutation-tested per
ADR-039.
"""
import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[2] / "web" / "static"
HTML = (STATIC / "capture.html").read_text(encoding="utf-8")
JS = (STATIC / "capture.js").read_text(encoding="utf-8")
WORKLET = (STATIC / "capture-worklet.js").read_text(encoding="utf-8")
CSS = (STATIC / "style.css").read_text(encoding="utf-8")
INDEX_HTML = (STATIC / "index.html").read_text(encoding="utf-8")


def test_every_local_asset_is_version_queried_with_one_shared_version():
    versions = set(re.findall(r'(?:href|src)="[\w./-]+\?v=(\d+)"', HTML))
    assert versions, "no cache-busting version found on capture.html's local assets"
    assert len(versions) == 1, f"capture.html assets reference different versions: {sorted(versions)}"


def test_capture_and_dashboard_pages_share_the_same_asset_version():
    """style.css is loaded by both pages — a version bump on one page's copy without
    the other reintroduces exactly the staleness trap ADR-036 named (Appendix A row 32),
    just split across two pages instead of within one."""
    capture_versions = set(re.findall(r'style\.css\?v=(\d+)', HTML))
    dashboard_versions = set(re.findall(r'style\.css\?v=(\d+)', INDEX_HTML))
    assert capture_versions and dashboard_versions
    assert capture_versions == dashboard_versions, (
        f"capture.html uses style.css?v={capture_versions}, "
        f"index.html uses v={dashboard_versions}"
    )


def test_worklet_module_url_is_version_queried():
    """capture.js loads the worklet via `audioWorklet.addModule(url)` — a plain browser
    fetch, invisible to the no-store middleware's usual JS/CSS/HTML matching, and easy
    to forget to bump."""
    assert re.search(r'addModule\("capture-worklet\.js\?v=\d+"\)', JS)


def test_every_element_the_script_reaches_for_exists_in_the_markup():
    wanted = set(re.findall(r'\$\("([\w-]+)"\)', JS))
    present = set(re.findall(r'id="([\w-]+)"', HTML))
    assert wanted, "no $() lookups found; did capture.js move?"
    assert not (wanted - present), f"capture.js reads ids absent from capture.html: {sorted(wanted - present)}"


def test_every_class_used_has_a_style_rule():
    used = set()
    for m in re.finditer(r'class="([^"]+)"', HTML):
        used |= set(m.group(1).split())
    # capture.js sets className via template literals (`capture-msg ${role}`) and one
    # literal (`live-turn-card`) — the interpolated role half (.candidate/.agent) is
    # covered by test_conversation_roles_used_by_the_script_have_style_rules instead.
    used |= {"capture-msg", "live-turn-card"}

    defined = set(re.findall(r'\.([A-Za-z][\w-]*)', CSS))
    assert used, "no classes found; did the markup move?"
    assert not (used - defined), f"unstyled classes: {sorted(used - defined)}"


@pytest.mark.parametrize("role", ["candidate", "agent"])
def test_conversation_roles_used_by_the_script_have_style_rules(role):
    assert f'className = `capture-msg ${{role}}`' in JS or "capture-msg" in JS
    assert re.search(rf'\.capture-msg\.{role}\b', CSS), f"no CSS rule for .capture-msg.{role}"


def test_three_panel_states_all_start_correctly_hidden_or_not():
    """Exactly one of idle/live/done should be visible on page load — the other two
    carry `hidden` in the markup, and JS toggles `.hidden` at runtime rather than
    `style.display` (index.html's own convention, ADR-036's artifact-authoring rule)."""
    idle = re.search(r'<div id="capture-idle"([^>]*)>', HTML)
    live = re.search(r'<div id="capture-live"([^>]*)>', HTML)
    done = re.search(r'<div id="capture-done"([^>]*)>', HTML)
    assert idle and live and done
    assert "hidden" not in idle.group(1)
    assert "hidden" in live.group(1)
    assert "hidden" in done.group(1)


def test_start_and_end_buttons_are_wired_to_handlers():
    assert re.search(r'\$\("start-btn"\)\.addEventListener\("click",\s*startSession\)', JS)
    assert re.search(r'\$\("end-btn"\)\.addEventListener\("click",\s*endSession\)', JS)


def test_view_analysis_link_targets_the_dashboard_with_a_session_query_param():
    assert 'index.html?session=' in JS


def test_dashboard_supports_the_session_query_param_capture_links_to():
    """The other half of the deep link this page relies on — if index.html's own
    bootstrap ever drops `?session=` support, capture.html's "View Analysis" link
    silently stops working even though nothing here changed."""
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "URLSearchParams(location.search).get(\"session\")" in app_js


def test_worklet_always_returns_true_from_process():
    """An AudioWorkletProcessor.process() that returns anything falsy tells the browser
    to garbage-collect the node — audio capture would silently stop after one block."""
    # Every `return` in the process() method body must be `return true`.
    body_start = WORKLET.index("process(inputs)")
    body_end = WORKLET.index("_emit(samples)")
    body = WORKLET[body_start:body_end]
    returns = re.findall(r'return\s+(\w+)\s*;', body)
    assert returns, "no return statements found in process() — did the method move?"
    assert all(r == "true" for r in returns), f"process() returns something other than true: {returns}"


def test_worklet_never_connects_to_destination_in_capture_js():
    """Connecting the mic-observing worklet to audioContext.destination would create an
    audible echo of the candidate's own voice — capture.js deliberately leaves it
    unconnected (comment explains why); this guards the omission stays deliberate."""
    assert "workletNode" in JS
    assert "worklet.connect(" not in JS.replace(" ", "")


def test_status_log_caps_line_length_and_line_count():
    """ADR-043: a real session froze the browser tab after the backend's guessed
    reply.audio payload key turned out wrong, dumping the raw event (a multi-kilobyte
    base64 blob) as one status line, repeated once per audio chunk with no cap on
    either dimension. The backend now truncates its own messages, but a frontend that
    trusts the server to always behave is one future bug away from the same freeze —
    both caps must exist here independently."""
    assert re.search(r'MAX_LINE_CHARS\s*=\s*\d+', JS), "no per-line length cap in capture.js"
    assert re.search(r'MAX_STATUS_LINES\s*=\s*\d+', JS), "no line-count cap in capture.js"
    assert "removeChild" in JS, "old lines are never evicted once the cap is reached"
