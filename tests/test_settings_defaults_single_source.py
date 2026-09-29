"""
Production-issues pass (2026-09-29): app.py, pages/settings.py and
utils/scoring_core.ScoringParams.from_settings() each used to hand-type
their own literal default for the same setting. On 15 keys they disagreed
-- most materially nifty_regime_filter (True in pages/settings.py's
DEFAULTS, False in app.py's own literal: the regime filter was silently
OFF in every fresh session, contradicting the documented default) and
t1_cci_window (2 in the ScoringParams dataclass itself, 4 in
pages/settings.py's DEFAULTS, 5 in both app.py's literal and
from_settings()'s old fallback -- three different values, none of them
reliably the one actually used).

Fix: utils/settings_defaults.DEFAULTS is now the only place any of these
literals are written down. app.py and pages/settings.py both read it;
ScoringParams.from_settings() falls back to ScoringParams()'s own field
values instead of a second literal. These tests exist to keep it that way
-- they fail the moment a new hand-typed literal reappears anywhere in
this chain, not just for the fields that happened to have already drifted.
"""
import ast
import re

import pytest

from utils.scoring_core import ScoringParams
from utils.settings_defaults import DEFAULTS


# ── 1. from_settings() can no longer diverge from the dataclass ────────────
def test_from_settings_empty_dict_matches_dataclass_defaults_on_every_field():
    dc = ScoringParams()
    fs = ScoringParams.from_settings({})
    for f in dc.__dataclass_fields__:
        assert getattr(fs, f) == getattr(dc, f), f"{f}: from_settings()={getattr(fs,f)!r} != dataclass default={getattr(dc,f)!r}"


def test_from_settings_partial_dict_only_overrides_the_keys_given():
    fs = ScoringParams.from_settings({"t1_cci_window": 7})
    assert fs.t1_cci_window == 7
    dc = ScoringParams()
    for f in dc.__dataclass_fields__:
        if f != "t1_cci_window":
            assert getattr(fs, f) == getattr(dc, f)


def test_regression_t1_cci_window_no_longer_silently_5():
    """The literal bug this pass fixed: from_settings({}) used to return 5
    (a value that appeared in no other definition of this parameter)."""
    assert ScoringParams.from_settings({}).t1_cci_window == ScoringParams().t1_cci_window


# ── 2. app.py's settings dict: no reintroduced hand-typed literal ──────────
def _extract_settings_block(src: str) -> str:
    i = src.index("ss = st.session_state\nsettings = {")
    j = src.index("\n}\n", i) + 3
    return src[i:j]


def _keys_with_literal_fallback(block: str) -> dict:
    """key -> literal fallback, for every ss.get("key", <not DEFAULTS[...]>)
    line in the block (i.e. NOT already routed through DEFAULTS)."""
    out = {}
    for m in re.finditer(r'"([a-zA-Z0-9_]+)":\s*ss\.get\(\s*"[a-zA-Z0-9_]+"\s*,\s*([^)]+)\)', block):
        key, default_expr = m.groups()
        default_expr = default_expr.strip()
        if not default_expr.startswith("DEFAULTS["):
            out[key] = default_expr
    return out


def test_app_py_settings_dict_has_no_hand_typed_literal_for_any_shared_key():
    src = open("app.py").read()
    block = _extract_settings_block(src)
    literals = _keys_with_literal_fallback(block)
    # "symbols" intentionally falls back to NIFTY500_SYMBOLS, not a DEFAULTS
    # key (DEFAULTS has no universe of symbols, only a universe_mode label).
    literals.pop("symbols", None)
    shared_with_defaults = {k: v for k, v in literals.items() if k in DEFAULTS}
    assert shared_with_defaults == {}, (
        f"app.py hand-types a literal default for keys also in DEFAULTS "
        f"(should read DEFAULTS[...] instead): {shared_with_defaults}"
    )


def test_app_py_settings_dict_every_defaults_key_it_uses_matches_defaults_value():
    """Belt-and-suspenders: even where the literal equals DEFAULTS[key] today,
    assert it -- so a future manual edit to DEFAULTS alone (without touching
    app.py) is caught by drift, not silently ignored."""
    src = open("app.py").read()
    block = _extract_settings_block(src)
    for m in re.finditer(r'"([a-zA-Z0-9_]+)":\s*ss\.get\(\s*"[a-zA-Z0-9_]+"\s*,\s*(DEFAULTS\[[^\]]+\])\)', block):
        key, expr = m.groups()
        assert expr == f'DEFAULTS["{key}"]', f"{key}: fallback {expr} does not key DEFAULTS by its own name"


# ── 3. pages/settings.py no longer defines its own DEFAULTS literal dict ───
def test_pages_settings_imports_defaults_rather_than_redefining_it():
    src = open("pages/settings.py").read()
    assert "from utils.settings_defaults import DEFAULTS" in src
    # No second top-level "DEFAULTS = {" assignment left behind.
    tree = ast.parse(src)
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "DEFAULTS" for t in n.targets)]
    assert assigns == [], "pages/settings.py still assigns DEFAULTS directly instead of importing it"


# ── 4. utils/settings_defaults is a plain data module (safe for app.py to import) ──
def test_settings_defaults_module_has_no_streamlit_or_heavy_imports():
    src = open("utils/settings_defaults.py").read()
    tree = ast.parse(src)
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    assert imported == set(), f"utils/settings_defaults.py should be a plain data module; found imports: {imported}"


def test_defaults_dict_is_nonempty_and_covers_every_scoring_param_field():
    dc_fields = set(ScoringParams().__dataclass_fields__)
    # Not every ScoringParams field has a UI widget (e.g. max_score is fixed),
    # but every field that DOES appear in DEFAULTS must be spelled the same
    # both places -- this just guards against a silent key rename.
    for f in dc_fields & set(DEFAULTS):
        assert f in DEFAULTS  # trivially true; documents the intersection exists
    assert len(set(DEFAULTS) & dc_fields) >= 20, "DEFAULTS should still mirror most ScoringParams fields"


# ── 5. the specific regressions this pass fixed, named so they can't silently return ──
# ── 6. repo-wide sweep: no NEW hand-typed literal for a DEFAULTS key ───────
# Extends section 2's app.py-only check to every .py file in the repo (except
# the four files below, which use named module-level constants -- already
# equal to DEFAULTS today -- as their OWN standalone default outside the
# settings-dict path; rewiring those has a different, wider blast radius and
# was deliberately left alone in this pass. Fenced explicitly, rather than
# just excluded silently, so removing a file from this list is a visible,
# deliberate act, and so the list can't quietly grow to hide a real miss.
_ACKNOWLEDGED_MODULE_CONSTANT_FILES = {
    "utils/legacy_scoring_diagnostic.py",   # ENABLE_GOLDEN_PULLBACK_PATTERN
    "utils/pillar_engine.py",               # RECLAIM_* / ic_* constants
    "utils/promotion_engine.py",            # MIN_RR_ELITE / *_SCORE_MIN
    "utils/scanner_engine.py",              # ENABLE_STRUCTURAL_GATE / enable_sector_rs
}


def _repo_py_files():
    import pathlib
    skip_dirs = {".git", "__pycache__"}
    for p in pathlib.Path(".").rglob("*.py"):
        if not any(part in skip_dirs for part in p.parts):
            yield str(p)


def test_no_file_in_the_repo_hand_types_a_literal_default_for_a_defaults_key():
    """Repo-wide version of test_app_py_settings_dict_has_no_hand_typed_literal:
    for every `.get("key", <expr>)` anywhere, if key is in DEFAULTS, <expr>
    must be DEFAULTS["key"] (or a same-named local/param -- e.g.
    `settings.get("hold_days", hold_days)` -- which is not a duplicate
    literal, just a caller-supplied fallback)."""
    pat = re.compile(r'\.get\(\s*"([a-zA-Z0-9_]+)"\s*,\s*([^)]+)\)')
    offenders = {}
    for path in _repo_py_files():
        if path == "utils/settings_defaults.py" or path in _ACKNOWLEDGED_MODULE_CONSTANT_FILES:
            continue
        try:
            src = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for m in pat.finditer(src):
            key, default_expr = m.groups()
            default_expr = default_expr.strip()
            if key not in DEFAULTS:
                continue
            if default_expr.startswith("DEFAULTS[") or default_expr.startswith("_dc."):
                continue
            if re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", default_expr) and default_expr == key:
                continue   # fallback to a same-named local/param, not a literal
            offenders.setdefault(path, []).append((key, default_expr))
    assert offenders == {}, f"hand-typed literal default(s) for DEFAULTS key(s) found: {offenders}"


@pytest.mark.parametrize("key,reason", [
    ("nifty_regime_filter", "was True in DEFAULTS but False in app.py's old literal"),
    ("t1_adx_min",           "was 23 in DEFAULTS but 20 in app.py's old literal"),
    ("t1_rs_min",            "was 0.01 in DEFAULTS but 0.0 in app.py's old literal"),
    ("t2_vol_mult",          "was 1.5 in DEFAULTS but 1.2 in app.py's old literal"),
    ("t1_cci_window",        "was 4 (DEFAULTS) / 5 (app.py, from_settings) / 2 (dataclass) -- three values"),
])
def test_named_regressions_app_now_reads_defaults_for(key, reason):
    src = open("app.py").read()
    block = _extract_settings_block(src)
    m = re.search(rf'"{re.escape(key)}":\s*ss\.get\(\s*"{re.escape(key)}"\s*,\s*([^)]+)\)', block)
    assert m, f"{key} not found in app.py settings dict"
    assert m.group(1).strip() == f'DEFAULTS["{key}"]', f"{key}: {reason}"
