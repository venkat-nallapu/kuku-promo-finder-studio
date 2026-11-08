"""Automated Unit, E2E, and LLM Eval Test Suite (`tests/test_pipeline.py`).

Covers all 12 codepaths and failure modes from the `/plan-eng-review` test matrix.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.benchmark import (
    build_ffmpeg_command,
    compute_model_swap_diff,
    compute_naive_vs_hybrid_benchmark,
)
from src.candidates import (
    MAX_PROMO_DURATION,
    MIN_PROMO_DURATION,
    MIN_STITCH_SEGMENT_DURATION,
    build_all_candidates,
    extract_withheld_payoff,
    generate_episode_candidates,
    generate_stitched_candidates,
    is_continuation_dialogue_cue,
    is_dialogue_safe_end_timestamp,
    is_dialogue_safe_start_timestamp,
    is_dialogue_safe_timestamp,
    nms_deduplicate,
    score_window_structural,
    temporal_iou,
)
from src.llm_judge import (
    extract_json_payload,
    score_candidates_with_judge,
)
from src.parser import (
    extract_speaker_lore_tokens,
    load_episodes_from_zip,
    load_synthetic_demo_episodes,
    parse_episode_text,
)


def test_parser_tracks() -> None:
    """Verify 4-track separation (CAMERA, ACTION, SOUND_FX, DIALOGUE) and multi-paren headers."""
    sample = """EPISODE NUMBER: Episode 7
EPISODE LOGLINE: Test logline.
CONTINUITY NOTES: Test continuity.

--- SCRIPT START ---
SCENE 01
Scene heading: INT. HALL - NIGHT
Scene start: 00:00.00
Scene end: 00:30.00

**CAMERA:**
Start: (00:00.00) | End: (00:04.00)
Wide shot of the hall.

**ACTION:**
Start: (00:01.00) | End: (00:05.00)
Alexander steps forward.

**ALEXANDER (O.S.) (Cold, Commanding):**
Start: (00:05.00) | End: (00:12.00)
"Drop the blade now!"

**SOUND FX:**
Start: (00:12.00) | End: (00:15.00)
Metallic clang echoes to a cliffhanger peak.
--- SCRIPT END ---
"""
    ep = parse_episode_text(sample, "episode-007.txt")
    assert ep.episode_num == 7
    assert len(ep.cues) == 4
    assert len(ep.camera_cues) == 1
    assert len(ep.action_cues) == 1
    assert len(ep.dialogue_cues) == 1
    assert len(ep.sound_fx_cues) == 1
    dlg = ep.dialogue_cues[0]
    assert dlg.speaker == "ALEXANDER"
    assert dlg.emotion == "O.S., Cold, Commanding"
    assert ep.has_explicit_cliffhanger is True


def test_scene_offset_monotonic() -> None:
    """Verify Scene-Reset Clock Normalizer shifts Scene 2 when it resets to 00:00.00."""
    sample = """EPISODE NUMBER: Episode 2
SCENE 01
Scene heading: INT. ROOM A - DAY
Scene start: 00:00.00
Scene end: 00:30.00

**JULIANA (Calm):**
Start: (00:00.00) | End: (00:30.00)
"First scene line."

SCENE 02
Scene heading: INT. ROOM B - DAY
Scene start: 00:00.00
Scene end: 00:25.00

**ALEXANDER (Stern):**
Start: (00:00.00) | End: (00:25.00)
"Second scene line after clock reset."
"""
    ep = parse_episode_text(sample, "episode-002.txt")
    assert len(ep.scenes) == 2
    assert ep.scenes[1].clock_reset_applied is True
    assert ep.cues[1].start_sec == pytest.approx(30.0)
    assert ep.cues[1].end_sec == pytest.approx(55.0)
    assert ep.total_duration_sec == pytest.approx(55.0)


def test_dynamic_speaker_lore_set() -> None:
    """Verify dynamic speaker extraction captures character names and drops generic roles."""
    eps = load_synthetic_demo_episodes()
    tokens = extract_speaker_lore_tokens(eps)
    assert "clara" in tokens
    assert "damian" in tokens
    assert "vance" in tokens
    assert "guard" not in tokens
    assert "camera" not in tokens


def test_corrupt_input_fallback() -> None:
    """Verify untimed plain-text fallback (150 WPM) and empty input rejection."""
    with pytest.raises(ValueError):
        parse_episode_text("   ", "empty.txt")

    untimed = """EPISODE NUMBER: Episode 9
CLARA: You cannot lock these doors forever! We know the truth about the Duke's regiment and we will expose the forged treaty tonight!

DAMIAN: Then let them come and try to take the seal from my hand!
"""
    ep = parse_episode_text(untimed, "untimed.txt")
    assert len(ep.cues) >= 2
    assert ep.total_duration_sec >= 4.0


def test_dialogue_safe_cuts() -> None:
    """Verify is_dialogue_safe_timestamp blocks mid-sentence cuts."""
    eps = load_synthetic_demo_episodes()
    ep1 = eps[0]
    # Dialogue cue is 10.00 - 16.50
    assert is_dialogue_safe_timestamp(10.0, ep1.dialogue_cues) is True
    assert is_dialogue_safe_timestamp(13.2, ep1.dialogue_cues) is False
    assert is_dialogue_safe_timestamp(16.5, ep1.dialogue_cues) is True


def test_duration_bounds_20_to_60s() -> None:
    """Verify 100% of generated candidates satisfy 20.0s <= duration <= 60.0s."""
    eps = load_synthetic_demo_episodes()
    cands, _ = build_all_candidates(eps, max_single_per_episode=3, include_stitched=True)
    assert len(cands) >= 2
    for c in cands:
        assert MIN_PROMO_DURATION <= c.duration_sec <= MAX_PROMO_DURATION
        assert c.dialogue_safe is True


def test_cross_episode_payoff() -> None:
    """Verify episode-end cliffhangers pull their withheld payoff from Episode N+1."""
    eps = load_synthetic_demo_episodes()
    payoff_text, source = extract_withheld_payoff(
        eps[0], end_sec=eps[0].total_duration_sec - 1.0, next_episode=eps[1]
    )
    assert source == "next_episode"
    assert "Ep 02" in payoff_text


def test_spoken_only_lore_penalty() -> None:
    """Verify proper nouns in CAMERA/ACTION do not penalize standalone_clarity."""
    script = """EPISODE NUMBER: Episode 1
SCENE 01
Scene heading: INT. HALL - NIGHT
Scene start: 00:00.00
Scene end: 00:35.00

**CAMERA:**
Start: (00:00.00) | End: (00:10.00)
Tracking Juliana, Gerald, Clarissa, Catherine, Rupert, and Alexander across the room.

**ACTION:**
Start: (00:10.00) | End: (00:20.00)
Juliana slaps Gerald while Clarissa, Catherine, and Rupert gasp in horror.

**JULIANA (Furious):**
Start: (00:20.00) | End: (00:35.00)
"You betrayed your bride at the altar! Kneel before the guard!"
"""
    ep = parse_episode_text(script, "ep1.txt")
    lore_set = {"juliana", "gerald", "clarissa", "catherine", "rupert", "alexander"}
    cand = score_window_structural(ep, 0.0, 35.0, speaker_lore_tokens=lore_set)
    # Zero lore names were SPOKEN in DIALOGUE, so spoken_lore_tokens_found must be empty!
    assert cand.spoken_lore_tokens_found == []
    assert cand.standalone_clarity >= 18.0


def test_nms_iou_deduplication() -> None:
    """Verify temporal IoU math and Non-Maximum Suppression deduplication."""
    assert temporal_iou(10.0, 50.0, 12.0, 52.0) > 0.80
    assert temporal_iou(0.0, 30.0, 40.0, 70.0) == 0.0

    eps = load_synthetic_demo_episodes()
    cands = generate_episode_candidates(eps[0], {"clara", "damian"}, max_per_episode=3)
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            assert (
                temporal_iou(
                    cands[i].start_sec,
                    cands[i].end_sec,
                    cands[j].start_sec,
                    cands[j].end_sec,
                )
                < 0.45
            )


def test_multi_segment_stitches() -> None:
    """Verify multi-segment stitched candidates satisfy segment >= 6.0s and total 20-60s."""
    zip_path = Path("I Married a Broken Soldier. He was a Duke-scripts (2).zip")
    if not zip_path.exists():
        pytest.skip("Confidential zip not present")
    eps = load_episodes_from_zip(zip_path)
    lore = extract_speaker_lore_tokens(eps)
    stitches = generate_stitched_candidates(eps[2], lore, next_episode=eps[3])
    assert len(stitches) >= 1
    for st in stitches:
        assert len(st.segments) >= 2
        for seg in st.segments:
            assert seg.duration_sec >= MIN_STITCH_SEGMENT_DURATION
        assert MIN_PROMO_DURATION <= st.duration_sec <= MAX_PROMO_DURATION
        cmd = build_ffmpeg_command(st)
        assert "filter_complex" in cmd and "concat=n=" in cmd


def test_llm_3layer_rescue_and_cache(tmp_path: Path) -> None:
    """Verify Layer-1 JSON repair, Layer-3 HTTP 429 fallback, and SHA-256 disk caching."""
    noisy = 'Here is the output:\n```json\n{"evaluations": [{"candidate_id": "c1", "llm_score": 92.0,}]}\n```'
    parsed = extract_json_payload(noisy)
    assert parsed["evaluations"][0]["candidate_id"] == "c1"

    eps = load_synthetic_demo_episodes()
    cands, _ = build_all_candidates(eps)
    cache_file = tmp_path / "test_cache.json"

    # Test Layer-1 malformed JSON repair in live pipeline
    repaired_ranked, _ = score_candidates_with_judge(
        cands,
        provider="gemini",
        cache_path=cache_file,
        simulate_failure_mode="malformed_json",
    )
    assert any(c.llm_score == 91.5 for c in repaired_ranked)

    # Test Layer-3 HTTP 429 rate-limit fallback
    rescued_ranked, tel = score_candidates_with_judge(
        cands,
        provider="openai",
        cache_path=cache_file,
        simulate_failure_mode="http_429",
    )
    assert tel.rescued_batches >= 1
    assert all(c.rescued_fallback is True for c in rescued_ranked)

    # Test SHA-256 cache hit on semantic_reranker
    _, tel1 = score_candidates_with_judge(
        cands, provider="semantic_reranker", cache_path=cache_file
    )
    assert tel1.cache_hits == 0
    _, tel2 = score_candidates_with_judge(
        cands, provider="semantic_reranker", cache_path=cache_file
    )
    assert tel2.cache_hits == len(cands)


def test_e2e_50_episode_outputs() -> None:
    """Verify generated raw_outputs/ deliverables meet all assignment constraints."""
    out_dir = Path("raw_outputs")
    assert (out_dir / "ranked_promos_top20.json").exists()
    assert (out_dir / "all_50_episodes_promos.json").exists()
    assert (out_dir / "where_ai_got_it_wrong.md").exists()
    assert (out_dir / "quality_audit_failures.md").exists()
    assert (out_dir / "editor_cut_sheet.csv").exists()

    top20 = json.loads((out_dir / "ranked_promos_top20.json").read_text(encoding="utf-8"))
    all50 = json.loads((out_dir / "all_50_episodes_promos.json").read_text(encoding="utf-8"))
    assert len(top20) == 20
    assert len(all50) == 50
    for item in top20:
        assert 20.0 <= item["duration_sec"] <= 60.0
        assert item["dialogue_safe"] is True
        assert len(item["withheld_payoff_preview"]) > 10


def test_streamlit_ui_renders() -> None:
    """Verify the 4-tab Streamlit Promo Studio UI renders cleanly with zero exceptions."""
    import streamlit.testing.v1 as st_test

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = st_test.AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 4
    assert len(at.metric) >= 10
    assert len(at.dataframe) >= 4


def test_split_sentence_continuation_guard() -> None:
    """Regression test: reject start/end cuts that split a sentence across a "...and" continuation cue."""
    script = """EPISODE NUMBER: Episode 36
SCENE 01
Scene heading: INT. HALL - NIGHT
Scene start: 00:00.00
Scene end: 00:45.00

**ALEXANDER (Cold):**
Start: (00:00.00) | End: (00:10.00)
"They framed my regiment for treason..."

**ALEXANDER (Furious):**
Start: (00:10.50) | End: (00:20.00)
"...and killing my wife afterward!"

**CATHERINE (Shocked):**
Start: (00:20.00) | End: (00:45.00)
"Who told you about the order?"
"""
    ep = parse_episode_text(script, "ep36.txt")
    assert is_continuation_dialogue_cue(ep.dialogue_cues[1]) is True
    # t=10.50 is between two cues, NOT inside one, yet it starts on "...and killing"
    assert is_dialogue_safe_timestamp(10.50, ep.dialogue_cues) is True
    assert is_dialogue_safe_start_timestamp(10.50, ep.dialogue_cues) is False
    assert is_dialogue_safe_end_timestamp(10.00, ep.dialogue_cues) is False


def test_cold_open_stitch_monotonic_bounds_and_word_safe_rationale() -> None:
    """Regression test: cold-open stitches keep start_sec <= end_sec and rationales never slice words."""
    out_dir = Path("raw_outputs")
    top20 = json.loads((out_dir / "ranked_promos_top20.json").read_text(encoding="utf-8"))
    for item in top20:
        assert item["start_sec"] <= item["end_sec"]
        hook = item["opening_hook_line"]
        assert '"...' not in hook and "'..." not in hook
        assert "rever'" not in item["rationale"]
        assert "int'" not in item["rationale"]


def test_streamlit_cloud_missing_zip_hydration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test: verify Streamlit Cloud hydration when the confidential .zip is absent."""
    import app
    import streamlit.testing.v1 as st_test

    episodes, raw_cands, lore_tokens = app.load_cached_pack.__wrapped__("missing.zip")
    assert len(episodes) == 50
    assert len(raw_cands) >= 50
    assert round(sum(e.total_duration_sec for e in episodes) / 60.0, 1) == 102.7

    # Force Path.exists() to return False for the confidential .zip so AppTest executes the exact Streamlit Cloud branch
    orig_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if self.name.endswith(".zip"):
            return False
        return orig_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = st_test.AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 4




