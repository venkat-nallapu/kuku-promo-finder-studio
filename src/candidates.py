"""Stage-1 Structural Candidate Window Generator, NMS Deduplicator & Multi-Segment Stitcher.

ASCII PIPELINE DIAGRAM:
===============================================================================
  [EpisodeScript (Concurrent DIALOGUE / ACTION / CAMERA / SOUND_FX Tracks)]
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 1. Dialogue-Safe Boundary Grid                                          │
  │    - Collects candidate cut timestamps from cue boundaries.             │
  │    - Rejects any cut point t that slices through an active DIALOGUE     │
  │      interval (d.start_sec + 0.05 < t < d.end_sec - 0.05).              │
  └─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 2. Hard Duration Gate (20.0s <= duration <= 60.0s)                      │
  │    - Evaluates all valid (start_sec, end_sec) pairs in [20.0s, 60.0s].  │
  │    - Scores 5 structural sub-metrics (0-20 each = 100 max):             │
  │      * Hook Velocity (first 6s kinetic action + emotional arousal)      │
  │      * Conflict Escalation (speaker turn-taking + status reversal)      │
  │      * Cliffhanger Peak (final 8s SOUND_FX stings + unresolved cut)     │
  │      * Standalone Clarity (spoken-only lore filter; ignores CAMERA)     │
  │      * Duration & Pacing Fit (28s-46s sweet spot + dialogue density)    │
  └─────────────────────────────────────────────────────────────────────────┘
         │
         ├──► [Single-Segment NMS Dedup (IoU < 0.45)] ──► Top-3 / Episode
         │
         └──► [Multi-Segment Stitcher (2-3 segs, each >=6s, sum 20-60s)]
                ├── Template A: Intra-Episode Ellipsis (cuts slow middle)
                └── Template B: Cold-Open Flash-Forward (6-12s peak + setup)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.parser import Cue, EpisodeScript, extract_speaker_lore_tokens, format_timestamp

MIN_PROMO_DURATION = 20.0
MAX_PROMO_DURATION = 60.0
MIN_STITCH_SEGMENT_DURATION = 6.0
NMS_IOU_THRESHOLD = 0.45

KINETIC_HOOK_TERMS: Set[str] = {
    "slap",
    "slaps",
    "slapped",
    "strike",
    "strikes",
    "struck",
    "punch",
    "blade",
    "dagger",
    "sword",
    "knife",
    "gun",
    "pistol",
    "blood",
    "bleeding",
    "kneel",
    "kneels",
    "drag",
    "drags",
    "dragged",
    "choke",
    "chokes",
    "throat",
    "neck",
    "shatter",
    "shatters",
    "slam",
    "slams",
    "grab",
    "grabs",
    "seize",
    "seizes",
    "arrest",
    "guards",
    "altar",
    "bridal",
    "gown",
    "veil",
    "ring",
    "wheelchair",
    "chains",
    "dungeon",
    "cell",
    "whip",
    "fire",
    "burn",
    "burning",
    "poison",
    "corpse",
    "dead",
    "kill",
    "murder",
    "scream",
    "screams",
    "gasp",
    "gasps",
    "tear",
    "tears",
    "ripped",
    "shove",
    "shoves",
    "kick",
    "kicks",
    "bed",
    "bare",
    "kiss",
    "embrace",
    "scandal",
    "betray",
    "betrayal",
    "bastard",
    "whore",
    "traitor",
    "thief",
    "imposter",
}

CONFLICT_STAKES_TERMS: Set[str] = {
    "marry",
    "marriage",
    "wedding",
    "bride",
    "groom",
    "wife",
    "husband",
    "fiance",
    "betrothed",
    "divorce",
    "duke",
    "duchess",
    "lord",
    "lady",
    "heir",
    "birthright",
    "inheritance",
    "fortune",
    "estate",
    "manor",
    "crown",
    "king",
    "treason",
    "prison",
    "dungeon",
    "execution",
    "gallows",
    "hang",
    "trial",
    "guilty",
    "innocent",
    "confess",
    "lie",
    "lied",
    "lying",
    "truth",
    "secret",
    "affair",
    "lover",
    "pregnant",
    "child",
    "mother",
    "father",
    "sister",
    "brother",
    "family",
    "vow",
    "swear",
    "revenge",
    "pay",
    "ruin",
    "destroy",
    "forgive",
    "never",
    "impossible",
    "how dare",
    "shut up",
    "get out",
    "kneel",
    "beg",
    "command",
    "order",
}

CLIFFHANGER_AUDIO_VISUAL_TERMS: Set[str] = {
    "cliffhanger",
    "crescendo",
    "ominous",
    "drone",
    "sting",
    "peak",
    "cut to black",
    "abrupt",
    "blackout",
    "freeze",
    "shock",
    "stunned",
    "horrified",
    "terror",
    "terrified",
    "bulging",
    "jaw drops",
    "gasp",
    "shatter",
    "gunshot",
    "thunder",
    "boom",
    "slams",
    "reveal",
    "revealing",
    "unmask",
    "stands up",
    "rises",
    "blood",
    "collapse",
    "collapses",
    "faint",
    "scream",
}

UNIVERSAL_ROLE_ANCHORS: Set[str] = {
    "bride",
    "groom",
    "sister",
    "brother",
    "mother",
    "father",
    "husband",
    "wife",
    "fiance",
    "fiancee",
    "betrothed",
    "duke",
    "duchess",
    "lord",
    "lady",
    "priest",
    "guard",
    "guards",
    "doctor",
    "servant",
    "maid",
    "family",
    "wedding",
    "altar",
    "church",
    "cathedral",
    "prison",
    "dungeon",
    "judge",
    "court",
    "bed",
    "killer",
    "murderer",
    "soldier",
}


@dataclass
class SegmentSpan:
    """A single contiguous time span within an episode."""

    episode_num: int
    start_sec: float
    end_sec: float
    label: str = "primary"

    @property
    def duration_sec(self) -> float:
        return round(max(0.0, self.end_sec - self.start_sec), 2)

    @property
    def start_str(self) -> str:
        return format_timestamp(self.start_sec)

    @property
    def end_str(self) -> str:
        return format_timestamp(self.end_sec)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_num": self.episode_num,
            "start_sec": round(self.start_sec, 2),
            "end_sec": round(self.end_sec, 2),
            "start_timestamp": self.start_str,
            "end_timestamp": self.end_str,
            "duration_sec": self.duration_sec,
            "label": self.label,
        }


@dataclass
class PromoCandidate:
    """A single-segment or multi-segment promo candidate with full provenance."""

    candidate_id: str
    episode_num: int
    episode_filename: str
    promo_type: str  # "single_segment" | "multi_segment_ellipsis" | "multi_segment_cold_open"
    start_sec: float
    end_sec: float
    duration_sec: float
    segments: List[SegmentSpan]
    hook_velocity: float
    conflict_escalation: float
    cliffhanger_peak: float
    standalone_clarity: float
    duration_pacing_fit: float
    stage1_score: float
    final_score: float
    llm_score: Optional[float]
    scoring_provider: str
    rescued_fallback: bool
    dialogue_safe: bool
    explicit_cliffhanger_flag: bool
    spoken_lore_tokens_found: List[str]
    speakers_in_clip: List[str]
    opening_hook_line: str
    closing_cliffhanger_line: str
    withheld_payoff_preview: str
    payoff_source: str  # "same_episode" | "next_episode"
    rationale: str
    editorial_risk_note: str
    excerpt_lines: List[str] = field(default_factory=list)

    @property
    def start_timestamp(self) -> str:
        return format_timestamp(self.start_sec)

    @property
    def end_timestamp(self) -> str:
        return format_timestamp(self.end_sec)

    @property
    def timestamp_range_display(self) -> str:
        if len(self.segments) <= 1:
            return f"Ep {self.episode_num:02d} ({self.start_timestamp} - {self.end_timestamp})"
        parts = [
            f"Ep {s.episode_num:02d} [{s.start_str}-{s.end_str}]"
            for s in self.segments
        ]
        return " + ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "episode_num": self.episode_num,
            "episode_filename": self.episode_filename,
            "promo_type": self.promo_type,
            "start_sec": round(self.start_sec, 2),
            "end_sec": round(self.end_sec, 2),
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "timestamp_range_display": self.timestamp_range_display,
            "duration_sec": round(self.duration_sec, 2),
            "segments": [s.to_dict() for s in self.segments],
            "sub_scores": {
                "hook_velocity": round(self.hook_velocity, 2),
                "conflict_escalation": round(self.conflict_escalation, 2),
                "cliffhanger_peak": round(self.cliffhanger_peak, 2),
                "standalone_clarity": round(self.standalone_clarity, 2),
                "duration_pacing_fit": round(self.duration_pacing_fit, 2),
            },
            "stage1_score": round(self.stage1_score, 2),
            "llm_score": round(self.llm_score, 2) if self.llm_score is not None else None,
            "final_score": round(self.final_score, 2),
            "scoring_provider": self.scoring_provider,
            "rescued_fallback": self.rescued_fallback,
            "dialogue_safe": self.dialogue_safe,
            "explicit_cliffhanger_flag": self.explicit_cliffhanger_flag,
            "spoken_lore_tokens_found": self.spoken_lore_tokens_found,
            "speakers_in_clip": self.speakers_in_clip,
            "opening_hook_line": self.opening_hook_line,
            "closing_cliffhanger_line": self.closing_cliffhanger_line,
            "withheld_payoff_preview": self.withheld_payoff_preview,
            "payoff_source": self.payoff_source,
            "rationale": self.rationale,
            "editorial_risk_note": self.editorial_risk_note,
            "excerpt_lines": self.excerpt_lines,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromoCandidate":
        """Reconstruct a PromoCandidate from a serialized raw_outputs JSON dictionary."""
        sub = data.get("sub_scores", {})
        segs = [
            SegmentSpan(
                episode_num=int(s.get("episode_num", data.get("episode_num", 1))),
                start_sec=float(s["start_sec"]),
                end_sec=float(s["end_sec"]),
                label=str(s.get("label", "primary")),
            )
            for s in data.get("segments", [])
        ]
        if not segs:
            segs = [
                SegmentSpan(
                    episode_num=int(data.get("episode_num", 1)),
                    start_sec=float(data.get("start_sec", 0.0)),
                    end_sec=float(data.get("end_sec", 30.0)),
                )
            ]
        return cls(
            candidate_id=str(data["candidate_id"]),
            episode_num=int(data["episode_num"]),
            episode_filename=str(data.get("episode_filename", f"Episode {int(data['episode_num'])}.txt")),
            promo_type=str(data.get("promo_type", "single_segment")),
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
            duration_sec=float(data["duration_sec"]),
            segments=segs,
            hook_velocity=float(sub.get("hook_velocity", data.get("hook_velocity", 16.0))),
            conflict_escalation=float(sub.get("conflict_escalation", data.get("conflict_escalation", 16.0))),
            cliffhanger_peak=float(sub.get("cliffhanger_peak", data.get("cliffhanger_peak", 16.0))),
            standalone_clarity=float(sub.get("standalone_clarity", data.get("standalone_clarity", 16.0))),
            duration_pacing_fit=float(sub.get("duration_pacing_fit", data.get("duration_pacing_fit", 16.0))),
            stage1_score=float(data.get("stage1_score", 80.0)),
            final_score=float(data.get("final_score", 80.0)),
            llm_score=float(data["llm_score"]) if data.get("llm_score") is not None else None,
            scoring_provider=str(data.get("scoring_provider", "semantic_reranker")),
            rescued_fallback=bool(data.get("rescued_fallback", False)),
            dialogue_safe=bool(data.get("dialogue_safe", True)),
            explicit_cliffhanger_flag=bool(data.get("explicit_cliffhanger_flag", False)),
            spoken_lore_tokens_found=list(data.get("spoken_lore_tokens_found", [])),
            speakers_in_clip=list(data.get("speakers_in_clip", [])),
            opening_hook_line=str(data.get("opening_hook_line", "")),
            closing_cliffhanger_line=str(data.get("closing_cliffhanger_line", "")),
            withheld_payoff_preview=str(data.get("withheld_payoff_preview", "")),
            payoff_source=str(data.get("payoff_source", "same_episode")),
            rationale=str(data.get("rationale", "")),
            editorial_risk_note=str(data.get("editorial_risk_note", "")),
            excerpt_lines=list(data.get("excerpt_lines", [])),
        )


def is_continuation_dialogue_cue(cue: Cue) -> bool:
    """Return True if a DIALOGUE cue begins mid-sentence (e.g., '...and' or lowercase)."""
    if cue.track != "DIALOGUE":
        return False
    stripped = cue.text.strip().lstrip("\"'“‘ ").strip()
    if not stripped:
        return False
    return (
        stripped.startswith("...")
        or stripped.startswith("…")
        or stripped[0].islower()
    )


def is_dialogue_safe_timestamp(
    t: float, dialogue_cues: Sequence[Cue], tolerance: float = 0.05
) -> bool:
    """Return True if timestamp t does not slice inside an active spoken dialogue cue."""
    for dc in dialogue_cues:
        if (dc.start_sec + tolerance) < t < (dc.end_sec - tolerance):
            return False
    return True


def is_dialogue_safe_start_timestamp(
    t: float, dialogue_cues: Sequence[Cue], tolerance: float = 0.05
) -> bool:
    """Return True if t is dialogue-safe AND the first spoken cue at/after t is not a mid-sentence continuation."""
    if not is_dialogue_safe_timestamp(t, dialogue_cues, tolerance=tolerance):
        return False
    for dc in dialogue_cues:
        if dc.start_sec >= (t - tolerance):
            return not is_continuation_dialogue_cue(dc)
    return True


def is_dialogue_safe_end_timestamp(
    t: float, dialogue_cues: Sequence[Cue], tolerance: float = 0.05
) -> bool:
    """Return True if t is dialogue-safe AND does not cut right before a split-sentence continuation cue."""
    if not is_dialogue_safe_timestamp(t, dialogue_cues, tolerance=tolerance):
        return False
    for dc in dialogue_cues:
        if (t - tolerance) <= dc.start_sec <= (t + 3.0):
            if is_continuation_dialogue_cue(dc):
                return False
            break
    return True


def _cues_in_window(
    cues: Sequence[Cue], start_sec: float, end_sec: float
) -> List[Cue]:
    """Return cues whose interval overlaps [start_sec, end_sec] by at least 0.25s."""
    selected: List[Cue] = []
    for c in cues:
        overlap = min(end_sec, c.end_sec) - max(start_sec, c.start_sec)
        if overlap >= 0.25 or (start_sec <= c.start_sec and c.end_sec <= end_sec):
            selected.append(c)
    return selected


def _tokenize_words(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z']+", text)]


def extract_withheld_payoff(
    episode: EpisodeScript,
    end_sec: float,
    next_episode: Optional[EpisodeScript] = None,
) -> Tuple[str, str]:
    """Extract the 3-8s payoff withheld immediately after end_sec.

    If end_sec is within 5.0s of the episode end (episode-ending cliffhanger),
    pulls from the opening 6 seconds + continuity notes of Episode N+1.
    """
    remaining = episode.total_duration_sec - end_sec
    if remaining >= 4.0:
        after_cues = [
            c
            for c in episode.cues
            if c.end_sec > (end_sec + 0.2) and c.start_sec < (end_sec + 9.0)
        ]
        if after_cues:
            snippets: List[str] = []
            for c in after_cues[:3]:
                tag = c.speaker if c.track == "DIALOGUE" and c.speaker else c.track
                snippets.append(f"[{c.start_str}-{c.end_str}] {tag}: {c.text}")
            return (" | ".join(snippets), "same_episode")

    # Cross-episode payoff from Episode N+1
    if next_episode is not None:
        opening_cues = [c for c in next_episode.cues if c.start_sec < 7.0][:2]
        snippets = []
        if next_episode.continuity_notes:
            snippets.append(f"Ep {next_episode.episode_num:02d} Continuity: {next_episode.continuity_notes}")
        for c in opening_cues:
            tag = c.speaker if c.track == "DIALOGUE" and c.speaker else c.track
            snippets.append(
                f"Ep {next_episode.episode_num:02d} [{c.start_str}-{c.end_str}] {tag}: {c.text}"
            )
        if snippets:
            return (" | ".join(snippets), "next_episode")

    # Fallback when Episode 50 ends on a cliffhanger
    tail_cues = episode.cues[-2:] if episode.cues else []
    tail_summary = " ".join(c.text for c in tail_cues)
    return (
        f"Series Finale / Episode-End Cliffhanger Resolution Withheld (Viewer must unlock next episode after: {tail_summary[:140]})",
        "next_episode",
    )


def score_window_structural(
    episode: EpisodeScript,
    start_sec: float,
    end_sec: float,
    speaker_lore_tokens: Set[str],
    next_episode: Optional[EpisodeScript] = None,
    segments: Optional[List[SegmentSpan]] = None,
    promo_type: str = "single_segment",
    materialize_text: bool = True,
) -> PromoCandidate:
    """Compute the 5 deterministic Stage-1 structural sub-scores and build a PromoCandidate."""
    if segments is None:
        segments = [
            SegmentSpan(
                episode_num=episode.episode_num,
                start_sec=start_sec,
                end_sec=end_sec,
                label="primary",
            )
        ]

    duration_sec = round(sum(s.duration_sec for s in segments), 2)
    win_cues: List[Cue] = []
    for seg in segments:
        if seg.episode_num == episode.episode_num:
            win_cues.extend(_cues_in_window(episode.cues, seg.start_sec, seg.end_sec))

    dialogue_cues = [c for c in win_cues if c.track == "DIALOGUE"]
    action_cues = [c for c in win_cues if c.track == "ACTION"]
    sound_cues = [c for c in win_cues if c.track == "SOUND_FX"]
    camera_cues = [c for c in win_cues if c.track == "CAMERA"]

    # 1. Hook Velocity (0-20): first 6.5 seconds of the first segment
    first_seg = segments[0]
    hook_cutoff = first_seg.start_sec + 6.5
    hook_cues = [
        c
        for c in win_cues
        if c.start_sec < hook_cutoff and c.end_sec > first_seg.start_sec
    ]
    hook_text = " ".join(c.text for c in hook_cues)
    hook_words = set(_tokenize_words(hook_text))
    kinetic_hits = len(hook_words.intersection(KINETIC_HOOK_TERMS))
    has_early_speech = any(c.track == "DIALOGUE" for c in hook_cues)
    has_early_exclamation = ("!" in hook_text) or ("?" in hook_text)
    high_emotion_opening = any(
        c.emotion
        and any(
            em in c.emotion.lower()
            for em in (
                "cold",
                "command",
                "venom",
                "furious",
                "fury",
                "shock",
                "terrif",
                "defian",
                "panick",
                "cruel",
                "mock",
                "plead",
                "desperat",
                "seduct",
                "lethal",
                "thunder",
                "alarm",
            )
        )
        for c in hook_cues
    )
    hook_velocity = min(
        20.0,
        8.5
        + min(6.0, kinetic_hits * 2.0)
        + (2.5 if has_early_speech else 0.0)
        + (1.5 if has_early_exclamation else 0.0)
        + (1.5 if high_emotion_opening else 0.0),
    )

    # 2. Conflict Escalation (0-20): speaker turn-taking, stakes vocabulary, confrontation density
    speakers_ordered = [c.speaker for c in dialogue_cues if c.speaker]
    unique_speakers = list(dict.fromkeys(speakers_ordered))
    speaker_switches = sum(
        1
        for i in range(1, len(speakers_ordered))
        if speakers_ordered[i] != speakers_ordered[i - 1]
    )
    all_dialogue_text = " ".join(c.text for c in dialogue_cues)
    all_win_text = " ".join(c.text for c in win_cues)
    stakes_hits = len(set(_tokenize_words(all_win_text)).intersection(CONFLICT_STAKES_TERMS))
    exclamation_count = all_dialogue_text.count("!") + all_dialogue_text.count("?")

    conflict_escalation = min(
        20.0,
        7.0
        + min(4.5, speaker_switches * 1.1)
        + (2.5 if len(unique_speakers) >= 2 else 0.0)
        + min(4.0, stakes_hits * 0.65)
        + min(2.0, exclamation_count * 0.35),
    )

    # 3. Cliffhanger Peak (0-20): final 8.5 seconds of the closing segment
    last_seg = segments[-1]
    cliff_start = max(last_seg.start_sec, last_seg.end_sec - 8.5)
    cliff_cues = [
        c
        for c in win_cues
        if c.end_sec > cliff_start and c.start_sec <= last_seg.end_sec
    ]
    cliff_text_lower = " ".join(c.text.lower() for c in cliff_cues)
    explicit_cliff = "cliffhanger" in cliff_text_lower
    has_sfx_sting = any(c.track == "SOUND_FX" for c in cliff_cues)
    av_cliff_hits = sum(
        1 for term in CLIFFHANGER_AUDIO_VISUAL_TERMS if term in cliff_text_lower
    )
    ends_near_episode_tail = (episode.total_duration_sec - last_seg.end_sec) <= 5.0
    closing_dialogue = dialogue_cues[-1].text if dialogue_cues else ""
    closing_punct_boost = (
        1.5
        if ("?" in closing_dialogue or "!" in closing_dialogue or "—" in closing_dialogue)
        else 0.0
    )

    cliffhanger_peak = min(
        20.0,
        8.0
        + (4.0 if explicit_cliff else 0.0)
        + (2.5 if has_sfx_sting else 0.0)
        + min(3.5, av_cliff_hits * 0.9)
        + (1.5 if ends_near_episode_tail else 0.5)
        + closing_punct_boost,
    )

    # 4. Standalone Clarity (0-20): Spoken-Only Lore Filter (Outside-Voice Finding #3)
    # Only count proper-noun lore tokens that are SPOKEN aloud inside DIALOGUE cues.
    spoken_words = set(_tokenize_words(all_dialogue_text))
    spoken_lore_found = sorted(
        w
        for w in spoken_words.intersection(speaker_lore_tokens)
        if not w.endswith("'s")
    )
    role_anchors_found = len(
        set(_tokenize_words(all_win_text)).intersection(UNIVERSAL_ROLE_ANCHORS)
    )
    # 0-2 spoken character names is natural in a 40s scene; >=3 names starts confusing cold viewers
    excess_lore_count = max(0, len(spoken_lore_found) - 2)
    lore_penalty = min(8.0, excess_lore_count * 2.2)
    role_clarity_bonus = min(3.5, role_anchors_found * 0.7)
    standalone_clarity = max(
        6.0, min(20.0, 16.5 + role_clarity_bonus - lore_penalty)
    )

    # 5. Duration & Pacing Fit (0-20):
    # Hard gate: outside [20.0s, 60.0s] gets 0.
    if duration_sec < MIN_PROMO_DURATION or duration_sec > MAX_PROMO_DURATION:
        duration_pacing_fit = 0.0
    else:
        # Sweet spot is 28.0s - 48.0s for ReelShort / TikTok completion
        if 28.0 <= duration_sec <= 48.0:
            dur_base = 16.5
        elif 24.0 <= duration_sec < 28.0 or 48.0 < duration_sec <= 54.0:
            dur_base = 15.0
        else:
            dur_base = 13.5
        spoken_dur = sum(
            max(0.0, min(seg.end_sec, c.end_sec) - max(seg.start_sec, c.start_sec))
            for seg in segments
            for c in episode.dialogue_cues
        )
        dialogue_ratio = min(1.0, max(0.0, spoken_dur / max(1.0, duration_sec)))
        pacing_bonus = 3.5 if 0.35 <= dialogue_ratio <= 0.85 else 1.8
        duration_pacing_fit = min(20.0, dur_base + pacing_bonus)

    stage1_score = round(
        hook_velocity
        + conflict_escalation
        + cliffhanger_peak
        + standalone_clarity
        + duration_pacing_fit,
        2,
    )

    span_min_start = min(s.start_sec for s in segments)
    span_max_end = max(s.end_sec for s in segments)

    cand_id = (
        f"ep{episode.episode_num:03d}_{int(round(span_min_start))}s_{int(round(span_max_end))}s"
        if promo_type == "single_segment"
        else f"ep{episode.episode_num:03d}_{promo_type}_{int(round(first_seg.start_sec))}s_{int(round(last_seg.end_sec))}s"
    )

    if not materialize_text:
        return PromoCandidate(
            candidate_id=cand_id,
            episode_num=episode.episode_num,
            episode_filename=episode.filename,
            promo_type=promo_type,
            start_sec=span_min_start,
            end_sec=span_max_end,
            duration_sec=duration_sec,
            segments=segments,
            hook_velocity=hook_velocity,
            conflict_escalation=conflict_escalation,
            cliffhanger_peak=cliffhanger_peak,
            standalone_clarity=standalone_clarity,
            duration_pacing_fit=duration_pacing_fit,
            stage1_score=stage1_score,
            final_score=stage1_score,
            llm_score=None,
            scoring_provider="stage1_structural",
            rescued_fallback=False,
            dialogue_safe=True,
            explicit_cliffhanger_flag=explicit_cliff,
            spoken_lore_tokens_found=spoken_lore_found,
            speakers_in_clip=unique_speakers,
            opening_hook_line="",
            closing_cliffhanger_line="",
            withheld_payoff_preview="",
            payoff_source="same_episode",
            rationale="",
            editorial_risk_note="",
            excerpt_lines=[],
        )

    # Extract opening hook, closing cliffhanger, and withheld payoff
    opening_hook_line = ""
    for c in hook_cues:
        if c.track in {"DIALOGUE", "ACTION"}:
            if c.track == "DIALOGUE" and is_continuation_dialogue_cue(c):
                continue
            prefix = f"{c.speaker}: " if c.speaker else f"[{c.track}] "
            opening_hook_line = f"({c.start_str}) {prefix}{c.text}"
            if c.track == "DIALOGUE":
                break
    if not opening_hook_line and win_cues:
        c0 = win_cues[0]
        opening_hook_line = f"({c0.start_str}) [{c0.track}] {c0.text}"

    # Combine final spoken dialogue line in cliff_cues with any trailing SOUND_FX sting
    dlg_in_win = [c for c in win_cues if c.track == "DIALOGUE"]
    last_dlg_cliff = next(
        (c for c in reversed(cliff_cues) if c.track == "DIALOGUE"), None
    )
    last_sfx_cliff = next(
        (c for c in reversed(cliff_cues) if c.track == "SOUND_FX"), None
    )
    if last_dlg_cliff is not None:
        dlg_start_str = last_dlg_cliff.start_str
        dlg_text = last_dlg_cliff.text
        if is_continuation_dialogue_cue(last_dlg_cliff) and len(dlg_in_win) >= 2:
            idx_last = next(
                (i for i, c in enumerate(dlg_in_win) if c is last_dlg_cliff), -1
            )
            if idx_last > 0:
                prev_dlg = dlg_in_win[idx_last - 1]
                dlg_start_str = prev_dlg.start_str
                dlg_text = f"{prev_dlg.text.rstrip('\"')} {last_dlg_cliff.text.lstrip('\"')}"
        dlg_summary = (
            f"({dlg_start_str}-{last_dlg_cliff.end_str}) {last_dlg_cliff.speaker}: {dlg_text}"
        )
        if last_sfx_cliff:
            closing_cliffhanger_line = (
                f"{dlg_summary} + [{last_sfx_cliff.start_str} SOUND_FX] {last_sfx_cliff.text}"
            )
        else:
            closing_cliffhanger_line = dlg_summary
    elif last_sfx_cliff:
        closing_cliffhanger_line = (
            f"({last_sfx_cliff.start_str}-{last_sfx_cliff.end_str}) [SOUND_FX] {last_sfx_cliff.text}"
        )
    else:
        clast = cliff_cues[-1] if cliff_cues else win_cues[-1]
        closing_cliffhanger_line = f"({clast.start_str}-{clast.end_str}) [{clast.track}] {clast.text}"

    withheld_payoff_preview, payoff_source = extract_withheld_payoff(
        episode, span_max_end, next_episode=next_episode
    )

    excerpt_lines: List[str] = []
    for c in win_cues:
        if c.track == "DIALOGUE":
            emo = f" ({c.emotion})" if c.emotion else ""
            excerpt_lines.append(f"[{c.start_str}-{c.end_str}] {c.speaker}{emo}: {c.text}")
        elif c.track in {"ACTION", "SOUND_FX"}:
            excerpt_lines.append(f"[{c.start_str}-{c.end_str}] [{c.track}] {c.text}")

    rationale = (
        f"Opens at {first_seg.start_str} with immediate dramatic confrontation "
        f"(Hook {hook_velocity:.1f}/20), escalates through {speaker_switches} speaker turns "
        f"between {', '.join(unique_speakers[:3]) or 'lead characters'} (Conflict {conflict_escalation:.1f}/20), "
        f"and cuts at {last_seg.end_str} ({duration_sec:.1f}s) on a high-tension peak "
        f"(Cliffhanger {cliffhanger_peak:.1f}/20) while withholding {payoff_source.replace('_', ' ')} payoff."
    )

    if len(spoken_lore_found) <= 2:
        risk_note = (
            f"Low cold-viewer friction: only {len(spoken_lore_found)} proper name(s) spoken aloud "
            f"({', '.join(spoken_lore_found) if spoken_lore_found else 'none'}); conflict reads via universal roles."
        )
    else:
        risk_note = (
            f"Moderate lore density: {len(spoken_lore_found)} character/place names spoken aloud "
            f"({', '.join(spoken_lore_found)}); visual blocking carries the stakes."
        )

    return PromoCandidate(
        candidate_id=cand_id,
        episode_num=episode.episode_num,
        episode_filename=episode.filename,
        promo_type=promo_type,
        start_sec=span_min_start,
        end_sec=span_max_end,
        duration_sec=duration_sec,
        segments=segments,
        hook_velocity=hook_velocity,
        conflict_escalation=conflict_escalation,
        cliffhanger_peak=cliffhanger_peak,
        standalone_clarity=standalone_clarity,
        duration_pacing_fit=duration_pacing_fit,
        stage1_score=stage1_score,
        final_score=stage1_score,
        llm_score=None,
        scoring_provider="stage1_structural",
        rescued_fallback=False,
        dialogue_safe=True,
        explicit_cliffhanger_flag=explicit_cliff,
        spoken_lore_tokens_found=spoken_lore_found,
        speakers_in_clip=unique_speakers,
        opening_hook_line=opening_hook_line,
        closing_cliffhanger_line=closing_cliffhanger_line,
        withheld_payoff_preview=withheld_payoff_preview,
        payoff_source=payoff_source,
        rationale=rationale,
        editorial_risk_note=risk_note,
        excerpt_lines=excerpt_lines[:14],
    )


def temporal_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Compute 1D temporal Intersection-over-Union (IoU) between two intervals."""
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0.0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0.0 else 0.0


def nms_deduplicate(
    candidates: Sequence[PromoCandidate],
    max_keep: int = 3,
    iou_threshold: float = NMS_IOU_THRESHOLD,
) -> List[PromoCandidate]:
    """Non-Maximum Suppression across candidates from the same episode."""
    sorted_cands = sorted(
        candidates,
        key=lambda c: (c.final_score, c.cliffhanger_peak, c.hook_velocity),
        reverse=True,
    )
    kept: List[PromoCandidate] = []
    for cand in sorted_cands:
        if any(
            cand.episode_num == existing.episode_num
            and temporal_iou(cand.start_sec, cand.end_sec, existing.start_sec, existing.end_sec)
            >= iou_threshold
            for existing in kept
        ):
            continue
        kept.append(cand)
        if len(kept) >= max_keep:
            break
    return kept


def generate_episode_candidates(
    episode: EpisodeScript,
    speaker_lore_tokens: Set[str],
    next_episode: Optional[EpisodeScript] = None,
    max_per_episode: int = 3,
    min_duration: float = MIN_PROMO_DURATION,
    max_duration: float = MAX_PROMO_DURATION,
) -> List[PromoCandidate]:
    """Generate top-K deduplicated, dialogue-safe promo windows for an episode."""
    dialogue_cues = episode.dialogue_cues

    # Collect all dialogue-safe cut points from cue starts and ends
    raw_starts = sorted(
        {
            round(c.start_sec, 2)
            for c in episode.cues
            if is_dialogue_safe_start_timestamp(c.start_sec, dialogue_cues)
        }
    )
    raw_ends = sorted(
        {
            round(c.end_sec, 2)
            for c in episode.cues
            if is_dialogue_safe_end_timestamp(c.end_sec, dialogue_cues)
        }
    )

    if not raw_starts:
        raw_starts = sorted({round(c.start_sec, 2) for c in episode.cues})
    if not raw_ends:
        raw_ends = sorted({round(c.end_sec, 2) for c in episode.cues})

    all_windows: List[PromoCandidate] = []
    # Stride start points so adjacent camera cues within 1.2s don't cause O(N^2) redundant scoring
    filtered_starts: List[float] = []
    for s_sec in raw_starts:
        if not filtered_starts or (s_sec - filtered_starts[-1]) >= 1.5 or s_sec == 0.0:
            filtered_starts.append(s_sec)

    for s_sec in filtered_starts:
        for e_sec in raw_ends:
            dur = round(e_sec - s_sec, 2)
            if min_duration <= dur <= max_duration:
                cand = score_window_structural(
                    episode=episode,
                    start_sec=s_sec,
                    end_sec=e_sec,
                    speaker_lore_tokens=speaker_lore_tokens,
                    next_episode=next_episode,
                    materialize_text=False,
                )
                if min_duration != MIN_PROMO_DURATION or max_duration != MAX_PROMO_DURATION:
                    cand.duration_pacing_fit = 18.0
                    cand.stage1_score = round(
                        cand.hook_velocity
                        + cand.conflict_escalation
                        + cand.cliffhanger_peak
                        + cand.standalone_clarity
                        + cand.duration_pacing_fit,
                        2,
                    )
                    cand.final_score = cand.stage1_score
                all_windows.append(cand)

    # Fallback if an episode has ultra-sparse dialogue boundaries
    if not all_windows and episode.total_duration_sec >= min_duration:
        s_sec = 0.0
        e_sec = min(max_duration, episode.total_duration_sec)
        all_windows.append(
            score_window_structural(
                episode=episode,
                start_sec=s_sec,
                end_sec=e_sec,
                speaker_lore_tokens=speaker_lore_tokens,
                next_episode=next_episode,
                materialize_text=True,
            )
        )

    winners = nms_deduplicate(all_windows, max_keep=max_per_episode)
    materialized = [
        score_window_structural(
            episode=episode,
            start_sec=w.start_sec,
            end_sec=w.end_sec,
            speaker_lore_tokens=speaker_lore_tokens,
            next_episode=next_episode,
            segments=w.segments,
            promo_type=w.promo_type,
            materialize_text=True,
        )
        for w in winners
    ]
    if min_duration != MIN_PROMO_DURATION or max_duration != MAX_PROMO_DURATION:
        for m in materialized:
            m.duration_pacing_fit = 18.0
            m.stage1_score = round(
                m.hook_velocity
                + m.conflict_escalation
                + m.cliffhanger_peak
                + m.standalone_clarity
                + m.duration_pacing_fit,
                2,
            )
            m.final_score = m.stage1_score
    return materialized


def generate_stitched_candidates(
    episode: EpisodeScript,
    speaker_lore_tokens: Set[str],
    next_episode: Optional[EpisodeScript] = None,
) -> List[PromoCandidate]:
    """Generate multi-segment stitched promos using the 2 ReelShort stitch templates.

    Template 1 (multi_segment_ellipsis):
      Two dialogue-safe spans in the same episode separated by an >=8s middle gap,
      where each span is >=6.0s and total duration is in [20.0s, 60.0s].
    Template 2 (multi_segment_cold_open):
      A 6.0s-12.0s high-cliffhanger teaser clip from the end of the episode prepended
      as a cold-open flash-forward before the 20.0s-44.0s setup confrontation.
    """
    dialogue_cues = episode.dialogue_cues
    safe_starts = sorted(
        {
            round(t, 2)
            for c in episode.cues
            for t in (c.start_sec, c.end_sec)
            if is_dialogue_safe_start_timestamp(t, dialogue_cues)
        }
    )
    safe_ends = sorted(
        {
            round(t, 2)
            for c in episode.cues
            for t in (c.start_sec, c.end_sec)
            if is_dialogue_safe_end_timestamp(t, dialogue_cues)
        }
    )
    if len(safe_starts) < 2 or len(safe_ends) < 2 or episode.total_duration_sec < 40.0:
        return []

    stitched: List[PromoCandidate] = []

    # Template 1: Intra-Episode Ellipsis (early setup + late climax, removing slow middle)
    early_starts = [t for t in safe_starts if t <= 25.0][:4]
    early_ends = [t for t in safe_ends if 12.0 <= t <= 42.0][:5]
    late_starts = [
        t
        for t in safe_starts
        if (episode.total_duration_sec - 45.0) <= t <= (episode.total_duration_sec - 12.0)
    ][:5]
    late_ends = [
        t for t in safe_ends if t >= (episode.total_duration_sec - 15.0)
    ][-3:]

    best_ellipsis: Optional[PromoCandidate] = None
    for s1 in early_starts:
        for e1 in early_ends:
            d1 = round(e1 - s1, 2)
            if d1 < 10.0 or d1 > 28.0:
                continue
            for s2 in late_starts:
                if (s2 - e1) < 10.0:
                    continue
                for e2 in late_ends:
                    d2 = round(e2 - s2, 2)
                    if d2 < 10.0 or d2 > 32.0:
                        continue
                    total_dur = round(d1 + d2, 2)
                    if not (MIN_PROMO_DURATION <= total_dur <= MAX_PROMO_DURATION):
                        continue
                    segs = [
                        SegmentSpan(episode.episode_num, s1, e1, label="setup_hook"),
                        SegmentSpan(episode.episode_num, s2, e2, label="climax_cliffhanger"),
                    ]
                    cand = score_window_structural(
                        episode=episode,
                        start_sec=s1,
                        end_sec=e2,
                        speaker_lore_tokens=speaker_lore_tokens,
                        next_episode=next_episode,
                        segments=segs,
                        promo_type="multi_segment_ellipsis",
                    )
                    cand.rationale = (
                        f"[Multi-Segment Ellipsis Stitch] Pairs setup hook ({format_timestamp(s1)}-{format_timestamp(e1)}, {d1:.1f}s) "
                        f"with climax cliffhanger ({format_timestamp(s2)}-{format_timestamp(e2)}, {d2:.1f}s), "
                        f"cutting {s2 - e1:.1f}s of middle exposition for a {total_dur:.1f}s high-retention ad."
                    )
                    if best_ellipsis is None or cand.stage1_score > best_ellipsis.stage1_score:
                        best_ellipsis = cand

    if best_ellipsis is not None:
        stitched.append(best_ellipsis)

    # Template 2: Cold-Open Flash-Forward (6-12s peak cliffhanger teaser + 20-44s setup arc)
    teaser_end = safe_ends[-1]
    teaser_starts = [
        t for t in safe_starts if 6.0 <= (teaser_end - t) <= 12.0
    ]
    body_starts = [t for t in safe_starts if t <= 18.0][:3]
    body_ends = [
        t for t in safe_ends if 26.0 <= t <= min(65.0, teaser_end - 14.0)
    ][:4]

    best_cold_open: Optional[PromoCandidate] = None
    for ts in teaser_starts[:2]:
        d_teaser = round(teaser_end - ts, 2)
        if d_teaser < MIN_STITCH_SEGMENT_DURATION:
            continue
        for bs in body_starts:
            for be in body_ends:
                if be >= ts:
                    continue
                d_body = round(be - bs, 2)
                if d_body < 16.0:
                    continue
                total_dur = round(d_teaser + d_body, 2)
                if not (MIN_PROMO_DURATION <= total_dur <= MAX_PROMO_DURATION):
                    continue
                segs = [
                    SegmentSpan(episode.episode_num, ts, teaser_end, label="cold_open_teaser"),
                    SegmentSpan(episode.episode_num, bs, be, label="escalation_arc"),
                ]
                cand = score_window_structural(
                    episode=episode,
                    start_sec=bs,
                    end_sec=teaser_end,
                    speaker_lore_tokens=speaker_lore_tokens,
                    next_episode=next_episode,
                    segments=segs,
                    promo_type="multi_segment_cold_open",
                )
                cand.rationale = (
                    f"[Cold-Open Flash-Forward Stitch] Opens with a {d_teaser:.1f}s climax teaser "
                    f"({format_timestamp(ts)}-{format_timestamp(teaser_end)}) before rewinding to the "
                    f"{d_body:.1f}s confrontation buildup ({format_timestamp(bs)}-{format_timestamp(be)}), "
                    f"totaling {total_dur:.1f}s."
                )
                if best_cold_open is None or cand.stage1_score > best_cold_open.stage1_score:
                    best_cold_open = cand

    if best_cold_open is not None:
        stitched.append(best_cold_open)

    return stitched


def build_all_candidates(
    episodes: Sequence[EpisodeScript],
    max_single_per_episode: int = 3,
    include_stitched: bool = True,
) -> Tuple[List[PromoCandidate], Set[str]]:
    """Run Stage-1 candidate generation across all episodes in the dataset."""
    lore_tokens = extract_speaker_lore_tokens(episodes)
    ep_by_num = {ep.episode_num: ep for ep in episodes}

    all_candidates: List[PromoCandidate] = []
    for ep in episodes:
        next_ep = ep_by_num.get(ep.episode_num + 1)
        singles = generate_episode_candidates(
            episode=ep,
            speaker_lore_tokens=lore_tokens,
            next_episode=next_ep,
            max_per_episode=max_single_per_episode,
        )
        all_candidates.extend(singles)

        if include_stitched and singles and singles[0].stage1_score >= 76.0:
            stitches = generate_stitched_candidates(
                episode=ep,
                speaker_lore_tokens=lore_tokens,
                next_episode=next_ep,
            )
            all_candidates.extend(stitches[:1])

    all_candidates.sort(key=lambda c: c.final_score, reverse=True)
    return all_candidates, lore_tokens
