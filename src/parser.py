"""Multi-format, multi-track screenplay parser and scene-clock normalizer.

ASCII DATA FLOW & MULTI-TRACK TIMELINE NORMALIZATION:
===============================================================================
  [.zip Archive / .txt Files / Uploaded Script / Synthetic Demo Fixture]
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 1. Header & Scene Block Extractor                                       │
  │    - EPISODE NUMBER, EPISODE LOGLINE, CONTINUITY NOTES                  │
  │    - SCENE 01..N headers with Scene start / Scene end                   │
  │    - Scene-Reset Clock Normalizer: if Scene K resets to 00:00.00 while  │
  │      prior_scene_end > 0, shifts all Scene K cues by +prior_scene_end.  │
  └─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 2. Concurrent 4-Track Cue Separation (fixes 2,392 overlapping cues)     │
  │    Track A: DIALOGUE  -> **CHARACTER (Emotion):** + spoken quote        │
  │    Track B: ACTION    -> **ACTION:** physical blocking & slap/reveal    │
  │    Track C: CAMERA    -> **CAMERA:** shot framing & cut-to-black        │
  │    Track D: SOUND_FX  -> **SOUND FX:** audio stings & cliffhanger peaks │
  └─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 3. Show-Agnostic Dynamic Speaker Lore Token Extractor                   │
  │    - Extracts proper-noun character names from speaker tags across the  │
  │      loaded pack (filtering out generic roles like GUARD / PRIEST).     │
  └─────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import os
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import zipfile

MAX_ARCHIVE_BYTES = 50 * 1024 * 1024  # 50 MB safety cap

GENERIC_ROLE_STOPWORDS: Set[str] = {
    "CAMERA",
    "ACTION",
    "SOUND",
    "FX",
    "SOUND FX",
    "SFX",
    "MATRIARCH",
    "GUARD",
    "GUARDS",
    "SOLDIER",
    "SOLDIERS",
    "SERVANT",
    "SERVANTS",
    "MAID",
    "MAIDS",
    "BUTLER",
    "PRIEST",
    "BISHOP",
    "MINISTER",
    "DOCTOR",
    "PHYSICIAN",
    "HEALER",
    "CROWD",
    "GUEST",
    "GUESTS",
    "NOBLE",
    "NOBLES",
    "NOBLEMAN",
    "NOBLEWOMAN",
    "LADY",
    "LORD",
    "DUKE",
    "DUCHESS",
    "DOWAGER",
    "FATHER",
    "MOTHER",
    "SISTER",
    "BROTHER",
    "MAN",
    "WOMAN",
    "BOY",
    "GIRL",
    "CHILD",
    "VOICE",
    "NARRATOR",
    "COACHMAN",
    "FOOTMAN",
    "MESSENGER",
    "CAPTAIN",
    "GENERAL",
    "OFFICER",
    "INQUISITOR",
    "CLERK",
    "MAGISTRATE",
    "WARDEN",
    "JAILER",
    "EXECUTIONER",
    "MERCHANT",
    "INNKEEPER",
    "VILLAGER",
    "VILLAGERS",
    "PEASANT",
    "GENTLEMAN",
    "WIDOW",
    "BRIDE",
    "GROOM",
    "YOUNG",
    "OLD",
    "TALL",
    "SHORT",
    "SCARRED",
    "HOODED",
    "MASKED",
    "FEMALE",
    "MALE",
    "FIRST",
    "SECOND",
    "THIRD",
    "IN",
    "THE",
    "OF",
    "AND",
    "WITH",
    "GOLD",
    "GOWN",
    "BLACK",
    "WHITE",
    "RED",
    "ARISTOCRATIC",
    "BLOND",
    "BALD",
    "ELDER",
    "HIGH",
    "MYSTERIOUS",
    "BEARDED",
    "STRANGER",
    "FORMER",
    "THUG",
    "SCOUT",
    "FIGHTER",
    "SENTRY",
    "TOWNSMAN",
    "DRIVER",
    "CART",
    "CAP",
    "OFFICIAL",
    "CROWN",
    "POLICE",
    "CONSTABLE",
    "JUDGE",
    "INFORMANT",
    "CONSPIRATOR",
    "LOYALIST",
    "HOUSEKEEPER",
    "OFF-SCREEN",
    "OFF",
    "SCREEN",
}


@dataclass
class Cue:
    """A single timed cue on one of the 4 concurrent screenplay tracks."""

    cue_index: int
    scene_num: int
    track: str  # "DIALOGUE" | "ACTION" | "CAMERA" | "SOUND_FX"
    raw_header: str
    speaker: Optional[str]
    emotion: Optional[str]
    start_sec: float
    end_sec: float
    text: str

    @property
    def duration_sec(self) -> float:
        return max(0.0, round(self.end_sec - self.start_sec, 3))

    @property
    def start_str(self) -> str:
        return format_timestamp(self.start_sec)

    @property
    def end_str(self) -> str:
        return format_timestamp(self.end_sec)


@dataclass
class SceneMeta:
    """Metadata and normalized boundaries for a screenplay scene."""

    scene_num: int
    heading: str
    raw_start_sec: float
    raw_end_sec: float
    norm_start_sec: float
    norm_end_sec: float
    clock_reset_applied: bool = False


@dataclass
class EpisodeScript:
    """Parsed multi-track representation of a single episode screenplay."""

    episode_num: int
    episode_id: str
    filename: str
    logline: str
    continuity_notes: str
    scenes: List[SceneMeta] = field(default_factory=list)
    cues: List[Cue] = field(default_factory=list)
    normalization_notes: List[str] = field(default_factory=list)

    @property
    def total_duration_sec(self) -> float:
        if not self.cues and not self.scenes:
            return 0.0
        max_cue = max((c.end_sec for c in self.cues), default=0.0)
        max_scene = max((s.norm_end_sec for s in self.scenes), default=0.0)
        return round(max(max_cue, max_scene), 2)

    @property
    def dialogue_cues(self) -> List[Cue]:
        return [c for c in self.cues if c.track == "DIALOGUE"]

    @property
    def action_cues(self) -> List[Cue]:
        return [c for c in self.cues if c.track == "ACTION"]

    @property
    def camera_cues(self) -> List[Cue]:
        return [c for c in self.cues if c.track == "CAMERA"]

    @property
    def sound_fx_cues(self) -> List[Cue]:
        return [c for c in self.cues if c.track == "SOUND_FX"]

    @property
    def has_explicit_cliffhanger(self) -> bool:
        tail_cutoff = max(0.0, self.total_duration_sec - 15.0)
        for cue in self.cues:
            if cue.end_sec >= tail_cutoff and "cliffhanger" in cue.text.lower():
                return True
        return False


def parse_timestamp_to_sec(ts_str: str) -> float:
    """Parse MM:SS.ms, HH:MM:SS.ms, or SS.ms into float seconds."""
    cleaned = ts_str.strip().strip("()[]{}")
    parts = cleaned.split(":")
    try:
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return round(hours * 3600 + minutes * 60 + seconds, 3)
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return round(minutes * 60 + seconds, 3)
        return round(float(parts[0]), 3)
    except ValueError as exc:
        raise ValueError(f"Unrecognized timestamp format: {ts_str!r}") from exc


def format_timestamp(seconds: float) -> str:
    """Format float seconds as MM:SS.ms (e.g., 01:18.50)."""
    sec_clamped = max(0.0, float(seconds))
    mins = int(sec_clamped // 60)
    rem = sec_clamped - (mins * 60)
    return f"{mins:02d}:{rem:05.2f}"


_EP_NUM_RE = re.compile(r"EPISODE\s+NUMBER\s*:\s*Episode\s*(\d+)", re.IGNORECASE)
_FILENAME_EP_RE = re.compile(r"episode[-_\s]*0*(\d+)", re.IGNORECASE)
_LOGLINE_RE = re.compile(r"EPISODE\s+LOGLINE\s*:\s*(.+)", re.IGNORECASE)
_CONTINUITY_RE = re.compile(r"CONTINUITY\s+NOTES\s*:\s*(.+)", re.IGNORECASE)

_SCENE_SPLIT_RE = re.compile(r"(?m)^SCENE\s+(\d+)\s*$")
_SCENE_HEADING_RE = re.compile(r"(?m)^Scene heading\s*:\s*(.+)$", re.IGNORECASE)
_SCENE_START_RE = re.compile(r"(?m)^Scene start\s*:\s*(\S+)", re.IGNORECASE)
_SCENE_END_RE = re.compile(r"(?m)^Scene end\s*:\s*(\S+)", re.IGNORECASE)

# Matches standard Kuku format and common variations:
# **HEADER:**
# Start: (00:00.00) | End: (00:04.80)
# text
_CUE_BLOCK_RE = re.compile(
    r"\*\*([^*]+?):\*\*\s*\n"
    r"Start:\s*\(?([0-9:.]+)\)?\s*\|\s*End:\s*\(?([0-9:.]+)\)?\s*\n"
    r"([\s\S]*?)(?=(?:\n\*\*[^*]+?:\*\*|\n--- SCRIPT END ---|\Z))",
    re.MULTILINE,
)

# Alternate inline bracket format: [00:11.00 - 00:15.20] CHARACTER (Emotion): Text
_ALT_INLINE_CUE_RE = re.compile(
    r"\[([0-9:.]+)\s*-\s*([0-9:.]+)\]\s*([A-Za-z0-9_ ()',.-]+?):\s*(.+)"
)


def _classify_header(raw_header: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Classify a cue header into (track, speaker, emotion)."""
    header = raw_header.strip()
    upper = header.upper()
    if upper == "CAMERA":
        return ("CAMERA", None, None)
    if upper == "ACTION":
        return ("ACTION", None, None)
    if upper in {"SOUND FX", "SOUND_FX", "SFX", "AUDIO", "SOUND"}:
        return ("SOUND_FX", None, None)

    # Otherwise it's a character dialogue header, e.g. "JULIANA (V.O.) (Breathless, Defiant)"
    if "(" in header:
        speaker = header.split("(", 1)[0].strip()
        parens = re.findall(r"\(([^)]+)\)", header)
        emotion = ", ".join(p.strip() for p in parens if p.strip()) or None
        return ("DIALOGUE", speaker, emotion)
    return ("DIALOGUE", header, None)


def _parse_cues_from_block(
    block_text: str,
    scene_num: int,
    time_offset: float,
    start_cue_idx: int,
) -> List[Cue]:
    """Extract timed cues from a scene text block, applying time_offset."""
    cues: List[Cue] = []
    idx = start_cue_idx

    matches = list(_CUE_BLOCK_RE.finditer(block_text))
    if matches:
        for m in matches:
            raw_header = m.group(1).strip()
            raw_start = parse_timestamp_to_sec(m.group(2))
            raw_end = parse_timestamp_to_sec(m.group(3))
            body = m.group(4).strip()
            track, speaker, emotion = _classify_header(raw_header)
            start_sec = round(raw_start + time_offset, 3)
            end_sec = round(max(start_sec, raw_end + time_offset), 3)
            cues.append(
                Cue(
                    cue_index=idx,
                    scene_num=scene_num,
                    track=track,
                    raw_header=raw_header,
                    speaker=speaker,
                    emotion=emotion,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    text=body,
                )
            )
            idx += 1
        return cues

    # Try alternate inline bracket format: [00:10.00 - 00:14.00] SPEAKER: line
    inline_matches = list(_ALT_INLINE_CUE_RE.finditer(block_text))
    if inline_matches:
        for m in inline_matches:
            raw_start = parse_timestamp_to_sec(m.group(1))
            raw_end = parse_timestamp_to_sec(m.group(2))
            raw_header = m.group(3).strip()
            body = m.group(4).strip()
            track, speaker, emotion = _classify_header(raw_header)
            start_sec = round(raw_start + time_offset, 3)
            end_sec = round(max(start_sec, raw_end + time_offset), 3)
            cues.append(
                Cue(
                    cue_index=idx,
                    scene_num=scene_num,
                    track=track,
                    raw_header=raw_header,
                    speaker=speaker,
                    emotion=emotion,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    text=body,
                )
            )
            idx += 1
        return cues

    # Fallback: untimed plain dialogue/action paragraphs estimated at 150 WPM (2.5 words/sec)
    cursor = time_offset
    for para in re.split(r"\n\s*\n", block_text.strip()):
        cleaned = para.strip()
        if not cleaned or cleaned.startswith("---") or cleaned.startswith("Actions and dialogue"):
            continue
        words = max(1, len(cleaned.split()))
        dur = max(2.0, round(words / 2.5, 2))
        speaker_match = re.match(r"^([A-Z][A-Z\s]{1,25}):\s*(.+)$", cleaned, re.DOTALL)
        if speaker_match:
            track, speaker, emotion = _classify_header(speaker_match.group(1))
            text_body = speaker_match.group(2).strip()
        else:
            track, speaker, emotion = ("ACTION", None, None)
            text_body = cleaned
        cues.append(
            Cue(
                cue_index=idx,
                scene_num=scene_num,
                track=track,
                raw_header=speaker or track,
                speaker=speaker,
                emotion=emotion,
                start_sec=round(cursor, 3),
                end_sec=round(cursor + dur, 3),
                text=text_body,
            )
        )
        cursor += dur
        idx += 1
    return cues


def parse_episode_text(text: str, filename: str = "episode-001.txt") -> EpisodeScript:
    """Parse raw episode screenplay text into a normalized EpisodeScript."""
    if not text or not text.strip():
        raise ValueError(f"Empty screenplay text in {filename!r}")

    ep_num = 1
    ep_match = _EP_NUM_RE.search(text)
    if ep_match:
        ep_num = int(ep_match.group(1))
    else:
        fn_match = _FILENAME_EP_RE.search(filename)
        if fn_match:
            ep_num = int(fn_match.group(1))

    logline_match = _LOGLINE_RE.search(text)
    logline = logline_match.group(1).strip() if logline_match else ""

    cont_match = _CONTINUITY_RE.search(text)
    continuity_notes = cont_match.group(1).strip() if cont_match else ""

    base_name = os.path.basename(filename)
    episode_id = f"ep{ep_num:03d}"

    scenes: List[SceneMeta] = []
    all_cues: List[Cue] = []
    normalization_notes: List[str] = []

    scene_splits = list(_SCENE_SPLIT_RE.finditer(text))
    if scene_splits:
        cumulative_offset = 0.0
        prior_norm_end = 0.0
        for idx, sm in enumerate(scene_splits):
            scene_num = int(sm.group(1))
            block_start = sm.end()
            block_end = (
                scene_splits[idx + 1].start()
                if idx + 1 < len(scene_splits)
                else len(text)
            )
            scene_block = text[block_start:block_end]

            heading_m = _SCENE_HEADING_RE.search(scene_block)
            heading = heading_m.group(1).strip() if heading_m else f"SCENE {scene_num:02d}"

            start_m = _SCENE_START_RE.search(scene_block)
            end_m = _SCENE_END_RE.search(scene_block)
            raw_start = parse_timestamp_to_sec(start_m.group(1)) if start_m else 0.0
            raw_end = parse_timestamp_to_sec(end_m.group(1)) if end_m else 0.0

            # Scene-Reset Clock Normalizer:
            # If Scene > 1 resets its clock to 00:00.00 (or < prior_norm_end - 1.0),
            # shift this scene's timestamps forward by prior_norm_end.
            clock_reset = False
            if idx > 0 and raw_start < (prior_norm_end - 1.0):
                cumulative_offset = prior_norm_end - raw_start
                clock_reset = True
                normalization_notes.append(
                    f"Scene {scene_num:02d} clock reset detected (raw {format_timestamp(raw_start)}); "
                    f"shifted by +{cumulative_offset:.2f}s."
                )

            scene_cues = _parse_cues_from_block(
                scene_block,
                scene_num=scene_num,
                time_offset=cumulative_offset,
                start_cue_idx=len(all_cues),
            )
            all_cues.extend(scene_cues)

            norm_start = round(raw_start + cumulative_offset, 3)
            max_cue_end = max((c.end_sec for c in scene_cues), default=0.0)
            norm_end = round(max(raw_end + cumulative_offset, max_cue_end), 3)
            prior_norm_end = norm_end

            scenes.append(
                SceneMeta(
                    scene_num=scene_num,
                    heading=heading,
                    raw_start_sec=raw_start,
                    raw_end_sec=raw_end,
                    norm_start_sec=norm_start,
                    norm_end_sec=norm_end,
                    clock_reset_applied=clock_reset,
                )
            )
    else:
        # Single-block script without explicit SCENE headers
        all_cues = _parse_cues_from_block(
            text, scene_num=1, time_offset=0.0, start_cue_idx=0
        )
        if not all_cues:
            raise ValueError(f"No valid cues found in {filename!r}")
        max_end = max((c.end_sec for c in all_cues), default=0.0)
        scenes.append(
            SceneMeta(
                scene_num=1,
                heading="SCENE 01",
                raw_start_sec=0.0,
                raw_end_sec=max_end,
                norm_start_sec=0.0,
                norm_end_sec=max_end,
                clock_reset_applied=False,
            )
        )
        normalization_notes.append("Synthesized single scene block from untimed/flat script.")

    if not all_cues:
        raise ValueError(f"Zero cues extracted from {filename!r}")

    return EpisodeScript(
        episode_num=ep_num,
        episode_id=episode_id,
        filename=base_name,
        logline=logline,
        continuity_notes=continuity_notes,
        scenes=scenes,
        cues=all_cues,
        normalization_notes=normalization_notes,
    )


def extract_speaker_lore_tokens(episodes: Iterable[EpisodeScript]) -> Set[str]:
    """Extract proper-noun character names dynamically from speaker tags across episodes.

    Returns lowercase proper-noun tokens (length >= 3) that appear in DIALOGUE speaker
    headers, excluding generic role titles (GUARD, PRIEST, MATRIARCH, etc.).
    """
    tokens: Set[str] = set()
    for ep in episodes:
        for cue in ep.dialogue_cues:
            if not cue.speaker:
                continue
            cleaned = re.sub(r"[^A-Za-z\s'-]", " ", cue.speaker).strip()
            if cleaned.upper() in GENERIC_ROLE_STOPWORDS:
                continue
            for word in cleaned.split():
                upper_w = word.upper().strip("'-")
                if len(upper_w) >= 3 and upper_w not in GENERIC_ROLE_STOPWORDS:
                    tokens.add(upper_w.lower())
    return tokens


def load_episodes_from_zip(
    zip_source: Union[str, Path, bytes, io.BytesIO],
) -> List[EpisodeScript]:
    """Load and sort all episode `.txt` files from a zip archive path or byte stream."""
    if isinstance(zip_source, (str, Path)):
        path = Path(zip_source)
        if not path.exists():
            raise FileNotFoundError(f"Archive not found: {path}")
        if path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError(
                f"Archive exceeds 50MB limit ({path.stat().st_size} bytes): {path}"
            )
        zf = zipfile.ZipFile(path, "r")
    elif isinstance(zip_source, bytes):
        if len(zip_source) > MAX_ARCHIVE_BYTES:
            raise ValueError(f"Uploaded archive exceeds 50MB limit ({len(zip_source)} bytes)")
        zf = zipfile.ZipFile(io.BytesIO(zip_source), "r")
    else:
        zf = zipfile.ZipFile(zip_source, "r")

    episodes: List[EpisodeScript] = []
    with zf:
        txt_names = sorted(
            n
            for n in zf.namelist()
            if n.lower().endswith(".txt")
            and not os.path.basename(n).startswith(".")
            and "__MACOSX" not in n
        )
        if not txt_names:
            raise ValueError("Zip archive contains zero .txt screenplay files.")

        for name in txt_names:
            raw_bytes = zf.read(name)
            try:
                text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = raw_bytes.decode("latin-1", errors="replace")
            ep = parse_episode_text(text, filename=name)
            episodes.append(ep)

    episodes.sort(key=lambda e: (e.episode_num, e.filename))
    return episodes


def load_synthetic_demo_episodes() -> List[EpisodeScript]:
    """Return a synthetic 2-episode microdrama demo fixture when the confidential zip is absent."""
    ep1_text = """EPISODE NUMBER: Episode 1
EPISODE LOGLINE: At the royal gala, disgraced surgeon Clara is publicly slapped by her stepmother before the mysterious wheelchair-bound Duke rises to claim her as his wife.
CONTINUITY NOTES: Pilot episode establishing Clara's betrayal and the Duke's secret recovery.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. GRAND BALLROOM - NIGHT
Scene start: 00:00.00
Scene end: 01:05.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:05.00)
Wide tracking shot across crystal chandeliers as Clara enters in a torn dress.

**ACTION:**
Start: (00:05.00) | End: (00:10.00)
Lady Vance marches forward and slaps Clara across the cheek in front of the gasped crowd.

**LADY VANCE (Venomous, Loud):**
Start: (00:10.00) | End: (00:16.50)
"You think a penniless bastard belongs at this altar? Kneel and beg your sister for scraps!"

**CLARA (Cold, Defiant):**
Start: (00:16.50) | End: (00:23.00)
"I saved your husband's life on the operating table last night. Touch me again and I pull the stitches!"

**SOUND FX:**
Start: (00:23.00) | End: (00:25.50)
Heavy oak double doors slam open with a thunderous boom.

**DUKE DAMIAN (Commanding, Ice-Cold):**
Start: (00:25.50) | End: (00:34.00)
"Anyone who asks my bride to kneel will lose their knees before dawn."

**ACTION:**
Start: (00:34.00) | End: (00:42.00)
Damian rises slowly from his wheelchair, sending shockwaves of terror through the ballroom.

**LADY VANCE (Panicked, Trembling):**
Start: (00:42.00) | End: (00:48.00)
"Impossible! The war left you paralyzed!"

**DUKE DAMIAN (Lethal Whisper):**
Start: (00:48.00) | End: (00:55.00)
"I stayed in that chair to see who would betray my house. Guards, seize the bride's family!"

**SOUND FX:**
Start: (00:55.00) | End: (00:58.00)
Ominous string crescendo peaks on a sharp cliffhanger sting.

**CAMERA:**
Start: (00:58.00) | End: (01:05.00)
Extreme close-up on Clara's stunned eyes as Damian places the ducal ring on her finger. Cut to black.

--- SCRIPT END ---
"""
    ep2_text = """EPISODE NUMBER: Episode 2
EPISODE LOGLINE: Dragged into the Duke's private carriage, Clara discovers a hidden dagger aimed at her throat and a marriage contract signed in blood.
CONTINUITY NOTES: Directly follows Episode 1 ballroom arrest; reveals Damian's true motive.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. DUCAL CARRIAGE - NIGHT
Scene start: 00:00.00
Scene end: 00:52.00

Actions and dialogue:

**ACTION:**
Start: (00:00.00) | End: (00:06.00)
Damian pins Clara against the velvet seat as the carriage races through the storm, pressing a silver scalpel near her collarbone.

**DUKE DAMIAN (Suspicious, Intense):**
Start: (00:06.00) | End: (00:14.00)
"You knew the poison in my veins was Nightshade. Who sent you into my camp, Doctor?"

**CLARA (Unflinching, Sharp):**
Start: (00:14.00) | End: (00:23.00)
"If I wanted you dead, My Lord, I wouldn't have traded my own blood to flush your heart."

**ACTION:**
Start: (00:23.00) | End: (00:31.00)
Clara pulls back her torn sleeve, revealing the fresh transfusion scar matching the bandage on Damian's wrist.

**DUKE DAMIAN (Stunned, Softening):**
Start: (00:31.00) | End: (00:39.00)
"Wait... the masked medic in the burning tent was you?"

**SOUND FX:**
Start: (00:39.00) | End: (00:43.00)
Gunshot shatters the carriage window glass in an explosive cliffhanger ambush.

**CAMERA:**
Start: (00:43.00) | End: (00:52.00)
Damian throws his body over Clara as armed riders surround the carriage. Cut to black.

--- SCRIPT END ---
"""
    return [
        parse_episode_text(ep1_text, "episode-001-synthetic.txt"),
        parse_episode_text(ep2_text, "episode-002-synthetic.txt"),
    ]
