# Quality Judgement & Editorial Failure Audit (`quality_audit_failures.md`)

Per the Kuku AI Generalist evaluation rubric (*Quality Judgement: Can you tell when the output is good vs. mediocre? Did you include examples of where AI got it wrong?*), below are **4 real failure cases** discovered while auditing our pipeline on the 50-episode screenplay pack, how we diagnosed them, and the guardrails we built.

---

## Failure Case 1: The Concurrent Track Mid-Sentence Guillotine (Episode 1, `00:52.40`)
- **What happened in V0:** Our initial timestamp regex treated every `Start: (...) | End: (...)` block as a sequential list. On Episode 1, it proposed `00:11.00 - 00:52.40` (41.4s).
- **Why it was mediocre/broken:** Inspecting the raw script revealed that across the 50 episodes, **2,392 out of 3,727 cue pairs overlap in time** because `**CAMERA:**`, `**ACTION:**`, `**SOUND FX:**`, and `**CHARACTER:**` run on parallel tracks. Cutting at `00:52.40` sliced right through the middle of an active `CAMERA` push-in (`00:52.00 - 00:54.60`) and clipped the reaction beat.
- **Engineering Fix (`src/parser.py` + `src/candidates.py`):** Separated cues into 4 concurrent tracks and enforced `is_dialogue_safe_timestamp()`, guaranteeing that cut points never intersect an active spoken interval (`00:11.00 - 00:57.00` or `01:29.50 - 02:17.00`).

## Failure Case 2: The False-Positive 'Lore Penalty' on Stage Directions (Episode 14 & Episode 22)
- **What happened in V0:** To keep promos self-contained for cold TikTok viewers, we penalized windows mentioning character names (`Juliana`, `Alexander`, `Clarissa`, `Catherine`).
- **Why it was mediocre/broken:** In screenplay format, `**CAMERA:**` and `**ACTION:**` lines repeat character names in every sentence for production crew tracking (e.g., *'Tight close-up on Juliana's hand'*). A cold viewer watching the video **never hears camera directions**! Penalizing stage directions wrongly crushed silent physical-action hooks.
- **Engineering Fix (`src/candidates.py`):** Restricted the proper-noun lore filter strictly to **spoken `DIALOGUE` cues** and extracted the character name vocabulary dynamically from the show's speaker tags (`extract_speaker_lore_tokens()`), making the filter 100% show-agnostic for new uploads.

## Failure Case 3: The 'Zero-Second' Cliffhanger Payoff Blindspot (22 Episodes including Ep 1, Ep 15, Ep 37)
- **What happened in V0:** Our rule for verifying a true cliffhanger checked the `[end_sec, min(scene_end, end_sec + 6.0)]` window for the withheld resolution.
- **Why it was mediocre/broken:** In vertical microdramas, the strongest cliffhangers happen at the **very end of the episode** (22 of the 50 episodes explicitly contain `'cliffhanger'` or `'cut to black'` in their final cues). When `end_sec == scene_end`, `min(scene_end, end_sec + 6.0)` was a 0-second empty window!
- **Engineering Fix (`src/candidates.py`):** Built `extract_withheld_payoff()` with cross-episode lookahead: whenever `end_sec` lands within `5.0s` of the episode end, it automatically pulls the opening 6 seconds + `CONTINUITY NOTES` of **Episode `N+1`** as the verified withheld payoff.

## Failure Case 4: The 'Slow Middle Lull' in Multi-Location Episodes (Episode 18 & Episode 30)
- **What happened in V0:** Some episodes open with an explosive 15-second confrontation, spend 25 seconds walking down a corridor or changing rooms, and end with a 20-second twist. A single contiguous `20-60s` window had to include the 25-second corridor lull.
- **Why it was mediocre/broken:** Dead air between seconds 15 and 40 tanks short-form ad completion rates.
- **Engineering Fix (`src/candidates.py`):** Added two ReelShort multi-segment stitch templates (`multi_segment_ellipsis` and `multi_segment_cold_open`) that bridge non-overlapping dialogue-safe segments (`>=6.0s` each, summing to `20.0s-60.0s`) and export ready-to-run FFmpeg `filter_complex` concat commands.
