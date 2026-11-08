# TODOS

## Phase-2 Multimodal Production Pipeline (`src/benchmark.py`, `cli.py`)

### Automated FFmpeg Rough-Cut Video Slicer & Concat Runner
- **Priority:** P2
- **Description:** Execute generated `raw_outputs/ffmpeg_slice_commands.sh` directly against source `.mp4` episode renders when video assets are mounted in `episodes/ep_XX.mp4`, producing ready-to-upload `9:16` vertical ad cuts in `cuts/`.

### Whisper word-level ASR Timestamp Alignment
- **Priority:** P2
- **Description:** Cross-check screenplay dialogue timestamps against `faster-whisper` word-level audio timestamps on rendered episodes to snap cut out-points within `+-50ms` of vocal onset/offset before final audio fade-out.

### Paid-Social CTR / Thumbstop Feedback Calibration Loop
- **Priority:** P3
- **Description:** Ingest Meta Reels and YouTube Shorts 3-second thumbstop rate and 15-second hold rate telemetry to auto-tune the 5 Stage-2 sub-score weights (`Hook`, `Conflict`, `Cliffhanger`, `Clarity`, `Pacing`) per genre.

---

## Completed

- **T1 (`src/parser.py`): Multi-format 4-track concurrent screenplay parser, scene-reset clock normalizer & dynamic speaker lore extractor**
  **Completed:** v0.1.0.0 (2026-11-08)
- **T2 (`src/candidates.py`): Dialogue-safe 20.0s-60.0s window generator, Episode N+1 payoff extractor, NMS dedup & 2 ReelShort stitch templates**
  **Completed:** v0.1.0.0 (2026-11-08)
- **T3 (`src/llm_judge.py`): Swappable Stage-2 AI Judge with macro-batching, SHA-256 disk cache & 3-layer error rescue**
  **Completed:** v0.1.0.0 (2026-11-08)
- **T4 (`src/benchmark.py`, `cli.py`, `app.py`): 4-tab Streamlit Promo Studio UI, CLI batch runner & live interview constraint sandbox**
  **Completed:** v0.1.0.0 (2026-11-08)
- **C1-C3 Cherry-Picks: 50-Episode Tension Heatmap, Scale Economics Simulator, Model-Swap Diff, Naive-LLM Failure Audit & FFmpeg Exporter**
  **Completed:** v0.1.0.0 (2026-11-08)
- **Deliverables & QA: 2-Page `Kuku_Solution_Doc.pdf`, 12 `raw_outputs/` files, 15 automated `pytest` tests & 100/100 browser QA pass**
  **Completed:** v0.1.0.0 (2026-11-08)
