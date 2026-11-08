# Changelog

All notable changes to Kuku Promo Finder Studio are documented in this file.

## [0.1.0.0] - 2026-11-08

### Added
- **Two-Stage Hybrid Promo Discovery Engine (`src/parser.py`, `src/candidates.py`, `src/llm_judge.py`):** Parse 50-episode microdrama screenplay packs across 4 concurrent tracks (`CAMERA`, `ACTION`, `SOUND FX`, `DIALOGUE`), normalize scene-reset clocks monotonically, enforce dialogue-safe `20.0s-60.0s` cut boundaries, extract cross-episode (`Episode N+1`) withheld payoffs, and generate single-segment plus multi-segment (`multi_segment_ellipsis`, `multi_segment_cold_open`) ad cuts.
- **Swappable Stage-2 AI Judge & 3-Layer Rescue (`src/llm_judge.py`):** Switch between `semantic_reranker`, `heuristic`, `gemini` (`gemini-2.5-flash`), `openai` (`gpt-4o-mini`), `anthropic` (`claude-3-5-haiku-latest`), and `cli` (`claude -p`) with 5-call macro-batching, SHA-256 disk caching (`.cache/llm_cache.json`), and automatic fallback on rate limits or malformed JSON.
- **Interactive 4-Tab Streamlit Promo Studio & CLI Runner (`app.py`, `cli.py`, `src/benchmark.py`):** Inspect Top-20 ranked promos with timed screenplay excerpts and 1-click FFmpeg commands, explore the 50-episode series tension heatmap and 1-to-200 shows/week unit-economics simulator, compare Stage-1 vs. Stage-2 model swaps alongside 4 documented naive-LLM failure modes, and run live custom duration constraints or unseen `.zip`/`.txt` uploads.
- **Complete 50-Episode Raw Outputs & 2-Page Solution Document (`raw_outputs/`, `Kuku_Solution_Doc.pdf`):** Export `ranked_promos_top20.json`/`.md`, `all_50_episodes_promos.json`/`.csv`, `editor_cut_sheet.csv`, `ffmpeg_slice_commands.sh`, `where_ai_got_it_wrong.md`, `quality_audit_failures.md`, `model_swap_diff.json`, `series_tension_heatmap.json`, `scale_economics_table.json`, and the 2-page PDF architecture brief.
- **Automated Test Suite (`tests/test_pipeline.py`):** Verify all 6 modules across 15 unit, integration, regression, and headless Streamlit `AppTest` tests.

### Fixed
- **Split-Sentence & Cliffhanger Boundary Safety (`src/candidates.py`, `src/llm_judge.py`, `app.py`):** Block mid-sentence `"..."` continuation cuts, stitch two-part closing cliffhanger lines across camera cuts, enforce Stage-2 AI Judge parity in the Tab 4 custom constraint lab, write SHA-256 disk cache files atomically via `os.replace`, and replace astral-plane emojis with crisp BMP glyphs for headless Linux Chromium.
