# 🎬 Kuku Promo Finder Studio — AI Generalist Assignment (Problem 1)

**Problem Selected:** Problem 1 — *Find the Promos (Ads)* across the 50-episode microdrama screenplay pack (*I Married a Broken Soldier. He was a Duke*, 100 scenes, 3,727 concurrent cues, 102.8 minutes total runtime).

---

## 📦 Submission Deliverables Map

| Deliverable Required by Kuku Brief (p. 4) | File Path in Repository | Description |
|---|---|---|
| **1. Solution Doc (PDF, <= 2 pages)** | [`Kuku_Solution_Doc.pdf`](file:///home/venkat/projects/kuku/Kuku_Solution_Doc.pdf) ([source `.md`](file:///home/venkat/projects/kuku/docs/Kuku_Solution_Doc.md)) | Covers Problem Decomposition, AI Leverage & Model Swapping, Prototype Guide, Quality Judgement (*Where AI Got It Wrong*), and 100x Scale Unit Economics. |
| **2. Working Prototype (`app.py` + `cli.py`)** | [`app.py`](file:///home/venkat/projects/kuku/app.py), [`cli.py`](file:///home/venkat/projects/kuku/cli.py) | 4-tab Streamlit Promo Studio UI + CLI batch runner with 1-click model swapping (`semantic_reranker`, `heuristic`, `gemini`, `openai`, `anthropic`, `cli`), live `.zip`/`.txt` upload, and interactive constraint sliders for the shortlist interview. |
| **3. Raw Outputs (`raw_outputs/`)** | [`raw_outputs/`](file:///home/venkat/projects/kuku/raw_outputs) | Unedited outputs across all 50 episodes: [`ranked_promos_top20.md`](file:///home/venkat/projects/kuku/raw_outputs/ranked_promos_top20.md) / [`.json`](file:///home/venkat/projects/kuku/raw_outputs/ranked_promos_top20.json), [`all_50_episodes_promos.csv`](file:///home/venkat/projects/kuku/raw_outputs/all_50_episodes_promos.csv) / [`.json`](file:///home/venkat/projects/kuku/raw_outputs/all_50_episodes_promos.json), [`where_ai_got_it_wrong.md`](file:///home/venkat/projects/kuku/raw_outputs/where_ai_got_it_wrong.md), [`quality_audit_failures.md`](file:///home/venkat/projects/kuku/raw_outputs/quality_audit_failures.md), [`editor_cut_sheet.csv`](file:///home/venkat/projects/kuku/raw_outputs/editor_cut_sheet.csv), [`ffmpeg_slice_commands.sh`](file:///home/venkat/projects/kuku/raw_outputs/ffmpeg_slice_commands.sh), and [`model_swap_diff.json`](file:///home/venkat/projects/kuku/raw_outputs/model_swap_diff.json). |
| **4. Automated Verification Suite** | [`tests/test_pipeline.py`](file:///home/venkat/projects/kuku/tests/test_pipeline.py) | 12 unit, E2E, and LLM eval tests verifying 4-track parsing, scene-clock reset normalization, `20.0s-60.0s` bounds, dialogue-safe cuts, cross-episode `Episode N+1` payoffs, spoken-only lore filtering, NMS deduplication, multi-segment stitches, and 3-layer LLM rescue. |

---

## 🚀 Quickstart (`< 30 seconds`)

```bash
# 1. Activate virtualenv (or install via uv / pip)
source .venv/bin/activate

# 2. Run the 50-episode CLI batch pipeline (completes in ~6s)
python cli.py --provider semantic_reranker --output-dir raw_outputs

# 3. Swap Stage-2 AI Judge Model (Gemini / OpenAI / Anthropic / Local Claude CLI / Heuristic)
python cli.py --provider gemini --model gemini-2.5-flash
python cli.py --provider heuristic

# 4. Launch the interactive Streamlit Promo Studio UI
streamlit run app.py

# 5. Run the 12-test automated pytest verification suite
pytest -v
```

## 🔒 Confidentiality Guard
Per page 5 of the assignment brief (*"Please do not share this document or the datasets publicly"*), `.gitignore` excludes `*.zip` and raw extracted screenplay `.txt` files from public Git pushes while tracking `raw_outputs/` and a built-in synthetic demo fixture in `src/parser.py` so cloud deployments work out-of-the-box.
