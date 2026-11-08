# 🎬 Kuku Promo Finder Studio — AI Generalist Assignment (Problem 1)

**🌐 Live Deployed Prototype (Streamlit Cloud):** [**https://kuku-promo-finder-studio.streamlit.app/**](https://kuku-promo-finder-studio.streamlit.app/)  
**📦 GitHub Repository:** [**https://github.com/venkat-nallapu/kuku-promo-finder-studio**](https://github.com/venkat-nallapu/kuku-promo-finder-studio)  
**Problem Selected:** Problem 1 — *Find the Promos (Ads)* across the 50-episode microdrama screenplay pack (*I Married a Broken Soldier. He was a Duke*, 100 scenes, 3,727 concurrent cues, 102.7 minutes total runtime).

---

## 📦 Submission Deliverables Map

| Deliverable Required by Kuku Brief (p. 4) | File Path / Live Link | Description |
|---|---|---|
| **1. Solution Doc (PDF, <= 2 pages)** | [`Kuku_Solution_Doc.pdf`](Kuku_Solution_Doc.pdf) ([source `.md`](docs/Kuku_Solution_Doc.md)) | Covers Problem Decomposition, AI Leverage & Model Swapping, Prototype Guide, Quality Judgement (*Where AI Got It Wrong*), and 100x Scale Unit Economics. |
| **2. Working Prototype (`app.py` + `cli.py`)** | [**Live App (`kuku-promo-finder-studio.streamlit.app`)**](https://kuku-promo-finder-studio.streamlit.app/) · [`app.py`](app.py) · [`cli.py`](cli.py) | 4-tab Streamlit Promo Studio UI + CLI batch runner with 1-click model swapping (`semantic_reranker`, `heuristic`, `gemini`, `openai`, `anthropic`, `cli`), live `.zip`/`.txt` upload, and interactive constraint sliders for the shortlist interview. |
| **3. Raw Outputs (`raw_outputs/`)** | [`raw_outputs/`](raw_outputs) | Unedited outputs across all 50 episodes: [`ranked_promos_top20.md`](raw_outputs/ranked_promos_top20.md) / [`.json`](raw_outputs/ranked_promos_top20.json), [`all_50_episodes_promos.csv`](raw_outputs/all_50_episodes_promos.csv) / [`.json`](raw_outputs/all_50_episodes_promos.json), [`where_ai_got_it_wrong.md`](raw_outputs/where_ai_got_it_wrong.md), [`quality_audit_failures.md`](raw_outputs/quality_audit_failures.md), [`editor_cut_sheet.csv`](raw_outputs/editor_cut_sheet.csv), [`ffmpeg_slice_commands.sh`](raw_outputs/ffmpeg_slice_commands.sh), and [`model_swap_diff.json`](raw_outputs/model_swap_diff.json). |
| **4. Automated Verification Suite** | [`tests/test_pipeline.py`](tests/test_pipeline.py) | 16 unit, E2E, regression, LLM eval, and Streamlit `AppTest` tests verifying 4-track parsing, scene-clock reset normalization, `20.0s-60.0s` bounds, dialogue-safe cuts, cross-episode `Episode N+1` payoffs, spoken-only lore filtering, NMS deduplication, multi-segment stitches, 3-layer LLM rescue, and Streamlit Cloud zero-zip hydration. |

---

## 🚀 Quickstart (`< 30 seconds`)

- **Option A — Try Immediately in Browser (Zero Install):**  
  Open [**https://kuku-promo-finder-studio.streamlit.app/**](https://kuku-promo-finder-studio.streamlit.app/)

- **Option B — Run Locally:**
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

# 5. Run the 16-test automated pytest verification suite
pytest -v
```

## 🔒 Confidentiality Guard
Per page 5 of the assignment brief (*"Please do not share this document or the datasets publicly"*), `.gitignore` excludes `*.zip` and raw extracted screenplay `.txt` files from public Git pushes while tracking `raw_outputs/` and a built-in synthetic demo fixture in `src/parser.py` so cloud deployments work out-of-the-box.
