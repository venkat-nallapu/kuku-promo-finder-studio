# CLAUDE.md — Kuku Promo Finder Studio

## Project Commands
- **Run 50-Episode CLI Batch Pipeline:** `.venv/bin/python cli.py --provider semantic_reranker --output-dir raw_outputs`
- **Launch Interactive Streamlit Studio UI:** `.venv/bin/streamlit run app.py`
- **Run Automated Test Suite (15 tests):** `.venv/bin/pytest -v`
- **Regenerate 2-Page Solution Doc PDF:** `~/.claude/skills/gstack/make-pdf/dist/pdf generate --margins 0.4in --no-confidential docs/Kuku_Solution_Doc.md Kuku_Solution_Doc.pdf`

## Testing
- **Test Runner:** `.venv/bin/pytest -v`
- **Test Directory:** `tests/test_pipeline.py` (15 unit, integration, regression, and Streamlit `AppTest` tests)
- **Minimum / Target Coverage:** Minimum: 80%, Target: 100%

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
- Author a backlog-ready spec/issue → invoke /spec
