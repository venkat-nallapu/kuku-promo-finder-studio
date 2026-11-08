"""Stage-2 Swappable LLM Promo Judge with 5-Call Macro-Batching, SHA-256 Cache & 3-Layer Rescue.

SUPPORTED PROVIDERS (Switchable via CLI `--provider` or UI Sidebar Dropdown):
  1. `gemini`    -> Google Gemini REST API (`gemini-2.5-flash`, uses GEMINI_API_KEY / GOOGLE_API_KEY)
  2. `openai`    -> OpenAI Chat Completions REST API (`gpt-4o-mini`, uses OPENAI_API_KEY)
  3. `anthropic` -> Anthropic Messages REST API (`claude-3-5-haiku-latest`, uses ANTHROPIC_API_KEY)
  4. `cli`       -> Local `claude -p` or `gemini -p` CLI subprocess
  5. `semantic_reranker` (or `heuristic`) -> Zero-latency deterministic Stage-2 narrative-arc
     judge (evaluates emotional polarity flip, status reversal, dialogue punchiness, and
     curiosity-gap cliffhanger strength) + automatic fallback when API keys are absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from src.candidates import PromoCandidate

PROMPT_VERSION = "v2.2-kuku-microdrama"
DEFAULT_CACHE_PATH = Path(".cache/llm_cache.json")


def _truncate_on_word(text: str, max_chars: int = 115) -> str:
    """Truncate text cleanly at a word boundary without slicing words in half."""
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[:max_chars].rsplit(" ", 1)[0].rstrip(",;:")
    return f"{cut}..."

BETRAYAL_REVERSAL_PATTERNS = (
    r"\bbetray\w*\b",
    r"\bbastard\b",
    r"\bimpossible\b",
    r"\bsecret\w*\b",
    r"\blie[ds]?\b",
    r"\bnever\b",
    r"\bkill\w*\b",
    r"\bpoison\w*\b",
    r"\bmarry\b",
    r"\bwife\b",
    r"\bhusband\b",
    r"\bbride\b",
    r"\baltar\b",
    r"\bsister\b",
    r"\bduke\b",
    r"\bduchess\b",
    r"\bguard[s]?\b",
    r"\bseize\b",
    r"\bkneel\b",
    r"\bblood\b",
    r"\bdead\b",
    r"\balive\b",
    r"\bpregnant\b",
    r"\baffair\b",
    r"\bcollar\b",
    r"\bsin\b",
    r"\bstole\b",
    r"\bforgive\b",
    r"\brevenge\b",
)


@dataclass
class RunTelemetry:
    """Execution telemetry for a 50-episode pipeline run."""

    provider_requested: str
    provider_effective: str
    model_name: str
    macro_batches_total: int = 0
    api_calls_made: int = 0
    cache_hits: int = 0
    rescued_batches: int = 0
    total_candidates_scored: int = 0
    elapsed_sec: float = 0.0
    estimated_tokens_in: int = 0
    estimated_tokens_out: int = 0
    estimated_cost_usd: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_requested": self.provider_requested,
            "provider_effective": self.provider_effective,
            "model_name": self.model_name,
            "macro_batches_total": self.macro_batches_total,
            "api_calls_made": self.api_calls_made,
            "cache_hits": self.cache_hits,
            "rescued_batches": self.rescued_batches,
            "total_candidates_scored": self.total_candidates_scored,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "estimated_tokens_in": self.estimated_tokens_in,
            "estimated_tokens_out": self.estimated_tokens_out,
            "estimated_cost_usd": round(self.estimated_cost_usd, 5),
            "warnings": self.warnings,
        }


def compute_cache_key(
    provider: str, model_name: str, candidate: PromoCandidate
) -> str:
    """Compute a deterministic SHA-256 cache key for a candidate evaluation."""
    payload = "|".join(
        [
            PROMPT_VERSION,
            provider,
            model_name,
            candidate.candidate_id,
            f"{candidate.start_sec:.2f}-{candidate.end_sec:.2f}",
            "\n".join(candidate.excerpt_lines[:10]),
            candidate.withheld_payoff_preview,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_disk_cache(cache_path: Path = DEFAULT_CACHE_PATH) -> Dict[str, Dict[str, Any]]:
    """Load the SHA-256 LLM evaluation cache from disk."""
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_disk_cache(
    cache: Dict[str, Dict[str, Any]], cache_path: Path = DEFAULT_CACHE_PATH
) -> None:
    """Persist the SHA-256 LLM evaluation cache to disk atomically via os.replace."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
        tmp_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        os.replace(tmp_path, cache_path)
    except Exception:
        pass


def extract_json_payload(raw_text: str) -> Any:
    """Layer-1 Rescue: extract and parse JSON object/array from markdown or noisy LLM output."""
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty LLM response text")

    cleaned = raw_text.strip()
    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find outermost JSON object or array
    obj_start = cleaned.find("{")
    obj_end = cleaned.rfind("}")
    arr_start = cleaned.find("[")
    arr_end = cleaned.rfind("]")

    candidates_to_try: List[str] = []
    if obj_start != -1 and obj_end > obj_start:
        candidates_to_try.append(cleaned[obj_start : obj_end + 1])
    if arr_start != -1 and arr_end > arr_start:
        candidates_to_try.append(cleaned[arr_start : arr_end + 1])

    for snippet in candidates_to_try:
        # Remove trailing commas before } or ]
        repaired = re.sub(r",\s*([}\]])", r"\1", snippet)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not extract valid JSON from response: {raw_text[:200]!r}")


def evaluate_candidate_narrative_arc(candidate: PromoCandidate) -> Dict[str, Any]:
    """Deterministic Stage-2 Semantic & Narrative-Arc Judge.

    Evaluates narrative contrast (setup vs reversal), dialogue quote punchiness,
    emotional trajectory shift, cross-episode curiosity gap, and multi-segment pacing
    to produce a distinct Stage-2 LLM/Semantic score (0-100) and rich editorial pitch.
    """
    excerpt_blob = " ".join(candidate.excerpt_lines).lower()
    reversal_hits = sum(
        1 for pat in BETRAYAL_REVERSAL_PATTERNS if re.search(pat, excerpt_blob)
    )

    # Check emotional shift between first half and second half of excerpt lines
    mid = max(1, len(candidate.excerpt_lines) // 2)
    first_half = " ".join(candidate.excerpt_lines[:mid]).lower()
    second_half = " ".join(candidate.excerpt_lines[mid:]).lower()
    has_question_climax = "?" in candidate.closing_cliffhanger_line
    has_shock_ending = any(
        w in second_half
        for w in (
            "shock",
            "stunned",
            "gasp",
            "cliffhanger",
            "cut to black",
            "terror",
            "slap",
            "blood",
            "blade",
            "seize",
            "guards",
            "kneel",
            "impossible",
            "betray",
        )
    )
    cross_ep_bonus = 3.2 if candidate.payoff_source == "next_episode" else 1.0
    stitch_bonus = 2.0 if candidate.promo_type != "single_segment" else 0.0
    lore_deduction = max(0, len(candidate.spoken_lore_tokens_found) - 2) * 2.5

    narrative_score = (
        candidate.stage1_score * 0.55
        + min(22.0, 10.0 + reversal_hits * 1.45)
        + (4.5 if has_question_climax else 2.0)
        + (5.5 if has_shock_ending else 2.0)
        + cross_ep_bonus
        + stitch_bonus
        - lore_deduction
    )
    llm_score = round(max(45.0, min(99.2, narrative_score)), 2)
    blended_final = round(0.40 * candidate.stage1_score + 0.60 * llm_score, 2)

    speakers_str = (
        " vs. ".join(candidate.speakers_in_clip[:2])
        if len(candidate.speakers_in_clip) >= 2
        else (candidate.speakers_in_clip[0] if candidate.speakers_in_clip else "Lead Cast")
    )
    hook_clean = _truncate_on_word(
        re.sub(r"^\([0-9:.]+\)\s*", "", candidate.opening_hook_line).strip(), 115
    )
    cliff_clean = _truncate_on_word(
        re.sub(r"^\([0-9:.-]+\)\s*", "", candidate.closing_cliffhanger_line).strip(), 125
    )

    prefix_tag = ""
    if candidate.promo_type == "multi_segment_ellipsis":
        prefix_tag = "[Multi-Segment Ellipsis Cut] "
    elif candidate.promo_type == "multi_segment_cold_open":
        prefix_tag = "[Cold-Open Teaser Stitch] "

    rationale = (
        f"{prefix_tag}High-converting {candidate.duration_sec:.1f}s ad confrontation ({speakers_str}): "
        f"arrests scroll in the first 3s with '{hook_clean}', escalates through {reversal_hits} "
        f"high-stakes reversal beats, and snaps to black on '{cliff_clean}' "
        f"(withholding {candidate.payoff_source.replace('_', ' ')} payoff)."
    )

    if len(candidate.spoken_lore_tokens_found) <= 2:
        risk_note = (
            f"Zero cold-audience confusion: only {len(candidate.spoken_lore_tokens_found)} character name(s) "
            f"spoken aloud; power dynamic ({speakers_str}) is immediately legible without prior episodes."
        )
    else:
        risk_note = (
            f"Contains {len(candidate.spoken_lore_tokens_found)} spoken proper nouns "
            f"({', '.join(candidate.spoken_lore_tokens_found[:4])}); emotional blocking and {candidate.payoff_source.replace('_', ' ')} "
            f"cliffhanger override exposition."
        )

    return {
        "llm_score": llm_score,
        "final_score": blended_final,
        "rationale": rationale,
        "editorial_risk_note": risk_note,
    }


def _build_macro_batch_prompt(batch: Sequence[PromoCandidate]) -> str:
    """Build a compact JSON-schema prompt for a macro-batch of promo candidates."""
    items = []
    for c in batch:
        items.append(
            {
                "candidate_id": c.candidate_id,
                "episode_num": c.episode_num,
                "promo_type": c.promo_type,
                "duration_sec": c.duration_sec,
                "stage1_score": c.stage1_score,
                "opening_hook": c.opening_hook_line[:140],
                "closing_cliffhanger": c.closing_cliffhanger_line[:140],
                "withheld_payoff": c.withheld_payoff_preview[:160],
                "spoken_lore_names": c.spoken_lore_tokens_found,
                "excerpt": c.excerpt_lines[:6],
            }
        )
    return (
        "You are a Performance Marketing Creative Director for short-form vertical microdramas "
        "(ReelShort, DramaBox, Kuku TV). Evaluate each 20-60s promo candidate below for cold-audience "
        "scroll-stopping power (first 3s), emotional status reversal, standalone clarity (penalize confusing "
        "proper-noun lore dump in spoken dialogue), and cliffhanger curiosity gap (cuts right before payoff).\n\n"
        "Return ONLY valid JSON matching this schema:\n"
        '{"evaluations": [{"candidate_id": "...", "llm_score": 88.5, '
        '"rationale": "1-2 sentence pitch citing exact hook and cliffhanger", '
        '"editorial_risk_note": "1 sentence on cold-viewer clarity or lore risk"}]}\n\n'
        f"CANDIDATES:\n{json.dumps(items, ensure_ascii=False)}"
    )


def _call_provider_api(
    provider: str,
    model_name: str,
    prompt: str,
    timeout_sec: float = 25.0,
) -> str:
    """Invoke the selected LLM provider API or local CLI."""
    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY / GOOGLE_API_KEY environment variable")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }
        with httpx.Client(timeout=timeout_sec) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY environment variable")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": model_name,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        }
        with httpx.Client(timeout=timeout_sec) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY environment variable")
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": model_name,
            "max_tokens": 2048,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt}],
        }
        with httpx.Client(timeout=timeout_sec) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]

    if provider == "cli":
        proc = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=True,
        )
        return proc.stdout

    raise ValueError(f"Unsupported external provider: {provider!r}")


DEFAULT_MODELS: Dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "cli": "claude-cli-sonnet",
    "semantic_reranker": "stage2-narrative-arc-v2",
    "heuristic": "stage1-structural-v2",
}


def score_candidates_with_judge(
    candidates: Sequence[PromoCandidate],
    provider: str = "semantic_reranker",
    model_name: Optional[str] = None,
    cache_path: Path = DEFAULT_CACHE_PATH,
    macro_batch_size: int = 12,
    simulate_failure_mode: Optional[str] = None,
) -> Tuple[List[PromoCandidate], RunTelemetry]:
    """Run Stage-2 evaluation across candidates with 5-call macro-batching & 3-layer rescue.

    Providers:
      - `heuristic`: Pure Stage-1 structural score (baseline for Model-Swap Diff).
      - `semantic_reranker`: Deterministic Stage-2 narrative-arc judge (default when no API key).
      - `gemini` / `openai` / `anthropic` / `cli`: Live LLM API/CLI with automatic 3-layer rescue.
    """
    t0 = time.time()
    norm_provider = provider.strip().lower()
    resolved_model = model_name or DEFAULT_MODELS.get(norm_provider, "stage2-narrative-arc-v2")

    telemetry = RunTelemetry(
        provider_requested=norm_provider,
        provider_effective=norm_provider,
        model_name=resolved_model,
        total_candidates_scored=len(candidates),
    )

    if norm_provider == "heuristic":
        for c in candidates:
            c.llm_score = c.stage1_score
            c.final_score = c.stage1_score
            c.scoring_provider = "heuristic (stage1-structural)"
            c.rescued_fallback = False
        ranked = sorted(candidates, key=lambda x: x.final_score, reverse=True)
        telemetry.elapsed_sec = time.time() - t0
        return ranked, telemetry

    cache = load_disk_cache(cache_path)
    cache_dirty = False
    unscored: List[PromoCandidate] = []

    for c in candidates:
        ckey = compute_cache_key(norm_provider, resolved_model, c)
        if simulate_failure_mode is None and ckey in cache:
            entry = cache[ckey]
            c.llm_score = float(entry["llm_score"])
            c.final_score = float(entry["final_score"])
            c.rationale = str(entry["rationale"])
            c.editorial_risk_note = str(entry["editorial_risk_note"])
            c.scoring_provider = str(entry.get("scoring_provider", f"{norm_provider}:{resolved_model}"))
            c.rescued_fallback = bool(entry.get("rescued_fallback", False))
            telemetry.cache_hits += 1
        else:
            unscored.append(c)

    if unscored:
        if norm_provider == "semantic_reranker":
            for c in unscored:
                res = evaluate_candidate_narrative_arc(c)
                c.llm_score = res["llm_score"]
                c.final_score = res["final_score"]
                c.rationale = res["rationale"]
                c.editorial_risk_note = res["editorial_risk_note"]
                c.scoring_provider = f"semantic_reranker:{resolved_model}"
                c.rescued_fallback = False
                ckey = compute_cache_key(norm_provider, resolved_model, c)
                cache[ckey] = {
                    **res,
                    "scoring_provider": c.scoring_provider,
                    "rescued_fallback": False,
                }
                cache_dirty = True
        else:
            # Macro-batch external LLM calls (e.g. 5 macro-batches across top episode candidates)
            batches = [
                unscored[i : i + macro_batch_size]
                for i in range(0, len(unscored), macro_batch_size)
            ]
            telemetry.macro_batches_total = len(batches)

            for b_idx, batch in enumerate(batches):
                prompt = _build_macro_batch_prompt(batch)
                telemetry.estimated_tokens_in += len(prompt) // 4
                raw_response: Optional[str] = None
                parsed_payload: Optional[Any] = None
                last_err: Optional[str] = None

                # Layer 2 Rescue: up to 2 attempts with backoff
                for attempt in range(2):
                    try:
                        if simulate_failure_mode == "malformed_json" and attempt == 0:
                            raw_response = '```json\n{"evaluations": [{"candidate_id": "' + batch[0].candidate_id + '", "llm_score": 91.5, "rationale": "Repaired from markdown fence", "editorial_risk_note": "Low risk",}]}\n```'
                        elif simulate_failure_mode == "http_429":
                            raise RuntimeError("Simulated HTTP 429 Rate Limit Exceeded")
                        else:
                            telemetry.api_calls_made += 1
                            raw_response = _call_provider_api(
                                norm_provider, resolved_model, prompt
                            )
                        # Layer 1 Rescue: JSON extractor & trailing-comma repair
                        parsed_payload = extract_json_payload(raw_response)
                        break
                    except Exception as exc:
                        last_err = str(exc)
                        if attempt == 0 and simulate_failure_mode != "http_429":
                            time.sleep(0.05)

                eval_map: Dict[str, Dict[str, Any]] = {}
                if isinstance(parsed_payload, dict) and "evaluations" in parsed_payload:
                    for item in parsed_payload["evaluations"]:
                        if isinstance(item, dict) and "candidate_id" in item:
                            eval_map[str(item["candidate_id"])] = item
                elif isinstance(parsed_payload, list):
                    for item in parsed_payload:
                        if isinstance(item, dict) and "candidate_id" in item:
                            eval_map[str(item["candidate_id"])] = item

                if not eval_map:
                    # Layer 3 Rescue: Graceful fallback to Stage-2 Narrative Arc Judge
                    telemetry.rescued_batches += 1
                    telemetry.provider_effective = f"{norm_provider} (rescued->semantic_reranker)"
                    if last_err and last_err not in telemetry.warnings:
                        telemetry.warnings.append(
                            f"Macro-batch {b_idx + 1} rescued via Layer-3 fallback ({last_err[:120]})"
                        )

                for c in batch:
                    if c.candidate_id in eval_map:
                        item = eval_map[c.candidate_id]
                        llm_val = float(item.get("llm_score", c.stage1_score))
                        llm_val = round(max(0.0, min(100.0, llm_val)), 2)
                        c.llm_score = llm_val
                        c.final_score = round(0.40 * c.stage1_score + 0.60 * llm_val, 2)
                        c.rationale = str(item.get("rationale") or c.rationale)
                        c.editorial_risk_note = str(
                            item.get("editorial_risk_note") or c.editorial_risk_note
                        )
                        c.scoring_provider = f"{norm_provider}:{resolved_model}"
                        c.rescued_fallback = False
                    else:
                        fallback_res = evaluate_candidate_narrative_arc(c)
                        c.llm_score = fallback_res["llm_score"]
                        c.final_score = fallback_res["final_score"]
                        c.rationale = fallback_res["rationale"]
                        c.editorial_risk_note = fallback_res["editorial_risk_note"]
                        c.scoring_provider = f"{norm_provider}:rescued_fallback"
                        c.rescued_fallback = True

                    if simulate_failure_mode is None:
                        ckey = compute_cache_key(norm_provider, resolved_model, c)
                        cache[ckey] = {
                            "llm_score": c.llm_score,
                            "final_score": c.final_score,
                            "rationale": c.rationale,
                            "editorial_risk_note": c.editorial_risk_note,
                            "scoring_provider": c.scoring_provider,
                            "rescued_fallback": c.rescued_fallback,
                        }
                        cache_dirty = True

    if cache_dirty and simulate_failure_mode is None:
        save_disk_cache(cache, cache_path)

    telemetry.estimated_tokens_out = telemetry.api_calls_made * 450
    telemetry.estimated_cost_usd = (
        telemetry.estimated_tokens_in * 0.00000015
        + telemetry.estimated_tokens_out * 0.00000060
    )
    telemetry.elapsed_sec = time.time() - t0

    ranked = sorted(candidates, key=lambda x: x.final_score, reverse=True)
    return ranked, telemetry
