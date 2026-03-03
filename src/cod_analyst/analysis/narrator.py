"""LLM narration layer — translates ML insights into human-readable text.

Wraps OpenAI and Gemini APIs with structured prompts for:
- Scouting report narration
- Strategy explanation
- Loss analysis narrative
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from cod_analyst.analysis.profiler import ScoutingReport
from cod_analyst.config import AppConfig
from cod_analyst.game.models import StrategySuggestion

logger = logging.getLogger(__name__)


@dataclass
class NarrationResult:
    """Result of LLM narration."""
    text: str
    model_used: str
    tokens_used: int = 0


class LLMNarrator:
    """Translates structured analysis into natural language.

    Supports OpenAI (GPT-4o) and Google Gemini (2.0 Flash).
    Uses map grid aliases for callouts in output.
    """

    def __init__(self, cfg: AppConfig):
        self._cfg = cfg
        self._provider = cfg.analysis.llm_provider

    def _call_openai(self, system_prompt: str, user_prompt: str) -> NarrationResult:
        """Call OpenAI API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package not installed")

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=self._cfg.analysis.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self._cfg.analysis.max_tokens,
            temperature=self._cfg.analysis.temperature,
        )

        text = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0

        return NarrationResult(
            text=text,
            model_used=self._cfg.analysis.openai_model,
            tokens_used=tokens,
        )

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> NarrationResult:
        """Call Google Gemini API."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise RuntimeError("google-genai package not installed")

        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(
            self._cfg.analysis.gemini_model,
            system_instruction=system_prompt,
        )

        response = model.generate_content(user_prompt)
        text = response.text if response.text else ""

        return NarrationResult(
            text=text,
            model_used=self._cfg.analysis.gemini_model,
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> NarrationResult:
        """Route to the configured LLM provider."""
        if self._provider == "openai":
            return self._call_openai(system_prompt, user_prompt)
        elif self._provider == "gemini":
            return self._call_gemini(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self._provider}")

    def narrate_scouting_report(self, report: ScoutingReport) -> NarrationResult:
        """Convert a scouting report into a polished narrative.

        Returns a professional-quality scouting brief suitable for
        team briefings.
        """
        system_prompt = (
            "You are a professional Call of Duty League analyst preparing a scouting "
            "report for your team's coaching staff. Write in a clear, concise, "
            "actionable style. Use specific map callouts and reference data when available. "
            "Focus on what the opponent does, their tendencies, and how to counter them. "
            "Keep it under 400 words."
        )

        user_prompt = f"""
Convert this structural scouting report into a polished narrative:

{report.to_text()}

Weapon preferences: {report.weapon_preferences}

Format as a professional briefing with clear sections for:
1. Overview (one-liner)
2. Key Tendencies (routes, setups, pace)
3. Trading & Engagement Style
4. Recommended Counters
5. Watch Out For
"""
        return self._call_llm(system_prompt, user_prompt)

    def narrate_strategy(self, suggestions: list[StrategySuggestion]) -> NarrationResult:
        """Convert strategy suggestions into a coherent game plan narrative."""
        system_prompt = (
            "You are a Call of Duty League analyst writing a pre-round strategy brief. "
            "Synthesize the following data-driven suggestions into a coherent 2-3 paragraph "
            "game plan. Be specific about positioning and timing."
        )

        suggestions_text = "\n".join(
            f"- [{s.confidence:.0%}] {s.content}" for s in suggestions
        )

        user_prompt = f"Strategy suggestions:\n{suggestions_text}\n\nWrite a coherent game plan."

        return self._call_llm(system_prompt, user_prompt)

    def narrate_loss_analysis(self, analysis: StrategySuggestion) -> NarrationResult:
        """Convert loss analysis into an actionable post-mortem."""
        system_prompt = (
            "You are a CDL analyst reviewing a lost round. Write a brief, constructive "
            "post-mortem that identifies what went wrong and suggests specific adjustments. "
            "Keep it under 200 words and focus on actionable fixes, not blame."
        )

        user_prompt = f"Round {analysis.source_round_id} loss analysis:\n{analysis.content}"

        return self._call_llm(system_prompt, user_prompt)
