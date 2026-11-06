"""
adhd_module.py - Task chunking logic for PUGU's ADHD tools.

chunk_task() tries the Groq API (free tier) first and falls back to a generic
template when the API key is missing, the free limit is reached, or the call
fails. This module has no Streamlit dependency.
"""

from __future__ import annotations

import json
import logging
import re

from groq import Groq

logger = logging.getLogger(__name__)

MIN_STEPS = 4
MAX_STEPS = 6

_SYSTEM_PROMPT = (
    "You are a calm, encouraging executive-function coach for people with ADHD. "
    "Break the user's task into between 4 and 6 concrete micro-steps. "
    "Rules: each step is one clear action that starts with a verb; "
    "each step should take roughly the number of minutes the user gives; "
    "the first step must be tiny and easy so that starting feels effortless; "
    "keep every step under 20 words; no numbering and no emojis. "
    "Respond with ONLY a JSON array of strings and nothing else."
)


def _clean_step(step: str) -> str:
    """Remove any leading numbering or bullet characters from a step."""
    return re.sub(r"^\s*(?:\d+[.)]|[-*\u2022])\s*", "", step).strip()


def _parse_steps(raw: str) -> list[str]:
    """Extract and validate a JSON array of steps from the model's reply."""
    match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in the model response.")

    data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of steps.")

    steps = [_clean_step(str(item)) for item in data]
    steps = [step for step in steps if step]
    if len(steps) < MIN_STEPS:
        raise ValueError(f"Model returned only {len(steps)} usable steps.")
    return steps[:MAX_STEPS]


def chunk_task_ai(task: str, chunk_minutes: int, api_key: str, model: str) -> list[str]:
    """Ask a Groq-hosted model to split `task` into 4-6 micro-steps. Raises on failure."""
    client = Groq(api_key=api_key, timeout=30.0, max_retries=1)
    completion = client.chat.completions.create(
        model=model,
        max_tokens=600,
        temperature=0.3,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Task: {task}\nMinutes available per step: {chunk_minutes}",
            },
        ],
    )
    raw = completion.choices[0].message.content or ""
    return _parse_steps(raw)


def chunk_task_fallback(task: str, chunk_minutes: int) -> list[str]:
    """Return generic, always-available steps when the AI cannot be used."""
    label = task if len(task) <= 60 else task[:57].rstrip() + "..."
    return [
        f"Get ready: gather everything you need for \"{label}\" and clear your workspace.",
        "Write one sentence that describes what \"finished\" looks like.",
        f"Start tiny: work on the very first small piece for {chunk_minutes} minutes.",
        f"Keep going: do the next small piece for {chunk_minutes} minutes.",
        "Wrap up: quickly check your work and write down anything still left to do.",
    ]


def chunk_task(
    task: str, chunk_minutes: int, api_key: str | None, model: str
) -> tuple[list[str], bool]:
    """
    Chunk a task into micro-steps.

    Returns (steps, used_ai). `used_ai` is False when the fallback was used.
    """
    if api_key:
        try:
            return chunk_task_ai(task, chunk_minutes, api_key, model), True
        except Exception:  # rate limit, network, or parse failure: degrade gracefully
            logger.warning("AI task chunking failed; using fallback.", exc_info=True)
    return chunk_task_fallback(task, chunk_minutes), False