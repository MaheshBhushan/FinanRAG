"""One Claude client, shared by every backend in the benchmark.

All three retrieval strategies generate with the same model at the same
settings. That is the whole point: if the flat, graph and RLM backends used
different generators, the accuracy column would be measuring model quality
rather than retrieval strategy, which is not what this benchmark claims.

Also the single place cost is accounted, so cost_usd means the same thing in
every row of the results table.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-5"

# USD per million tokens. Sonnet 5 list is $3/$15; an introductory rate of
# $2/$10 runs through 2026-08-31. The benchmark was run inside the intro
# window, so report both -- a reader reproducing it later pays list.
PRICING = {
    "claude-sonnet-5": {
        "input": 2.00, "output": 10.00,          # intro, through 2026-08-31
        "input_list": 3.00, "output_list": 15.00,
    },
}

# Cache writes cost ~1.25x the input rate, reads ~0.1x.
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10

# Same for every backend, so generation quality is held constant.
EFFORT = "medium"
MAX_TOKENS = 4096


def _load_env() -> None:
    """Read .env without adding a python-dotenv import to the call path."""
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()
_client = anthropic.Anthropic()


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.calls += other.calls

    def cost_usd(self, list_price: bool = False) -> float:
        p = PRICING[MODEL]
        inp = p["input_list"] if list_price else p["input"]
        out = p["output_list"] if list_price else p["output"]
        return (
            self.input_tokens * inp
            + self.cache_write_tokens * inp * CACHE_WRITE_MULT
            + self.cache_read_tokens * inp * CACHE_READ_MULT
            + self.output_tokens * out
        ) / 1e6


@dataclass
class LLMResult:
    text: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    stop_reason: str | None = None


def complete(system: str, user: str, *, cache_system: bool = True,
             max_tokens: int = MAX_TOKENS) -> LLMResult:
    """One Claude call, with usage and wall-clock recorded.

    cache_system caches the system prompt prefix. The RLM backend reuses a
    large fixed system prompt across many recursive calls, so this is a real
    cost difference between backends -- and one the table should show rather
    than hide.
    """
    system_block = [{"type": "text", "text": system}]
    if cache_system:
        system_block[0]["cache_control"] = {"type": "ephemeral"}

    t0 = time.perf_counter()
    # Streaming so a long answer can't trip the HTTP timeout.
    with _client.messages.stream(
        model=MODEL,
        max_tokens=max_tokens,
        system=system_block,
        thinking={"type": "adaptive"},
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": user}],
    ) as stream:
        msg = stream.get_final_message()
    latency_ms = (time.perf_counter() - t0) * 1000

    text = "".join(b.text for b in msg.content if b.type == "text")
    u = msg.usage
    return LLMResult(
        text=text,
        usage=Usage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            calls=1,
        ),
        latency_ms=latency_ms,
        stop_reason=msg.stop_reason,
    )
