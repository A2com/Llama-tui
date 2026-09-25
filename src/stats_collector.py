import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class TokenStats:
    is_generating: bool = False
    gen_tps: Optional[float] = None
    avg_tps: Optional[float] = None
    peak_tps: Optional[float] = None
    prompt_tokens: int = 0
    prompt_cached: int = 0
    n_ctx: int = 0
    total_generated: int = 0

    @property
    def cache_hit_ratio(self) -> Optional[float]:
        processed = self.prompt_tokens + self.prompt_cached
        if processed <= 0:
            return None
        return self.prompt_cached / processed

    @property
    def ctx_used_pct(self) -> Optional[float]:
        if self.n_ctx <= 0:
            return None
        return (self.prompt_tokens / self.n_ctx) * 100.0


class StatsCollector:
    MAX_HISTORY = 60

    def __init__(self, port: int):
        self._port = port
        self._last_n_decoded: Optional[int] = None
        self._last_time: Optional[float] = None
        self._total_generated: int = 0
        self.last_stats = TokenStats()
        self.history: deque[float] = deque(maxlen=self.MAX_HISTORY)
        self.cache_history: deque[float] = deque(maxlen=self.MAX_HISTORY)
        self._last_was_generating: bool = False

    def poll(self) -> TokenStats:
        try:
            resp = httpx.get(f"http://localhost:{self._port}/slots", timeout=1.0)
            slot = resp.json()[0]
        except Exception:
            self._reset()
            self.last_stats = TokenStats(total_generated=self._total_generated)
            return self.last_stats

        is_processing = slot.get("is_processing", False)
        next_token = slot.get("next_token", [{}])
        n_decoded = next_token[0].get("n_decoded", 0) if next_token else 0
        n_ctx = slot.get("n_ctx", 0)
        n_prompt = slot.get("n_prompt_tokens", 0)
        n_cached = slot.get("n_prompt_tokens_cache", 0)
        processed = n_prompt + n_cached
        cache_pct = (n_cached / processed * 100.0) if processed > 0 else 0.0
        now = time.monotonic()

        # Archivage cache : uniquement pendant la génération, ou un seul point
        # final à la transition génér→idle (jamais pendant l'idle prolongé).
        was_generating = self._last_was_generating
        if is_processing:
            self.cache_history.append(cache_pct)
        elif was_generating:
            self.cache_history.append(cache_pct)
        self._last_was_generating = is_processing

        avg_tps = sum(self.history) / len(self.history) if self.history else None
        peak_tps = max(self.history) if self.history else None

        if not is_processing:
            if self._last_n_decoded:
                self._total_generated += self._last_n_decoded
            self._reset()
            self.last_stats = TokenStats(
                is_generating=False,
                avg_tps=avg_tps,
                peak_tps=peak_tps,
                prompt_tokens=n_prompt,
                prompt_cached=n_cached,
                n_ctx=n_ctx,
                total_generated=self._total_generated,
            )
            return self.last_stats

        tps = self._compute_tps(n_decoded, now)
        if tps is not None:
            self.history.append(tps)
        avg_tps = sum(self.history) / len(self.history) if self.history else None
        peak_tps = max(self.history) if self.history else None

        self.last_stats = TokenStats(
            is_generating=True,
            gen_tps=tps,
            avg_tps=avg_tps,
            peak_tps=peak_tps,
            prompt_tokens=n_prompt,
            prompt_cached=n_cached,
            n_ctx=n_ctx,
            total_generated=self._total_generated + n_decoded,
        )
        return self.last_stats

    def _compute_tps(self, n_decoded: int, now: float) -> Optional[float]:
        """Calcule t/s depuis le delta n_decoded. Retourne None si reset ou premier poll."""
        if self._last_n_decoded is None:
            self._last_n_decoded = n_decoded
            self._last_time = now
            return None

        if n_decoded < self._last_n_decoded:
            # Nouvelle génération — archive le total puis reset
            self._total_generated += self._last_n_decoded
            self._last_n_decoded = n_decoded
            self._last_time = now
            return None

        delta_tokens = n_decoded - self._last_n_decoded
        delta_time = now - self._last_time

        self._last_n_decoded = n_decoded
        self._last_time = now

        if delta_time <= 0 or delta_tokens <= 0:
            return None

        return delta_tokens / delta_time

    def _reset(self) -> None:
        self._last_n_decoded = None
        self._last_time = None