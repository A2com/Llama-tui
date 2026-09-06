import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import httpx


@dataclass
class BenchResult:
    predicted_per_second: float
    prompt_per_second: float
    predicted_n: int
    prompt_n: int
    draft_n: Optional[int] = None
    draft_n_accepted: Optional[int] = None

    @property
    def draft_acceptance_rate(self) -> Optional[float]:
        if not self.draft_n:
            return None
        return self.draft_n_accepted / self.draft_n


def parse_timings(response_json: dict) -> BenchResult:
    t = response_json["timings"]
    return BenchResult(
        predicted_per_second=t["predicted_per_second"],
        prompt_per_second=t["prompt_per_second"],
        predicted_n=t["predicted_n"],
        prompt_n=t["prompt_n"],
        draft_n=t.get("draft_n"),
        draft_n_accepted=t.get("draft_n_accepted"),
    )


def run_bench(prompts: List[str], port: int = 8082, max_tokens: int = 200,
              n_runs: int = 3, temperature: float = 0.0) -> dict:
    results = {}
    for prompt in prompts:
        runs = []
        for _ in range(n_runs):
            resp = httpx.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={
                    "model": "bench",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            runs.append(parse_timings(resp.json()))
        results[prompt] = {
            "predicted_per_second_median": statistics.median(r.predicted_per_second for r in runs),
            "prompt_per_second_median": statistics.median(r.prompt_per_second for r in runs),
            "runs": [asdict(r) for r in runs],
        }
    return results


def write_results(results: dict, path: Path) -> None:
    path.write_text(json.dumps(results, indent=2))
