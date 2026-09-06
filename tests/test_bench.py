import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.bench import BenchResult, parse_timings, run_bench, write_results


def _mock_completion(predicted_per_second=20.0, prompt_per_second=55.0,
                      predicted_n=47, prompt_n=68, draft_n=None, draft_n_accepted=None):
    timings = {
        "prompt_n": prompt_n,
        "prompt_per_second": prompt_per_second,
        "predicted_n": predicted_n,
        "predicted_per_second": predicted_per_second,
    }
    if draft_n is not None:
        timings["draft_n"] = draft_n
        timings["draft_n_accepted"] = draft_n_accepted
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = lambda: None
    resp.json.return_value = {
        "choices": [{"message": {"content": "réponse"}}],
        "timings": timings,
    }
    return resp


def test_parse_timings_extracts_core_fields():
    body = {"timings": {"predicted_per_second": 20.35, "prompt_per_second": 54.93,
                         "predicted_n": 47, "prompt_n": 68}}
    result = parse_timings(body)
    assert result.predicted_per_second == 20.35
    assert result.prompt_per_second == 54.93
    assert result.predicted_n == 47
    assert result.prompt_n == 68
    assert result.draft_n is None
    assert result.draft_n_accepted is None


def test_parse_timings_extracts_draft_acceptance_when_present():
    body = {"timings": {"predicted_per_second": 20.0, "prompt_per_second": 50.0,
                         "predicted_n": 47, "prompt_n": 68,
                         "draft_n": 36, "draft_n_accepted": 28}}
    result = parse_timings(body)
    assert result.draft_n == 36
    assert result.draft_n_accepted == 28
    assert result.draft_acceptance_rate == pytest.approx(28 / 36)


def test_run_bench_posts_stream_false_and_returns_per_prompt_medians():
    with patch("httpx.post", return_value=_mock_completion(predicted_per_second=20.0)) as mock_post:
        results = run_bench(["prompt A"], port=8082, n_runs=1)
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["stream"] is False
    assert call_kwargs["json"]["messages"][0]["content"] == "prompt A"
    assert results["prompt A"]["predicted_per_second_median"] == 20.0


def test_run_bench_takes_median_across_runs():
    responses = [
        _mock_completion(predicted_per_second=18.0),
        _mock_completion(predicted_per_second=20.0),
        _mock_completion(predicted_per_second=22.0),
    ]
    with patch("httpx.post", side_effect=responses):
        results = run_bench(["prompt A"], port=8082, n_runs=3)
    assert results["prompt A"]["predicted_per_second_median"] == 20.0
    assert len(results["prompt A"]["runs"]) == 3


def test_write_results_writes_json(tmp_path):
    out = tmp_path / "bench.json"
    write_results({"a": 1}, out)
    assert json.loads(out.read_text()) == {"a": 1}
