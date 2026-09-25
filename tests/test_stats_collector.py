import time
import pytest
from unittest.mock import patch, MagicMock
from src.stats_collector import StatsCollector, TokenStats


@pytest.fixture
def collector():
    return StatsCollector(port=11434)


def _mock_slot(is_processing: bool, n_decoded: int, n_ctx: int = 0,
               n_prompt: int = 0, n_cached: int = 0) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = [{
        "is_processing": is_processing,
        "next_token": [{"n_decoded": n_decoded}],
        "n_ctx": n_ctx,
        "n_prompt_tokens": n_prompt,
        "n_prompt_tokens_cache": n_cached,
    }]
    return resp


def test_initial_state_is_idle(collector):
    stats = collector.last_stats
    assert stats.is_generating is False
    assert stats.gen_tps is None


def test_poll_returns_idle_when_not_processing(collector):
    with patch("httpx.get", return_value=_mock_slot(False, 0)):
        stats = collector.poll()
    assert stats.is_generating is False
    assert stats.gen_tps is None


def test_poll_returns_generating_when_processing(collector):
    with patch("httpx.get", return_value=_mock_slot(True, 10)):
        stats = collector.poll()
    assert stats.is_generating is True


def test_tps_computed_from_n_decoded_delta(collector):
    t0 = time.monotonic()
    with patch("httpx.get", return_value=_mock_slot(True, 50)), \
         patch("time.monotonic", return_value=t0):
        collector.poll()

    with patch("httpx.get", return_value=_mock_slot(True, 70)), \
         patch("time.monotonic", return_value=t0 + 1.0):
        stats = collector.poll()

    assert stats.gen_tps == pytest.approx(20.0, rel=0.05)


def test_tps_resets_when_n_decoded_resets(collector):
    t0 = time.monotonic()
    with patch("httpx.get", return_value=_mock_slot(True, 100)), \
         patch("time.monotonic", return_value=t0):
        collector.poll()

    # n_decoded repart de 0 → nouvelle génération
    with patch("httpx.get", return_value=_mock_slot(True, 5)), \
         patch("time.monotonic", return_value=t0 + 0.5):
        stats = collector.poll()

    assert stats.gen_tps is None


def test_history_accumulates_tps_values(collector):
    t0 = time.monotonic()
    with patch("httpx.get", return_value=_mock_slot(True, 0)), \
         patch("time.monotonic", return_value=t0):
        collector.poll()

    for i in range(5):
        with patch("httpx.get", return_value=_mock_slot(True, (i + 1) * 10)), \
             patch("time.monotonic", return_value=t0 + (i + 1) * 1.0):
            collector.poll()

    assert len(collector.history) == 5
    assert all(v > 0 for v in collector.history)


def test_history_capped_at_max(collector):
    t0 = time.monotonic()
    with patch("httpx.get", return_value=_mock_slot(True, 0)), \
         patch("time.monotonic", return_value=t0):
        collector.poll()

    for i in range(collector.MAX_HISTORY + 10):
        with patch("httpx.get", return_value=_mock_slot(True, (i + 1) * 5)), \
             patch("time.monotonic", return_value=t0 + (i + 1) * 0.5):
            collector.poll()

    assert len(collector.history) == collector.MAX_HISTORY


def test_poll_returns_idle_on_server_error(collector):
    with patch("httpx.get", side_effect=Exception("connection refused")):
        stats = collector.poll()
    assert stats.is_generating is False
    assert stats.gen_tps is None


# ── Champs étendus (prompt/cache/ctx/avg/peak) ───────────────────────────────

def test_prompt_and_ctx_propagated(collector):
    with patch("httpx.get", return_value=_mock_slot(False, 0, n_ctx=131072, n_prompt=512, n_cached=128)):
        stats = collector.poll()
    assert stats.n_ctx == 131072
    assert stats.prompt_tokens == 512
    assert stats.prompt_cached == 128


def test_cache_hit_ratio(collector):
    s = TokenStats(prompt_tokens=384, prompt_cached=128)
    assert s.cache_hit_ratio == pytest.approx(0.25, rel=0.01)


def test_cache_hit_ratio_none_when_no_prompt():
    assert TokenStats().cache_hit_ratio is None


def test_ctx_used_pct(collector):
    s = TokenStats(prompt_tokens=13107, n_ctx=131072)
    assert s.ctx_used_pct == pytest.approx(10.0, rel=0.01)


def test_ctx_used_pct_none_when_no_ctx():
    assert TokenStats().ctx_used_pct is None


def test_avg_and_peak_tps_track_history(collector):
    t0 = time.monotonic()
    for i in range(3):
        with patch("httpx.get", return_value=_mock_slot(True, (i + 1) * 10)), \
             patch("time.monotonic", return_value=t0 + i * 1.0):
            collector.poll()
        with patch("httpx.get", return_value=_mock_slot(True, (i + 1) * 10 + 10)), \
             patch("time.monotonic", return_value=t0 + i * 1.0 + 0.5):
            stats = collector.poll()
    assert stats.avg_tps is not None
    assert stats.peak_tps is not None
    assert stats.peak_tps >= stats.avg_tps


def test_total_generated_accumulates_across_generations(collector):
    t0 = time.monotonic()
    # 1ère génération: 20 tokens
    with patch("httpx.get", return_value=_mock_slot(True, 20)), \
         patch("time.monotonic", return_value=t0):
        collector.poll()
    # reset → nouvelle génération 5 tokens (archive les 20)
    with patch("httpx.get", return_value=_mock_slot(True, 5)), \
         patch("time.monotonic", return_value=t0 + 1.0):
        collector.poll()
    with patch("httpx.get", return_value=_mock_slot(False, 0)), \
         patch("time.monotonic", return_value=t0 + 1.5):
        stats = collector.poll()
    assert stats.total_generated >= 20


def test_cache_history_not_polluted_when_idle(collector):
    """Idle sans génération → cache_history ne reçoit pas de points à 0%."""
    with patch("httpx.get", return_value=_mock_slot(False, 0, n_prompt=10, n_cached=2)):
        collector.poll()
    with patch("httpx.get", return_value=_mock_slot(False, 0)):
        collector.poll()
    with patch("httpx.get", return_value=_mock_slot(False, 0)):
        collector.poll()
    assert len(collector.cache_history) == 0, (
        f"cache_history polluée en idle: {list(collector.cache_history)}")


def test_cache_history_records_final_value_on_generation_end(collector):
    """Fin de génération : le dernier cache % est archivé une seule fois."""
    with patch("httpx.get", return_value=_mock_slot(False, 0)):
        collector.poll()
    # génération avec prompt partiellement caché
    with patch("httpx.get", return_value=_mock_slot(True, 10, n_ctx=1000, n_prompt=100, n_cached=50)):
        collector.poll()
    # fin de génération
    with patch("httpx.get", return_value=_mock_slot(False, 0, n_ctx=1000, n_prompt=100, n_cached=50)):
        stats = collector.poll()
    with patch("httpx.get", return_value=_mock_slot(False, 0)):
        collector.poll()
    vals = list(collector.cache_history)
    assert len(vals) == 2, f"attendu 2 points (gén + fin), obtenu {vals}"
