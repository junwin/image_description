"""Older-post sampling uses saved stats without hiding recent publications."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from image_description.cli import ratings_cli

NOW = datetime(2026, 11, 7, 12, tzinfo=timezone.utc)


def entry(days=28, **overrides):
    return {
        "platform": "mastodon", "post_id": "old",
        "date": (NOW - timedelta(days=days)).isoformat(),
        "engagement": {"likes": 5, "reposts": 9, "comments": 2, "score": 11,
                       "scanned_at": (NOW - timedelta(days=7)).isoformat()},
        **overrides,
    }


@pytest.fixture
def scanner(monkeypatch, tmp_path):
    monkeypatch.setattr(ratings_cli, "_now", lambda: NOW)
    monkeypatch.setattr(ratings_cli.time, "sleep", lambda _: None)
    fetch = Mock(return_value=(8, 9, 2))
    monkeypatch.setitem(ratings_cli.FETCHERS, "mastodon", fetch)
    path = tmp_path / "photo.json"

    def run(entries, **kwargs):
        path.write_text(json.dumps({"publish_history": entries}))
        result = ratings_cli.scan_sidecar(str(path), **kwargs)
        return result, json.loads(path.read_text())

    return run, fetch


@pytest.mark.parametrize("draw,expected_calls", [(0.199, 1), (0.2, 0), (0.9, 0)])
def test_probability_boundary_preserves_skipped_stats(scanner, monkeypatch, draw, expected_calls):
    run, fetch = scanner
    monkeypatch.setattr(ratings_cli.random, "random", lambda: draw)
    original = entry()
    result, saved = run([original])
    assert fetch.call_count == result["live_fetches"] == expected_calls
    if not expected_calls:
        assert result["platforms"] == [{
            "platform": "mastodon", "skipped": "older post",
            "scanned_at": original["engagement"]["scanned_at"],
        }]
        assert saved["publish_history"][0]["engagement"] == original["engagement"]
        assert saved["engagement_score"] == 11


@pytest.mark.parametrize("changes", [
    {"engagement": None}, {"engagement": {}},
    {"date": None}, {"date": ""}, {"date": "invalid"},
    {"engagement": {"score": 11, "scanned_at": "invalid"}},
])
def test_missing_stats_or_unknown_dates_fetch_normally(scanner, changes):
    run, fetch = scanner
    result, _ = run([entry(**changes)], older_post_probability=0)
    assert result["live_fetches"] == fetch.call_count == 1


@pytest.mark.parametrize("days", [0, 20, 21, -1])
def test_recent_posts_and_exact_threshold_are_not_sampled(scanner, days):
    run, fetch = scanner
    run([entry(days)], older_post_probability=0)
    fetch.assert_called_once()


def test_each_platform_publication_uses_its_own_date(scanner):
    run, fetch = scanner
    result, _ = run([entry(), entry(2, post_id="new")], older_post_probability=0)
    fetch.assert_called_once_with("new")
    assert result["live_fetches"] == 1


def test_custom_age_threshold_and_naive_date(scanner):
    run, fetch = scanner
    run([entry(date=(NOW - timedelta(days=28)).replace(tzinfo=None).isoformat())],
        older_than_weeks=5, older_post_probability=0)
    fetch.assert_called_once()


def test_fresh_cache_wins_even_with_probability_one(scanner):
    run, fetch = scanner
    post = entry()
    post["engagement"]["scanned_at"] = NOW.isoformat()
    result, _ = run([post], older_post_probability=1)
    fetch.assert_not_called()
    assert result["platforms"][0]["cached"] is True


def test_refresh_bypasses_cache_and_sampling(scanner):
    run, fetch = scanner
    old, fresh = entry(), entry(post_id="fresh")
    fresh["engagement"]["scanned_at"] = NOW.isoformat()
    run([old, fresh], refresh=True, older_post_probability=0)
    assert fetch.call_count == 2


def test_probability_one_fetches_all_stale_posts(scanner):
    run, fetch = scanner
    run([entry()], older_post_probability=1)
    fetch.assert_called_once()


def test_platform_filter_still_applies_to_refresh(scanner):
    run, fetch = scanner
    run([entry()], refresh=True, platforms=["facebook"])
    fetch.assert_not_called()


def test_dry_run_does_not_write(scanner):
    run, fetch = scanner
    post = entry()
    _, saved = run([post], dry_run=True, refresh=True)
    fetch.assert_called_once()
    assert saved == {"publish_history": [post]}


@pytest.mark.parametrize("option,value", [
    ("--older-than-weeks", "-1"), ("--older-than-weeks", "nan"),
    ("--older-than-weeks", "inf"), ("--older-post-probability", "-0.1"),
    ("--older-post-probability", "1.1"), ("--older-post-probability", "nan"),
])
def test_invalid_options_rejected(option, value):
    with pytest.raises(SystemExit) as exc:
        ratings_cli.main(["scan", "unused", option, value])
    assert exc.value.code == 2


def test_cli_options_reach_scanner(monkeypatch):
    command = Mock()
    monkeypatch.setattr(ratings_cli, "cmd_scan", command)
    ratings_cli.main(["scan", "photos", "--older-than-weeks", "6",
                      "--older-post-probability", "0.1"])
    assert command.call_args.kwargs["older_than_weeks"] == 6
    assert command.call_args.kwargs["older_post_probability"] == 0.1


@pytest.mark.parametrize("state", ["skipped", "cached", "fetched", "error"])
def test_scan_output_includes_last_successful_check(scanner, capsys, state):
    run, fetch = scanner
    post = entry()
    if state == "cached":
        post["engagement"]["scanned_at"] = NOW.isoformat()
    if state == "error":
        fetch.side_effect = RuntimeError("service unavailable")
    # Use the path returned by the scan to exercise the actual command renderer.
    result, _ = run([post], older_post_probability=0)
    ratings_cli.cmd_scan(
        str(Path(result["path"]).parent), platforms=None,
        dry_run=False, as_json=False, refresh=state in {"fetched", "error"},
        file_delay=0, older_post_probability=0,
    )
    expected = NOW.isoformat() if state in {"cached", "fetched"} else post["engagement"]["scanned_at"]
    assert f"last checked: {expected}" in capsys.readouterr().out


def test_json_rows_include_check_timestamp(scanner):
    run, _ = scanner
    result, _ = run([entry()], refresh=True)
    assert result["platforms"][0]["scanned_at"] == NOW.isoformat()
