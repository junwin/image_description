"""Facebook Graph engagement collection through the shared ratings scanner."""
import json
from unittest.mock import Mock

import pytest

from image_description.cli import facebook_cli, ratings_cli


def _mock_response(data, status=200):
    response = Mock()
    response.json.return_value = data
    response.status_code = status
    response.raise_for_status.return_value = None
    return response


def test_facebook_engagement_counts(monkeypatch):
    monkeypatch.setattr(facebook_cli, "load_credentials", lambda: ("page-123", "page-token"))
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return _mock_response({
            "reactions": {"summary": {"total_count": 11}},
            "comments": {"summary": {"total_count": 3}},
            "shares": {"count": 2},
        })

    monkeypatch.setattr(facebook_cli.requests, "get", fake_get)
    engagement, score = ratings_cli.fetch_engagement("facebook", "page-123_456")
    assert calls[0][0] == f"{facebook_cli.API_BASE}/page-123_456"
    assert calls[0][1]["headers"]["Authorization"] == "Bearer page-token"
    assert "access_token" not in calls[0][1]["params"]
    assert "reactions" in calls[0][1]["params"]["fields"]
    assert engagement["likes"] == 11
    assert engagement["reposts"] == 2
    assert engagement["comments"] == 3
    assert score == 11 + 2 * 2 + 3 * 3


def test_facebook_missing_shares_is_zero(monkeypatch):
    monkeypatch.setattr(facebook_cli, "load_credentials", lambda: ("page", "token"))
    monkeypatch.setattr(facebook_cli.requests, "get", lambda *a, **k: _mock_response({
        "reactions": {"summary": {"total_count": 0}},
        "comments": {"summary": {"total_count": 0}},
    }))
    assert ratings_cli.fetch_facebook("photo-id") == (0, 0, 0)


def test_facebook_unavailable_summaries_fail_instead_of_recording_zero(monkeypatch):
    monkeypatch.setattr(facebook_cli, "load_credentials", lambda: ("page", "token"))
    monkeypatch.setattr(facebook_cli.requests, "get", lambda *a, **k: _mock_response({"id": "photo-id"}))
    with pytest.raises(RuntimeError, match="summaries unavailable"):
        ratings_cli.fetch_facebook("photo-id")


def test_facebook_sidecar_scan_updates_existing_history(monkeypatch, tmp_path):
    path = tmp_path / "sample.json"
    path.write_text(json.dumps({
        "title": "Photo",
        "publish_history": [{
            "platform": "facebook",
            "post_id": "page-123_456",
            "url": "https://www.facebook.com/page-123_456",
            "date": "2026-10-20T12:00:00+00:00",
        }],
    }), encoding="utf-8")
    monkeypatch.setattr(ratings_cli, "fetch_facebook", lambda post_id: (8, 2, 1))
    monkeypatch.setitem(ratings_cli.FETCHERS, "facebook", ratings_cli.fetch_facebook)
    monkeypatch.setattr(ratings_cli.time, "sleep", lambda *a: None)
    result = ratings_cli.scan_sidecar(str(path), platforms=["facebook"])
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert result["errors"] == []
    assert saved["publish_history"][0]["engagement"]["likes"] == 8
    assert saved["publish_history"][0]["engagement"]["reposts"] == 2
    assert saved["publish_history"][0]["engagement"]["comments"] == 1
    assert saved["engagement_score"] == 15


def test_facebook_graph_error_does_not_overwrite_previous_data(monkeypatch, tmp_path):
    path = tmp_path / "sample.json"
    old = {"likes": 7, "reposts": 1, "comments": 2, "score": 15, "scanned_at": "2020-01-01T00:00:00+00:00"}
    path.write_text(json.dumps({"publish_history": [
        {"platform": "facebook", "post_id": "bad-id", "engagement": old}
    ]}), encoding="utf-8")
    def failed(_):
        raise RuntimeError("Graph API permissions error")
    monkeypatch.setitem(ratings_cli.FETCHERS, "facebook", failed)
    result = ratings_cli.scan_sidecar(str(path), platforms=["facebook"], refresh=True)
    assert result["errors"]
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["publish_history"][0]["engagement"] == old


def test_graph_error_exposes_code_but_never_token(monkeypatch):
    monkeypatch.setattr(facebook_cli, "load_credentials", lambda: ("page", "TOP_SECRET_TOKEN"))
    def bad_response(*args, **kwargs):
        response = _mock_response({"error": {
            "message": "(#100) Unsupported field requested",
            "code": 100,
            "error_subcode": 33,
        }}, status=400)
        return response
    monkeypatch.setattr(facebook_cli.requests, "get", bad_response)
    with pytest.raises(RuntimeError, match=r"code 100 subcode 33") as error:
        ratings_cli.fetch_facebook("page_456")
    assert "TOP_SECRET_TOKEN" not in str(error.value)
