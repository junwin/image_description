"""Scores reward direct engagement while retaining distribution metrics."""
import json

from image_description.cli import ratings_cli
from image_description.sidecar import Sidecar


def test_reposts_do_not_change_platform_score(monkeypatch):
    monkeypatch.setitem(ratings_cli.FETCHERS, "mastodon", lambda post_id: (12, 28, 1))
    engagement, score = ratings_cli.fetch_engagement("mastodon", "post-1")
    assert engagement["reposts"] == 28
    assert engagement["score"] == score == 15


def test_rescore_existing_sidecar_without_network(tmp_path):
    path = tmp_path / "photo.json"
    path.write_text(json.dumps({
        "title": "Photograph",
        "publish_history": [
            {"platform": "mastodon", "post_id": "1",
             "engagement": {"likes": 12, "reposts": 28, "comments": 1,
                            "score": 71, "scanned_at": "2026-10-22T12:00:00+00:00"}},
            {"platform": "bluesky", "post_id": "2",
             "engagement": {"likes": 20, "reposts": 3, "comments": 3,
                            "score": 35, "scanned_at": "2026-10-22T12:00:00+00:00"}},
        ],
    }), encoding="utf-8")
    ratings_cli.cmd_score(str(tmp_path), as_json=False)
    data = json.loads(path.read_text(encoding="utf-8"))
    first, second = [p["engagement"] for p in data["publish_history"]]
    assert first["reposts"] == 28
    assert second["reposts"] == 3
    assert first["score"] == 15
    assert second["score"] == 29
    assert data["engagement_score"] == 44
    assert data["normalized_score"] == 22


def test_cached_scan_rescores_old_engagement(monkeypatch, tmp_path):
    path = tmp_path / "photo.json"
    path.write_text(json.dumps({
        "publish_history": [{
            "platform": "mastodon", "post_id": "1",
            "engagement": {"likes": 5, "reposts": 30, "comments": 2,
                           "score": 71, "scanned_at": ratings_cli._now_iso()}
        }]
    }), encoding="utf-8")
    monkeypatch.setitem(ratings_cli.FETCHERS, "mastodon",
                        lambda _: (_ for _ in ()).throw(AssertionError("should use cache")))
    ratings_cli.scan_sidecar(str(path))
    data = Sidecar.load(str(path))
    assert data.publish_history[0]["engagement"]["reposts"] == 30
    assert data.publish_history[0]["engagement"]["score"] == 11
    assert data.extra["engagement_score"] == 11
