"""Tests for social_utils shared module."""
import pytest
from image_description.social_utils import build_social_text, build_alt_text, process_hashtags


class TestProcessHashtags:
    def test_removes_places_and_genre(self):
        result = process_hashtags("#photography #places #sunset #genre")
        assert "#places" not in result
        assert "#genre" not in result
        assert "#photography" in result
        assert "#sunset" in result

    def test_prepends_photography_and_photo(self):
        result = process_hashtags("#sunset")
        tags = result.split()
        assert tags[0] == "#photography"
        assert tags[1] == "#photo"
        assert "#sunset" in tags

    def test_dedupes_prepended_tags(self):
        result = process_hashtags("#photography #photo #sunset")
        tags = result.split()
        # Should have exactly one of each
        assert tags.count("#photography") == 1
        assert tags.count("#photo") == 1

    def test_empty_input(self):
        assert process_hashtags("") == "#photography #photo"


class TestBuildSocialText:
    def test_uses_title_and_caption_field(self):
        sidecar = {
            "original_title": "Sunset Over Hills",
            "title": "Sunset",
            "social_caption": "A beautiful evening",
            "image_description": "Golden sunset",
            "hashtags": "#sunset #golden",
        }
        result = build_social_text(sidecar, caption_field="social_caption")
        assert "Sunset Over Hills" in result
        assert "A beautiful evening" in result
        assert "#photography" in result
        assert "#sunset" in result

    def test_falls_back_title_to_title(self):
        sidecar = {
            "title": "Fallback Title",
            "social_caption": "Caption text",
            "hashtags": "",
        }
        result = build_social_text(sidecar, caption_field="social_caption")
        assert "Fallback Title" in result
        assert "Caption text" in result

    def test_uses_custom_caption_field(self):
        sidecar = {
            "original_title": "Test",
            "social_caption": "Short social caption",
            "enhanced_description": "Longer enhanced description",
            "hashtags": "#test",
        }
        # Default: social_caption
        result_default = build_social_text(sidecar)
        assert "Short social caption" in result_default
        assert "Longer enhanced description" not in result_default

        # Custom: enhanced_description
        result_custom = build_social_text(sidecar, caption_field="enhanced_description")
        assert "Longer enhanced description" in result_custom
        assert "Short social caption" not in result_custom

    def test_extra_text_appended(self):
        sidecar = {
            "title": "Photo",
            "social_caption": "Nice view",
            "hashtags": "",
        }
        result = build_social_text(sidecar, extra_text="Check this out!")
        assert "Check this out!" in result

    def test_char_limit_truncates(self):
        sidecar = {
            "title": "A" * 50,
            "social_caption": "B" * 50,
            "hashtags": "",
        }
        result = build_social_text(sidecar, char_limit=20)
        assert len(result) <= 20
        assert result.endswith("...")

    def test_missing_caption_field_empty(self):
        sidecar = {
            "original_title": "Just a title",
            "hashtags": "#photo",
        }
        result = build_social_text(sidecar, caption_field="social_caption")
        assert "Just a title" in result
        # No caption from social_caption since it's missing
        assert "#photography" in result


class TestBuildAltText:
    def test_uses_image_description(self):
        sidecar = {
            "image_description": "A golden sunset over rolling hills",
            "title": "Sunset",
        }
        result = build_alt_text(sidecar)
        assert result == "A golden sunset over rolling hills"

    def test_falls_back_to_title(self):
        sidecar = {
            "image_description": "",
            "title": "Fallback Title",
        }
        result = build_alt_text(sidecar)
        assert result == "Fallback Title"

    def test_empty_when_nothing_available(self):
        sidecar = {}
        result = build_alt_text(sidecar)
        assert result == ""
