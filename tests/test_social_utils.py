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
    def test_default_uses_original_title_and_description(self):
        sidecar = {
            "original_title": "Sunset Over Hills",
            "title": "Sunset",
            "original_description": "Taken on the ridge trail at golden hour.",
            "social_caption": "A beautiful evening",
            "image_description": "Golden sunset",
            "hashtags": "#sunset #golden",
        }
        result = build_social_text(sidecar)
        assert "Sunset Over Hills" in result
        assert "Taken on the ridge trail at golden hour." in result
        # social_caption is NOT used by default
        assert "A beautiful evening" not in result
        assert "#photography" in result
        assert "#sunset" in result

    def test_falls_back_title_to_title(self):
        sidecar = {
            "title": "Fallback Title",
            "original_description": "Caption text",
            "hashtags": "",
        }
        result = build_social_text(sidecar)
        assert "Fallback Title" in result
        assert "Caption text" in result

    def test_falls_back_to_image_description_when_no_original_words(self):
        sidecar = {
            "title": "Fallback Title",
            "original_description": "",
            "image_description": "AI generated description",
            "social_caption": "AI short caption",
            "hashtags": "",
        }
        result = build_social_text(sidecar)
        assert "AI generated description" in result
        # social_caption is NOT used by default
        assert "AI short caption" not in result

    def test_explicit_caption_field_overrides_default(self):
        sidecar = {
            "original_title": "Test",
            "original_description": "My own words",
            "social_caption": "Short social caption",
            "enhanced_description": "Longer enhanced description",
            "hashtags": "#test",
        }
        # Explicit social_caption (still available when wanted)
        result = build_social_text(sidecar, caption_field="social_caption")
        assert "Short social caption" in result
        assert "My own words" not in result

        # Explicit arbitrary field
        result_custom = build_social_text(sidecar, caption_field="enhanced_description")
        assert "Longer enhanced description" in result_custom
        assert "My own words" not in result_custom

    def test_extra_text_appended(self):
        sidecar = {
            "title": "Photo",
            "original_description": "Nice view",
            "hashtags": "",
        }
        result = build_social_text(sidecar, extra_text="Check this out!")
        assert "Check this out!" in result

    def test_char_limit_truncates(self):
        sidecar = {
            "title": "A" * 50,
            "original_description": "B" * 50,
            "hashtags": "",
        }
        result = build_social_text(sidecar, char_limit=20)
        assert len(result) <= 20
        assert result.endswith("...")

    def test_title_only_when_no_caption_available(self):
        sidecar = {
            "original_title": "Just a title",
            "hashtags": "#photo",
        }
        result = build_social_text(sidecar)
        assert "Just a title" in result
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
