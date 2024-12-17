import pytest
from pathlib import Path

# Import the functions we want to test
from src.utils import sanitize_path_component, construct_gcs_path


@pytest.mark.parametrize(
    "input_str, expected",
    [
        # Basic sanitization
        ("Hello World", "hello_world"),
        ("hello-world", "hello-world"),
        ("hello_world", "hello_world"),
        # Special characters
        ("Hello@World!", "helloworld"),
        ("My&Story#Here", "mystoryhere"),
        ("Story (Draft)", "story_draft"),
        # Multiple spaces and mixed case
        ("My   Long    Story", "my_long_story"),
        ("UPPER lower MiXeD", "upper_lower_mixed"),
        # Non-alphanumeric characters
        ("café.story", "caféstory"),
        ("über-fast", "über-fast"),
    ],
)
def test_sanitize_path_component(input_str, expected):
    assert sanitize_path_component(input_str) == expected


@pytest.mark.parametrize(
    "story_name, language_name, bucket_name, expected",
    [
        # Basic path construction
        (
            "my_story.html",
            "Swedish",
            "test-bucket",
            "gs://test-bucket/swedish/my_story/my_story.html",
        ),
        # Story name with extension gets cleaned
        (
            "complex story.html",
            "French",
            "test-bucket",
            "gs://test-bucket/french/complex_story/complex_story.html",
        ),
        # Language with spaces and special chars
        (
            "story",
            "Modern Greek",
            "test-bucket",
            "gs://test-bucket/modern_greek/story/story.html",
        ),
        # Path-like story name gets cleaned
        (
            "folder/subfolder/story.html",
            "English",
            "test-bucket",
            "gs://test-bucket/english/story/story.html",
        ),
        # Special characters in all components
        (
            "my@story!.html",
            "Swiss-German",
            "test-bucket!",
            "gs://test-bucket!/swiss-german/mystory/mystory.html",
        ),
    ],
)
def test_construct_gcs_path(story_name, language_name, bucket_name, expected):
    result = construct_gcs_path(story_name, language_name, bucket_name)
    assert result == expected


def test_construct_gcs_path_default_config(monkeypatch):
    # Mock the config values
    class MockConfig:
        GCS_PUBLIC_BUCKET = "default-bucket"
        TARGET_LANGUAGE_NAME = "Spanish"

    monkeypatch.setattr("src.utils.config", MockConfig())

    # Test with defaults
    result = construct_gcs_path("story.html")
    assert result == "gs://default-bucket/spanish/story/story.html"

    # Test overriding just the language
    result = construct_gcs_path("story.html", language_name="French")
    assert result == "gs://default-bucket/french/story/story.html"

    # Test overriding just the bucket
    result = construct_gcs_path("story.html", bucket_name="custom-bucket")
    assert result == "gs://custom-bucket/spanish/story/story.html"
