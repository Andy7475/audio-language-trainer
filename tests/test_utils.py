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
