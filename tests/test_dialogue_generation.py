import unittest
from unittest.mock import patch, mock_open
from typing import List
from src.dialogue_generation import (
    get_current_story_part,
    get_last_recap,
    load_story_plan,
    load_recaps,
    get_story_recap,
)


class TestStoryProgressionFunctions(unittest.TestCase):
    def setUp(self):
        self.story_plan = {
            "exposition": "Start of the story",
            "rising_action": "Middle part",
            "climax": "Exciting part",
            "falling_action": "Wrapping up",
            "resolution": "The end",
        }

    def test_get_current_story_part_beginning(self):
        recaps = []
        result = get_current_story_part(recaps, self.story_plan)
        self.assertEqual(result, "exposition")

    def test_get_current_story_part_middle(self):
        recaps = ["Recap 1", "Recap 2", "Recap 3"]
        result = get_current_story_part(recaps, self.story_plan)
        self.assertEqual(result, "falling_action")

    def test_get_current_story_part_end(self):
        recaps = ["Recap 1", "Recap 2", "Recap 3", "Recap 4"]
        result = get_current_story_part(recaps, self.story_plan)
        self.assertEqual(result, "resolution")

    def test_get_last_recap_empty(self):
        recaps = []
        result = get_last_recap(recaps)
        self.assertEqual(result, "This is the beginning of the story.")

    def test_get_last_recap_with_recaps(self):
        recaps = ["Recap 1", "Recap 2", "Last recap"]
        result = get_last_recap(recaps)
        self.assertEqual(result, "Last recap")

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"exposition": "Start", "resolution": "End"}',
    )
    def test_load_story_plan(self, mock_file):
        result = load_story_plan()
        self.assertEqual(result, {"exposition": "Start", "resolution": "End"})

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"recaps": ["Recap 1", "Recap 2"]}',
    )
    def test_load_recaps(self, mock_file):
        result = load_recaps()
        self.assertEqual(result, ["Recap 1", "Recap 2"])

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_load_story_plan_file_not_found(self, mock_file):
        result = load_story_plan()
        self.assertEqual(result, {})

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_load_recaps_file_not_found(self, mock_file):
        result = load_recaps()
        self.assertEqual(result, [])

    @patch("os.path.exists")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"recaps": ["Recap 1", "Recap 2", "Recap 3"]}',
    )
    def test_get_story_recap_all(self, mock_file, mock_exists):
        mock_exists.return_value = True
        result = get_story_recap()
        self.assertEqual(result, ["Recap 1", "Recap 2", "Recap 3"])

    @patch("os.path.exists")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"recaps": ["Recap 1", "Recap 2", "Recap 3"]}',
    )
    def test_get_story_recap_limited(self, mock_file, mock_exists):
        mock_exists.return_value = True
        result = get_story_recap(2)
        self.assertEqual(result, ["Recap 2", "Recap 3"])

    @patch("os.path.exists")
    def test_get_story_recap_no_file(self, mock_exists):
        mock_exists.return_value = False
        result = get_story_recap()
        self.assertEqual(result, [])
