"""Tests for phrase search and coverage algorithms."""

import pytest
from typing import Set

from src.phrases.phrase_model import Phrase, Translation
from src.phrases.search import (
    find_minimum_coverage_phrases,
    find_phrases_by_token_coverage,
)
from src.models import BCP47Language


@pytest.fixture
def sample_phrases():
    """Create a set of sample phrases for testing."""
    phrases = []

    # Phrase 1: "She runs to the store daily"
    phrase1 = Phrase(
        phrase_hash="she_runs_to_the_store_daily_abc123",
        english="She runs to the store daily",
        english_lower="she runs to the store daily",
        tokens=["she", "runs", "to", "the", "store", "daily"],
        verbs=["run"],
        vocab=["store", "daily"],
    )
    # Add French translation
    phrase1.translations["fr-FR"] = Translation(
        phrase_hash="she_runs_to_the_store_daily_abc123",
        language=BCP47Language.get("fr-FR"),
        text="Elle court au magasin tous les jours",
        text_lower="elle court au magasin tous les jours",
        tokens=["Elle", "court", "au", "magasin", "tous", "les", "jours"],
    )
    phrases.append(phrase1)

    # Phrase 2: "I eat an apple"
    phrase2 = Phrase(
        phrase_hash="i_eat_an_apple_def456",
        english="I eat an apple",
        english_lower="i eat an apple",
        tokens=["i", "eat", "an", "apple"],
        verbs=["eat"],
        vocab=["apple"],
    )
    phrase2.translations["fr-FR"] = Translation(
        phrase_hash="i_eat_an_apple_def456",
        language=BCP47Language.get("fr-FR"),
        text="Je mange une pomme",
        text_lower="je mange une pomme",
        tokens=["Je", "mange", "une", "pomme"],
    )
    phrases.append(phrase2)

    # Phrase 3: "He sleeps in the bed"
    phrase3 = Phrase(
        phrase_hash="he_sleeps_in_the_bed_ghi789",
        english="He sleeps in the bed",
        english_lower="he sleeps in the bed",
        tokens=["he", "sleeps", "in", "the", "bed"],
        verbs=["sleep"],
        vocab=["bed"],
    )
    phrase3.translations["fr-FR"] = Translation(
        phrase_hash="he_sleeps_in_the_bed_ghi789",
        language=BCP47Language.get("fr-FR"),
        text="Il dort dans le lit",
        text_lower="il dort dans le lit",
        tokens=["Il", "dort", "dans", "le", "lit"],
    )
    phrases.append(phrase3)

    # Phrase 4: "They run and eat fast" - covers multiple verbs
    phrase4 = Phrase(
        phrase_hash="they_run_and_eat_fast_jkl012",
        english="They run and eat fast",
        english_lower="they run and eat fast",
        tokens=["they", "run", "and", "eat", "fast"],
        verbs=["run", "eat"],
        vocab=["fast"],
    )
    phrase4.translations["fr-FR"] = Translation(
        phrase_hash="they_run_and_eat_fast_jkl012",
        language=BCP47Language.get("fr-FR"),
        text="Ils courent et mangent vite",
        text_lower="ils courent et mangent vite",
        tokens=["Ils", "courent", "et", "mangent", "vite"],
    )
    phrases.append(phrase4)

    # Phrase 5: "The store has apples and beds"
    phrase5 = Phrase(
        phrase_hash="the_store_has_apples_mno345",
        english="The store has apples and beds",
        english_lower="the store has apples and beds",
        tokens=["the", "store", "has", "apples", "and", "beds"],
        verbs=["have"],
        vocab=["store", "apple", "bed"],
    )
    phrase5.translations["fr-FR"] = Translation(
        phrase_hash="the_store_has_apples_mno345",
        language=BCP47Language.get("fr-FR"),
        text="Le magasin a des pommes et des lits",
        text_lower="le magasin a des pommes et des lits",
        tokens=["Le", "magasin", "a", "des", "pommes", "et", "des", "lits"],
    )
    phrases.append(phrase5)

    return phrases


class TestMinimumCoveragePhrases:
    """Tests for find_minimum_coverage_phrases function."""

    def test_basic_coverage(self, sample_phrases):
        """Test basic coverage with simple verb/vocab sets."""
        target_verbs = {"run", "eat"}
        target_vocab = {"store", "apple"}

        result = find_minimum_coverage_phrases(
            sample_phrases, target_verbs, target_vocab
        )

        # Should find at least one phrase covering the targets
        assert len(result) > 0

        # Verify complete coverage
        covered_verbs = set()
        covered_vocab = set()
        for phrase in result:
            covered_verbs.update(phrase.verbs)
            covered_vocab.update(phrase.vocab)

        assert target_verbs.issubset(covered_verbs)
        assert target_vocab.issubset(covered_vocab)

    def test_optimal_single_phrase_coverage(self, sample_phrases):
        """Test that algorithm finds single phrase when it provides complete coverage."""
        # Phrase 4 has both "run" and "eat"
        target_verbs = {"run", "eat"}
        target_vocab = set()

        result = find_minimum_coverage_phrases(
            sample_phrases, target_verbs, target_vocab
        )

        # Should select exactly one phrase (phrase 4)
        assert len(result) == 1
        assert set(result[0].verbs) == {"run", "eat"}

    def test_greedy_selection_efficiency(self, sample_phrases):
        """Test that greedy algorithm selects phrases efficiently."""
        target_verbs = {"run", "eat", "sleep"}
        target_vocab = {"store", "apple", "bed"}

        result = find_minimum_coverage_phrases(
            sample_phrases, target_verbs, target_vocab
        )

        # Verify coverage is complete
        covered_verbs = set()
        covered_vocab = set()
        for phrase in result:
            covered_verbs.update(phrase.verbs)
            covered_vocab.update(phrase.vocab)

        assert target_verbs.issubset(covered_verbs)
        assert target_vocab.issubset(covered_vocab)

        # Should use fewer than all phrases
        assert len(result) < len(sample_phrases)

    def test_empty_targets(self, sample_phrases):
        """Test with empty target sets."""
        result = find_minimum_coverage_phrases(sample_phrases, set(), set())
        assert result == []

    def test_impossible_coverage(self, sample_phrases):
        """Test when complete coverage is impossible."""
        target_verbs = {"fly", "swim", "dance"}  # Verbs not in any phrase
        target_vocab = {"car", "house"}  # Vocab not in any phrase

        result = find_minimum_coverage_phrases(
            sample_phrases, target_verbs, target_vocab
        )

        # Should return empty or partial coverage
        # The function will stop when no more coverage is possible
        covered_verbs = set()
        covered_vocab = set()
        for phrase in result:
            covered_verbs.update(phrase.verbs)
            covered_vocab.update(phrase.vocab)

        # Should not cover the impossible items
        assert not target_verbs.issubset(covered_verbs)
        assert not target_vocab.issubset(covered_vocab)

    def test_mixed_coverage(self, sample_phrases):
        """Test coverage with mix of verbs only and vocab only."""
        # Some exist, some don't
        target_verbs = {"run", "fly"}  # run exists, fly doesn't
        target_vocab = {"store", "car"}  # store exists, car doesn't

        result = find_minimum_coverage_phrases(
            sample_phrases, target_verbs, target_vocab
        )

        covered_items = set()
        for phrase in result:
            covered_items.update(phrase.verbs)
            covered_items.update(phrase.vocab)

        # Should cover the existing items
        assert "run" in covered_items
        assert "store" in covered_items
        # But not the impossible ones
        assert "fly" not in covered_items
        assert "car" not in covered_items


class TestTokenCoveragePhrases:
    """Tests for find_phrases_by_token_coverage function."""

    def test_basic_token_search(self, sample_phrases):
        """Test basic token search in French."""
        target_tokens = {"mange", "pomme"}  # eat, apple in French

        results = find_phrases_by_token_coverage(
            sample_phrases, target_tokens, "fr-FR", min_coverage_ratio=0.5
        )

        # Should find phrase 2: "Je mange une pomme"
        assert len(results) > 0
        phrase, ratio, matched = results[0]
        assert "mange" in matched
        assert "pomme" in matched
        assert ratio == 1.0  # 100% coverage (both tokens found)

    def test_coverage_ratio_calculation(self, sample_phrases):
        """Test that coverage ratio is calculated correctly."""
        target_tokens = {"mange", "pomme", "vite"}  # 3 tokens

        results = find_phrases_by_token_coverage(
            sample_phrases, target_tokens, "fr-FR", min_coverage_ratio=0.3
        )

        # Phrase 2 has 2/3 tokens (mange, pomme) = 66.7%
        # Phrase 4 has 2/3 tokens (mange, vite) = 66.7%
        assert len(results) >= 2

        for phrase, ratio, matched in results:
            # Each should have at least 2 tokens matched
            assert len(matched) >= 2
            # Ratio should be at least 0.66 (2/3)
            assert ratio >= 0.66

    def test_min_coverage_ratio_filter(self, sample_phrases):
        """Test that min_coverage_ratio filters results correctly."""
        target_tokens = {"mange", "pomme", "vite", "court"}  # 4 tokens

        # With high threshold (75%), only phrases with 3+ tokens should match
        results_high = find_phrases_by_token_coverage(
            sample_phrases, target_tokens, "fr-FR", min_coverage_ratio=0.75
        )

        # With low threshold (25%), phrases with 1+ token should match
        results_low = find_phrases_by_token_coverage(
            sample_phrases, target_tokens, "fr-FR", min_coverage_ratio=0.25
        )

        # More results with lower threshold
        assert len(results_low) >= len(results_high)

        # All high threshold results should have high coverage
        for phrase, ratio, matched in results_high:
            assert ratio >= 0.75

    def test_case_insensitive_matching(self, sample_phrases):
        """Test that token matching is case-insensitive."""
        # Try with different cases
        target_tokens_lower = {"mange", "pomme"}
        target_tokens_upper = {"MANGE", "POMME"}
        target_tokens_mixed = {"Mange", "Pomme"}

        results_lower = find_phrases_by_token_coverage(
            sample_phrases, target_tokens_lower, "fr-FR"
        )
        results_upper = find_phrases_by_token_coverage(
            sample_phrases, target_tokens_upper, "fr-FR"
        )
        results_mixed = find_phrases_by_token_coverage(
            sample_phrases, target_tokens_mixed, "fr-FR"
        )

        # All should return same results
        assert len(results_lower) == len(results_upper) == len(results_mixed)

    def test_language_filtering(self, sample_phrases):
        """Test that search only looks at specified language."""
        target_tokens = {"run", "store"}  # English tokens

        # Search in French - should find nothing (tokens are English)
        results_fr = find_phrases_by_token_coverage(
            sample_phrases, target_tokens, "fr-FR", min_coverage_ratio=0.5
        )

        # Should not find English tokens in French translations
        assert len(results_fr) == 0

    def test_sorting_by_coverage(self, sample_phrases):
        """Test that results are sorted by coverage ratio."""
        target_tokens = {"mange", "pomme", "vite"}

        results = find_phrases_by_token_coverage(
            sample_phrases, target_tokens, "fr-FR", min_coverage_ratio=0.3
        )

        # Results should be sorted by ratio (descending)
        for i in range(len(results) - 1):
            ratio1 = results[i][1]
            ratio2 = results[i + 1][1]
            # Current ratio should be >= next ratio
            assert ratio1 >= ratio2

    def test_no_matching_phrases(self, sample_phrases):
        """Test when no phrases match the tokens."""
        target_tokens = {"xyz", "abc", "def"}  # Non-existent tokens

        results = find_phrases_by_token_coverage(
            sample_phrases, target_tokens, "fr-FR", min_coverage_ratio=0.1
        )

        # Should return empty list
        assert len(results) == 0

    def test_language_as_string(self, sample_phrases):
        """Test that language parameter accepts string."""
        target_tokens = {"mange", "pomme"}

        # Should work with string
        results_str = find_phrases_by_token_coverage(
            sample_phrases, target_tokens, "fr-FR"
        )

        # Should work with BCP47Language object
        results_obj = find_phrases_by_token_coverage(
            sample_phrases, target_tokens, BCP47Language.get("fr-FR")
        )

        # Both should return same results
        assert len(results_str) == len(results_obj)

    def test_partial_token_match(self, sample_phrases):
        """Test behavior with partial token matches."""
        target_tokens = {"court", "magasin", "xyz"}  # 2 exist, 1 doesn't

        results = find_phrases_by_token_coverage(
            sample_phrases, target_tokens, "fr-FR", min_coverage_ratio=0.5
        )

        # Should find phrase 1 which has "court" and "magasin"
        assert len(results) > 0
        best_phrase, ratio, matched = results[0]
        assert "court" in matched
        assert "magasin" in matched
        assert "xyz" not in matched
        # 2 out of 3 tokens = 66.7%
        assert ratio >= 0.66


class TestIntegrationScenarios:
    """Integration tests combining multiple search strategies."""

    def test_find_then_filter_by_tokens(self, sample_phrases):
        """Test finding coverage then filtering by language tokens."""
        # Step 1: Find minimum phrases for verb coverage
        target_verbs = {"run", "eat"}
        target_vocab = {"store"}

        coverage_phrases = find_minimum_coverage_phrases(
            sample_phrases, target_verbs, target_vocab
        )

        # Step 2: Filter by French tokens
        french_tokens = {"court", "mange"}
        token_results = find_phrases_by_token_coverage(
            coverage_phrases, french_tokens, "fr-FR", min_coverage_ratio=0.5
        )

        # Should find phrases that both cover verbs/vocab AND have French tokens
        assert len(token_results) > 0

    def test_complete_workflow(self, sample_phrases):
        """Test complete workflow: coverage -> token filtering -> selection."""
        # Goal: Cover verbs {run, eat, sleep} with phrases that have French tokens {court, mange}
        target_verbs = {"run", "eat", "sleep"}
        target_vocab = set()

        # Get coverage
        coverage = find_minimum_coverage_phrases(
            sample_phrases, target_verbs, target_vocab
        )

        # Filter by French tokens
        french_tokens = {"court", "mange"}
        filtered = find_phrases_by_token_coverage(
            coverage, french_tokens, "fr-FR", min_coverage_ratio=0.5
        )

        # Should have some results
        assert len(filtered) > 0

        # Verify all results have the required French tokens
        for phrase, ratio, matched in filtered:
            assert len(matched & french_tokens) > 0
