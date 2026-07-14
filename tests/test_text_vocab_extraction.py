"""Tests for text -> vocab_dict extraction backing phrases.search.add_tags_from_text.

Uses real spaCy (Swedish) and the real local Wiktionary sqlite db, consistent
with this repo's existing convention of not mocking either.
"""

from models import get_language
from nlp import extract_lemmas_and_pos, extract_token_lemma_pos
from phrases.search import _extract_text_vocab_dict, _passes_wiktionary_check

SV_TEXT = "Han sprang till affären och köpte ett äpple. Uh, det var gott."


class TestPassesWiktionaryCheck:
    def test_real_verb_passes(self):
        assert _passes_wiktionary_check("springa", "verbs", "sv") is True

    def test_real_noun_passes(self):
        assert _passes_wiktionary_check("äpple", "vocab", "sv") is True

    def test_disfluency_fails_both_buckets(self):
        assert _passes_wiktionary_check("uh", "verbs", "sv") is False
        assert _passes_wiktionary_check("uh", "vocab", "sv") is False


class TestExtractTextVocabDict:
    def test_disfluency_is_ignored_not_missing(self):
        vocab_dict, ignored = _extract_text_vocab_dict(
            SV_TEXT, get_language("sv-SE"), candidate_tokens=set()
        )
        assert "uh" in ignored
        assert "uh" not in vocab_dict["verbs"]
        assert "uh" not in vocab_dict["vocab"]

    def test_content_words_classified_by_lemma_without_candidate_tokens(self):
        # No candidate tokens supplied -> every word falls back to its lemma form.
        vocab_dict, _ = _extract_text_vocab_dict(
            SV_TEXT, get_language("sv-SE"), candidate_tokens=set()
        )
        assert "köpa" in vocab_dict["verbs"]
        assert "vara" in vocab_dict["verbs"]  # AUX counts as a verb
        assert "affär" in vocab_dict["vocab"]
        assert "äpple" in vocab_dict["vocab"]
        assert "gott" in vocab_dict["vocab"]

    def test_prefers_native_token_when_candidate_pool_has_it(self):
        vocab_dict, _ = _extract_text_vocab_dict(
            SV_TEXT, get_language("sv-SE"), candidate_tokens={"köpte"}
        )
        assert "köpte" in vocab_dict["verbs"]
        assert "köpa" not in vocab_dict["verbs"]

    def test_falls_back_to_lemma_without_matching_candidate_token(self):
        vocab_dict, _ = _extract_text_vocab_dict(
            SV_TEXT, get_language("sv-SE"), candidate_tokens=set()
        )
        assert "köpa" in vocab_dict["verbs"]
        assert "köpte" not in vocab_dict["verbs"]


class TestExtractLemmasAndPosRegression:
    """extract_lemmas_and_pos now delegates to _extract_token_lemma_pos_for_text
    internally (via the new extract_token_lemma_pos) — output must stay identical."""

    def test_output_matches_triples_lemma_and_pos(self):
        sentences = ["Han sprang hem.", "Hon äter ett äpple."]
        result = extract_lemmas_and_pos(sentences, "sv")

        expected = []
        for sentence in sentences:
            triples = extract_token_lemma_pos(sentence, "sv")
            expected.extend((lemma, pos) for _, lemma, pos in triples)

        assert result == expected
