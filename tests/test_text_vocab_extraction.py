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

    def test_falls_back_to_token_when_lemma_is_mislemmatized(self):
        # spaCy's Swedish model mislemmatizes the plural noun "bränder"
        # (fires) to "bränd" (an unrelated verb-participle, "burnt"), which
        # has no noun/adj/adv Wiktionary entry. The surface token "bränder"
        # itself does have a noun entry, so it should rescue the check.
        assert _passes_wiktionary_check("bränd", "vocab", "sv") is False
        assert _passes_wiktionary_check("bränd", "vocab", "sv", token="bränder") is True

    def test_token_fallback_does_not_rescue_real_disfluencies(self):
        # A genuine non-word shouldn't pass just because a "token" happens to
        # be supplied.
        assert _passes_wiktionary_check("uh", "vocab", "sv", token="uh") is False


class TestExtractTextVocabDict:
    """SV_TEXT is short - every content word in it only occurs once, so these
    tests pass min_occurrences=1 to disable rare-word filtering (see
    TestMinOccurrencesFiltering for the filtering behavior itself)."""

    def test_disfluency_is_ignored_not_missing(self):
        vocab_dict, ignored = _extract_text_vocab_dict(
            SV_TEXT, get_language("sv-SE"), candidate_tokens=set(), min_occurrences=1
        )
        assert "uh" in ignored
        assert "uh" not in vocab_dict["verbs"]
        assert "uh" not in vocab_dict["vocab"]

    def test_content_words_classified_by_lemma_without_candidate_tokens(self):
        # No candidate tokens supplied -> every word falls back to its lemma form.
        vocab_dict, _ = _extract_text_vocab_dict(
            SV_TEXT, get_language("sv-SE"), candidate_tokens=set(), min_occurrences=1
        )
        assert "köpa" in vocab_dict["verbs"]
        assert "vara" in vocab_dict["verbs"]  # AUX counts as a verb
        assert "affär" in vocab_dict["vocab"]
        assert "äpple" in vocab_dict["vocab"]
        assert "gott" in vocab_dict["vocab"]

    def test_prefers_native_token_when_candidate_pool_has_it(self):
        vocab_dict, _ = _extract_text_vocab_dict(
            SV_TEXT,
            get_language("sv-SE"),
            candidate_tokens={"köpte"},
            min_occurrences=1,
        )
        assert "köpte" in vocab_dict["verbs"]
        assert "köpa" not in vocab_dict["verbs"]

    def test_falls_back_to_lemma_without_matching_candidate_token(self):
        vocab_dict, _ = _extract_text_vocab_dict(
            SV_TEXT, get_language("sv-SE"), candidate_tokens=set(), min_occurrences=1
        )
        assert "köpa" in vocab_dict["verbs"]
        assert "köpte" not in vocab_dict["verbs"]


class TestMinOccurrencesFiltering:
    """Rare lemmas (occurring < min_occurrences times) are folded into
    ignored_tokens instead of appearing in vocab_dict."""

    def test_default_min_occurrences_filters_words_seen_twice(self):
        # "äpple" appears twice, everything else in SV_TEXT once -> with the
        # default min_occurrences=3, all of it is filtered as rare.
        text = SV_TEXT + " Han åt ett äpple till."
        vocab_dict, ignored = _extract_text_vocab_dict(
            text, get_language("sv-SE"), candidate_tokens=set()
        )
        assert "äpple" not in vocab_dict["vocab"]
        assert "äpple" in ignored

    def test_lemma_reaching_threshold_across_inflected_forms_is_kept(self):
        # "köpte" (past) and "köper" (present) are different surface forms of
        # the same lemma "köpa" - the count must be taken on the lemma so
        # these three occurrences together clear min_occurrences=3.
        text = (
            "Han köpte ett äpple. Hon köper ett äpple. "
            "Igår köpte de äpplen tillsammans."
        )
        vocab_dict, ignored = _extract_text_vocab_dict(
            text, get_language("sv-SE"), candidate_tokens=set()
        )
        assert "köpa" in vocab_dict["verbs"]
        assert "köpa" not in ignored

    def test_min_occurrences_one_disables_filtering_for_short_text(self):
        vocab_dict, ignored = _extract_text_vocab_dict(
            SV_TEXT, get_language("sv-SE"), candidate_tokens=set(), min_occurrences=1
        )
        assert "äpple" in vocab_dict["vocab"]
        assert "köpa" in vocab_dict["verbs"]
        assert "affär" in vocab_dict["vocab"]


class TestMislemmatizedTokenFallback:
    """End-to-end: _extract_text_vocab_dict should keep words whose spaCy
    lemma is wrong but whose surface token is a valid Wiktionary entry."""

    def test_bränder_kept_via_token_fallback(self):
        text = "Han berättar om bränder. Bränderna blåser aska."
        vocab_dict, ignored = _extract_text_vocab_dict(
            text, get_language("sv-SE"), candidate_tokens=set(), min_occurrences=1
        )
        assert "bränder" not in ignored
        assert "bränderna" not in ignored
        assert (
            "bränd" in vocab_dict["vocab"]
        )  # still the (wrong) lemma as coverage target


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
