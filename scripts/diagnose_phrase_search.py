"""Diagnostic script for find_phrases_by_vocab_dict algorithm.

Usage:
    cd "y:\\Python Scripts\\audio-language-trainer"
    python scripts/diagnose_phrase_search.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phrases.search import _load_phrases_with_translation, _to_lower

LANGUAGE_TAG = "sv-SE"


def _trace_greedy(phrases, target_verbs, target_vocab, language_tag, label):
    """Run greedy set-cover and trace each iteration."""
    remaining_verbs = _to_lower(set(target_verbs))
    remaining_vocab = _to_lower(set(target_vocab))
    pool = list(phrases)
    selected = []

    print(f"\n=== {label} ===")
    print(f"target_verbs={target_verbs}, target_vocab={target_vocab}")

    iteration = 0
    while remaining_verbs or remaining_vocab:
        iteration += 1
        best_phrase = None
        best_count = 0

        for phrase in pool:
            t = phrase.translations[language_tag]
            pv = _to_lower(set((t.verbs or []) + (t.tokens or [])))
            pw = _to_lower(set((t.vocab or []) + (t.tokens or [])))
            count = len(pv & remaining_verbs) + len(pw & remaining_vocab)
            if count > best_count:
                best_phrase = phrase
                best_count = count

        if best_phrase is None or best_count == 0:
            print(
                f"  iter {iteration}: no more coverage — missing verbs={remaining_verbs}, vocab={remaining_vocab}"
            )
            break

        t_best = best_phrase.translations[language_tag]
        raw_pv = (t_best.verbs or []) + (t_best.tokens or [])
        raw_pw = (t_best.vocab or []) + (t_best.tokens or [])

        print(f"\n  iter {iteration}: best='{t_best.text}' (covers {best_count})")
        print(f"    verbs stored: {t_best.verbs}")
        print(f"    tokens: {t_best.tokens}")

        selected.append(best_phrase)
        pool.remove(best_phrase)

        if "BUGGY" in label:
            # Original code: no lowercasing on removal
            removed_v = set(raw_pv) & remaining_verbs
            removed_w = set(raw_pw) & remaining_vocab
            remaining_verbs -= set(raw_pv)
            remaining_vocab -= set(raw_pw)
        else:
            # Fixed code: lowercase before subtracting
            removed_v = _to_lower(set(raw_pv)) & remaining_verbs
            removed_w = _to_lower(set(raw_pw)) & remaining_vocab
            remaining_verbs -= _to_lower(set(raw_pv))
            remaining_vocab -= _to_lower(set(raw_pw))

        print(f"    removed verbs: {removed_v}, removed vocab: {removed_w}")
        print(f"    remaining: verbs={remaining_verbs}, vocab={remaining_vocab}")

    print(f"\n  RESULT: {len(selected)} phrases selected")
    for p in selected:
        print(f"    '{p.translations[language_tag].text}'")
    return selected


def find_candidate_phrases(phrases, target_verbs, target_vocab, language_tag):
    """Show which phrases contain any of the target words."""
    print(f"\n--- Candidate phrases containing any of {target_verbs | target_vocab} ---")
    candidates = []
    for p in phrases:
        t = p.translations[language_tag]
        tokens_lower = _to_lower(set(t.tokens or []))
        verbs_lower = _to_lower(set(t.verbs or []))
        vocab_lower = _to_lower(set(t.vocab or []))

        hits_v = target_verbs & (verbs_lower | tokens_lower)
        hits_w = target_vocab & (vocab_lower | tokens_lower)
        if hits_v or hits_w:
            candidates.append(p)
            print(f"  '{t.text}'")
            print(f"    verbs={t.verbs}, vocab={t.vocab}")
            print(f"    tokens={t.tokens}")
            print(f"    hits: verbs={hits_v}, vocab={hits_w}")
    print(f"  Total candidates: {len(candidates)}")
    return candidates


def find_multi_verb_phrases(phrases, language_tag, min_verbs=2, limit=10):
    """Find phrases with multiple verbs — good for test fixture generation."""
    print(f"\n--- Phrases with {min_verbs}+ verbs in {language_tag} (top {limit}) ---")
    results = [
        p
        for p in phrases
        if len(p.translations[language_tag].verbs or []) >= min_verbs
    ]
    for p in results[:limit]:
        t = p.translations[language_tag]
        print(f"  EN: '{p.english}'")
        print(f"  {language_tag}: '{t.text}'")
        print(f"      verbs={t.verbs}, vocab={t.vocab}")
        print(f"      tokens={t.tokens}")
        print()
    return results


def main():
    print(f"Loading {LANGUAGE_TAG} phrases from Firestore (database=firephrases)...")
    phrases = _load_phrases_with_translation(language_tag=LANGUAGE_TAG)
    print(f"Loaded {len(phrases)} phrases with {LANGUAGE_TAG} translations")

    target_verbs = {"ska"}
    target_vocab = {"problem"}

    # 1. Show candidate phrases
    find_candidate_phrases(phrases, target_verbs, target_vocab, LANGUAGE_TAG)

    # 2. Trace buggy algorithm
    _trace_greedy(phrases, target_verbs, target_vocab, LANGUAGE_TAG, "BUGGY (current)")

    # 3. Trace fixed algorithm
    _trace_greedy(phrases, target_verbs, target_vocab, LANGUAGE_TAG, "FIXED (lowercase removals)")

    # 4. Output multi-verb phrases for test fixture generation
    find_multi_verb_phrases(phrases, LANGUAGE_TAG)


if __name__ == "__main__":
    main()
