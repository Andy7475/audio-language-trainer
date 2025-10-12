import pysrt
import re
import json
from typing import List, Dict, Optional, Tuple
import os
import sys

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.config_loader import config
from src.gcs_storage import upload_to_gcs, get_subtitles_path, get_story_translated_dialogue_path
from src.convert import clean_filename
from src.story import prepare_dialogue_with_wiktionary

def subriptime_to_seconds(srt_time) -> float:
    """Convert SubRipTime to float seconds for easier calculations"""
    return (
        srt_time.hours * 3600
        + srt_time.minutes * 60
        + srt_time.seconds
        + srt_time.milliseconds / 1000.0
    )


def clean_subtitle_text(text: str) -> str:
    """Remove square brackets and clean whitespace"""
    cleaned = re.sub(r"\[.*?\]", "", text)
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def is_subtitle_useful(text: str) -> bool:
    """Check if subtitle has meaningful content after cleaning"""
    return len(clean_subtitle_text(text)) > 0


def calculate_overlap_duration(sub1, sub2) -> float:
    """Calculate overlap duration between two subtitles in seconds"""
    start1, end1 = subriptime_to_seconds(sub1.start), subriptime_to_seconds(sub1.end)
    start2, end2 = subriptime_to_seconds(sub2.start), subriptime_to_seconds(sub2.end)
    return max(0, min(end1, end2) - max(start1, start2))


def calculate_duration(sub) -> float:
    """Calculate subtitle duration in seconds"""
    return subriptime_to_seconds(sub.end) - subriptime_to_seconds(sub.start)


def calculate_overlap_ratio(sub1, sub2) -> float:
    """Return overlap as ratio of longest subtitle duration"""
    overlap_duration = calculate_overlap_duration(sub1, sub2)
    if overlap_duration == 0:
        return 0.0

    longest_duration = max(calculate_duration(sub1), calculate_duration(sub2))
    return overlap_duration / longest_duration if longest_duration > 0 else 0.0


def find_best_match(
    target_sub, candidates: List, used_indices: set, threshold: float = 0.9
) -> Optional[Tuple[object, int]]:
    """Find best matching subtitle based on overlap ratio, excluding used indices"""
    best_match, best_index, best_ratio = None, None, 0.0

    for i, candidate in enumerate(candidates):
        if i in used_indices:
            continue

        ratio = calculate_overlap_ratio(target_sub, candidate)
        if ratio >= threshold and ratio > best_ratio:
            best_match, best_index, best_ratio = candidate, i, ratio

    return (best_match, best_index) if best_match else None


class CombinedSubtitle:
    """Combined subtitle object for sequential matching"""

    def __init__(self, subtitles):
        self.start = subtitles[0].start
        self.end = subtitles[-1].end
        self.text = " ".join(clean_subtitle_text(sub.text) for sub in subtitles)


def combine_consecutive_subtitles(
    subs: List, start_idx: int, count: int
) -> Optional[CombinedSubtitle]:
    """Combine consecutive subtitles into a single subtitle object"""
    if start_idx + count > len(subs):
        return None
    return CombinedSubtitle(subs[start_idx : start_idx + count])


def find_sequential_match(
    target_sub,
    candidates: List,
    used_indices: set,
    threshold: float = 0.9,
    max_combine: int = 3,
) -> Optional[Tuple[CombinedSubtitle, List[int]]]:
    """Try to match target with 1-3 consecutive unused English subtitles"""
    best_match, best_indices, best_ratio = None, None, 0.0

    for combine_count in range(1, max_combine + 1):
        for start_idx in range(len(candidates) - combine_count + 1):
            indices = list(range(start_idx, start_idx + combine_count))

            if any(idx in used_indices for idx in indices):
                continue

            combined = combine_consecutive_subtitles(
                candidates, start_idx, combine_count
            )
            if not combined:
                continue

            ratio = calculate_overlap_ratio(target_sub, combined)
            if ratio >= threshold and ratio > best_ratio:
                best_match, best_indices, best_ratio = combined, indices, ratio

    return (best_match, best_indices) if best_match else None


def load_subtitle_files(swedish_path: str, english_path: str) -> Tuple[List, List]:
    """Load both SRT files separately"""
    try:
        return list(pysrt.open(swedish_path)), list(pysrt.open(english_path))
    except Exception as e:
        raise Exception(f"Error loading subtitle files: {e}")


def normalize_timestamp(timestamp_str: str) -> str:
    """Convert SRT timestamp format to standard format (comma to period)"""
    return timestamp_str.replace(",", ".")


def clean_vocabulary_pair(pair: Dict) -> Dict:
    """Clean a single vocabulary pair for JSON serialisation"""
    return {
        "swedish": pair["swedish"],
        "english": pair["english"],
        "match_type": pair["match_type"],
        "swedish_timing": {
            "start": normalize_timestamp(pair["swedish_timing"]["start"]),
            "end": normalize_timestamp(pair["swedish_timing"]["end"]),
            "duration": round(pair["swedish_timing"]["duration"], 3),
        },
        "english_timing": (
            {
                "start": normalize_timestamp(pair["english_timing"]["start"]),
                "end": normalize_timestamp(pair["english_timing"]["end"]),
                "duration": round(pair["english_timing"]["duration"], 3),
            }
            if pair["english_timing"]
            else None
        ),
        "overlap_ratio": round(pair["overlap_ratio"], 3),
    }


def create_vocabulary_pairs(
    swedish_path: str, english_path: str, threshold: float = 0.9
) -> List[Dict]:
    """Two-pass matching: first single subtitles, then sequential combinations for unmatched"""
    swedish_subs, english_subs = load_subtitle_files(swedish_path, english_path)
    results = []
    used_english_indices = set()

    # First pass: Single subtitle matching
    print("First pass: Single subtitle matching...")
    for swedish_sub in swedish_subs:
        cleaned_swedish = clean_subtitle_text(swedish_sub.text)
        if not is_subtitle_useful(swedish_sub.text):
            continue

        match_result = find_best_match(
            swedish_sub, english_subs, used_english_indices, threshold
        )

        if match_result:
            english_match, english_index = match_result
            used_english_indices.add(english_index)

            entry = {
                "swedish": cleaned_swedish,
                "english": clean_subtitle_text(english_match.text),
                "match_type": "single",
                "swedish_timing": {
                    "start": str(swedish_sub.start),
                    "end": str(swedish_sub.end),
                    "duration": calculate_duration(swedish_sub),
                },
                "english_timing": {
                    "start": str(english_match.start),
                    "end": str(english_match.end),
                    "duration": calculate_duration(english_match),
                },
                "overlap_ratio": calculate_overlap_ratio(swedish_sub, english_match),
            }
        else:
            entry = {
                "swedish": cleaned_swedish,
                "english": None,
                "match_type": "none",
                "swedish_timing": {
                    "start": str(swedish_sub.start),
                    "end": str(swedish_sub.end),
                    "duration": calculate_duration(swedish_sub),
                },
                "english_timing": None,
                "overlap_ratio": 0.0,
            }

        results.append(entry)

    # Second pass: Sequential matching for unmatched Swedish subtitles
    print("Second pass: Sequential matching for unmatched entries...")
    unmatched_count = 0
    sequential_matches = 0

    for entry in results:
        if entry["english"] is None:
            unmatched_count += 1
            swedish_sub = next(
                sub
                for sub in swedish_subs
                if str(sub.start) == entry["swedish_timing"]["start"]
            )

            sequential_result = find_sequential_match(
                swedish_sub, english_subs, used_english_indices, threshold
            )

            if sequential_result:
                combined_english, english_indices = sequential_result

                for idx in english_indices:
                    used_english_indices.add(idx)

                entry.update(
                    {
                        "english": combined_english.text,
                        "match_type": f"sequential_{len(english_indices)}",
                        "english_timing": {
                            "start": str(combined_english.start),
                            "end": str(combined_english.end),
                            "duration": calculate_duration(combined_english),
                        },
                        "overlap_ratio": calculate_overlap_ratio(
                            swedish_sub, combined_english
                        ),
                    }
                )
                sequential_matches += 1

    print(
        f"Second pass: Found {sequential_matches} additional matches out of {unmatched_count} unmatched entries"
    )
    return results


def preview_results(results: List[Dict], count: int = 10, show_unmatched: bool = True):
    """Preview results with improved formatting and filtering options"""
    matched = [r for r in results if r["english"] is not None]
    unmatched = [r for r in results if r["english"] is None]

    print(f"\n=== MATCHING SUMMARY ===")
    print(f"Total Swedish entries: {len(results)}")
    print(f"Matched: {len(matched)}")
    print(
        f"  - Single matches: {len([r for r in matched if r['match_type'] == 'single'])}"
    )
    print(
        f"  - Sequential matches: {len([r for r in matched if r['match_type'].startswith('sequential')])}"
    )
    print(f"Unmatched: {len(unmatched)}")

    print(f"\n=== FIRST {count} MATCHED ENTRIES ===")
    for i, entry in enumerate(matched[:count]):
        print(f"\n--- Entry {i+1} ({entry['match_type']}) ---")
        print(f"Swedish: {entry['swedish']}")
        print(f"English: {entry['english']}")
        print(f"Overlap: {entry['overlap_ratio']:.2f}")
        print(
            f"Swedish: {entry['swedish_timing']['start']} -> {entry['swedish_timing']['end']}"
        )
        if entry["english_timing"]:
            print(
                f"English: {entry['english_timing']['start']} -> {entry['english_timing']['end']}"
            )

    if show_unmatched and unmatched:
        print(f"\n=== FIRST {min(5, len(unmatched))} UNMATCHED ENTRIES ===")
        for i, entry in enumerate(unmatched[:5]):
            print(f"\n--- Unmatched {i+1} ---")
            print(f"Swedish: {entry['swedish']}")
            print(
                f"Timing: {entry['swedish_timing']['start']} -> {entry['swedish_timing']['end']}"
            )


def save_vocabulary_pairs(vocabulary_pairs: List[Dict], filepath: str) -> List[Dict]:
    """Save vocabulary pairs to JSON file with clean formatting"""
    cleaned_pairs = [clean_vocabulary_pair(pair) for pair in vocabulary_pairs]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(cleaned_pairs, f, indent=2, ensure_ascii=False)

    return cleaned_pairs


def load_vocabulary_pairs(filepath: str) -> List[Dict]:
    """Load vocabulary pairs from JSON file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_story_dialgogue_from_vocabulary_pairs(vocabulary_pairs: List[Dict]) -> List[str]:
    """Extract dialogue lines from vocabulary pairs and format them
    in the same way as our story data structure"""

    dialogue_dict = {"introduction" : {"dialogue": [], "translated_dialogue": []}}

    for pair in vocabulary_pairs:

        english_text = pair["english"]
        translated_text = pair[config.TARGET_LANGUAGE_NAME.lower()]

        english_utterance = {"speaker": "Sam", "text": english_text}
        translated_utterance = {"speaker": "Sam", "text": translated_text}
        dialogue_dict["introduction"]["dialogue"].append(english_utterance)
        dialogue_dict["introduction"]["translated_dialogue"].append(translated_utterance)
    return dialogue_dict

def get_subtitle_story_name(title: str, episode: int) -> str:
    """Generate a clean story name from title and episode number"""
    clean_title = clean_filename(title)
    clean_episode = str(episode).zfill(2)
    return f"story_{clean_title}_ep{clean_episode}"

def upload_subtitle_dialogue_to_gcs(
    vocabulary_pairs: List[Dict], title: str, episode: int, 
) -> str:
    """Upload vocabulary pairs to GCS using the subtitle path structure"""

    story_name = get_subtitle_story_name(title, episode)

    equivalent_story_path = get_story_translated_dialogue_path(story_name=story_name, collection="Subtitles")

    dialogue_dict = get_story_dialgogue_from_vocabulary_pairs(vocabulary_pairs)
    dialogue_dict = prepare_dialogue_with_wiktionary(dialogue_dict)
    return upload_to_gcs(
        obj=dialogue_dict,
        bucket_name=config.GCS_PRIVATE_BUCKET,
        file_name=equivalent_story_path,
        content_type="application/json",
    )


if __name__ == "__main__":
    target_file = "../data/swedish.srt"
    source_file = "../data/english.srt"
    TITLE = "The Are Murders"
    EPISODE = 3

    try:
        vocabulary_pairs = create_vocabulary_pairs(
            target_file, source_file, threshold=0.8
        )
        preview_results(vocabulary_pairs, count=5, show_unmatched=True)

        # Clean and upload
        cleaned_pairs = {
            "subtitles": [clean_vocabulary_pair(pair) for pair in vocabulary_pairs]
        }
        file_name = get_subtitles_path(title=TITLE, episode=EPISODE)
        print(f"file_name: {file_name}")

        gcs_uri = upload_to_gcs(
            obj=cleaned_pairs,  # Test with first 2 entries
            bucket_name=config.GCS_PRIVATE_BUCKET,
            file_name=file_name,
            content_type="application/json",
        )

        print(f"Successfully uploaded to: {gcs_uri}")

        print("Also uploading dialogue format to equivalent story path...")
        gcs_dialogue_uri = upload_subtitle_dialogue_to_gcs(
            vocabulary_pairs, title=TITLE, episode=EPISODE
        )
        print(f"Successfully uploaded dialogue to: {gcs_dialogue_uri}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
