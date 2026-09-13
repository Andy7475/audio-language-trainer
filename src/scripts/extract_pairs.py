"""Extract Swedish/English sentence pairs from svenska-verb-fraser.json into a CSV.

Column A = Swedish, column B = English. All forms are included together;
no distinction is made between presens, preteritum and supinum.

Delete clusters from ALL_CLUSTERS (or pass your own list) to narrow the output.
"""

import csv
import json
from pathlib import Path

ALL_CLUSTERS = [
    "a-crowd",  # the -a crowd            22 verbs
    "split-de",  # split cluster, -de half 14 verbs
    "split-te",  # split cluster, -te half 14 verbs
    "stubs",  # the stubs                7 verbs
    "vowel-changers",  # the vowel changers      53 verbs
    "wildcards",  # the wildcards           22 verbs
]

HERE = Path(__file__).parent
JSON_PATH = HERE / "svenska-verb-fraser.json"
CSV_PATH = HERE / "verb-pairs.csv"


def extract_pairs(
    verb_clusters: list[str] = ALL_CLUSTERS, json_path: Path = JSON_PATH
) -> list[tuple[str, str]]:
    """Return [(swedish, english), ...] for the given cluster ids."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    wanted = set(verb_clusters)

    known = {c["id"] for c in data["clusters"]}
    unknown = wanted - known
    if unknown:
        raise ValueError(
            f"unknown cluster id(s): {sorted(unknown)}. " f"available: {sorted(known)}"
        )

    return [
        (example["sv"], example["en"])
        for cluster in data["clusters"]
        if cluster["id"] in wanted
        for verb in cluster["verbs"]
        for example in verb["examples"]
    ]


def write_csv(
    pairs: list[tuple[str, str]], csv_path: Path = CSV_PATH, header: bool = True
) -> Path:
    """Write the pairs to CSV. utf-8-sig so Excel reads å ä ö correctly."""
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(["swedish", "english"])
        writer.writerows(pairs)
    return csv_path


if __name__ == "__main__":
    pairs = extract_pairs()
    path = write_csv(pairs)
    print(f"{len(pairs)} pairs written to {path}")
