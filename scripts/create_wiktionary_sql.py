import gzip
import json
import sqlite3
from tqdm import tqdm
from urllib.parse import quote

# Configuration
input_file = "../outputs/wiktionary_dump/raw-wiktextract-data.jsonl.gz"
db_file = "wiktionary_pos.db"
batch_size = 1_000_000

print("Creating database...")
conn = sqlite3.connect(db_file)
c = conn.cursor()

# Pragmas for fast bulk loading — disable safety features we don't need for a rebuild
c.execute("PRAGMA journal_mode = OFF")
c.execute("PRAGMA synchronous = OFF")
c.execute("PRAGMA cache_size = -1048576")  # 1GB page cache
c.execute("PRAGMA temp_store = MEMORY")

c.execute("DROP TABLE IF EXISTS entries")

# No PRIMARY KEY during load — index is built once at the end after all data is inserted
c.execute("""CREATE TABLE entries
             (word TEXT NOT NULL,
              lang_code TEXT NOT NULL,
              pos TEXT NOT NULL,
              url TEXT NOT NULL)""")

print("Processing Wiktionary dump...")
batch = []
total_entries = 0
i = 0

c.execute("BEGIN")
with gzip.open(input_file, "rt", encoding="utf-8") as f:
    for i, line in enumerate(tqdm(f, desc="Processing entries")):
        try:
            item = json.loads(line)
            word = item.get("word")
            lang_code = item.get("lang_code")
            lang = item.get("lang")
            pos = item.get("pos")

            if not word or not lang_code or not lang or not pos:
                continue

            url = f"https://en.wiktionary.org/wiki/{word}#{lang}"
            safe_url = url.replace(" ", "_")
            safe_url = quote(safe_url, safe=":/#")

            batch.append((word, lang_code, pos, safe_url))
            total_entries += 1

            if len(batch) >= batch_size:
                c.executemany("INSERT INTO entries VALUES (?,?,?,?)", batch)
                conn.commit()
                c.execute("BEGIN")
                batch = []

        except json.JSONDecodeError as e:
            print(f"Error parsing line {i}: {e}")
        except Exception as e:
            print(f"Error on line {i}: {e}")

if batch:
    c.executemany("INSERT INTO entries VALUES (?,?,?,?)", batch)
    conn.commit()

# Deduplicate (same word+lang+pos can appear multiple times in the dump)
print("\nDeduplicating...")
c.execute("""
    DELETE FROM entries WHERE rowid NOT IN (
        SELECT MIN(rowid) FROM entries GROUP BY word, lang_code, pos
    )
""")
conn.commit()

# Build the unique index once, over the final data
print("Creating index...")
c.execute("CREATE UNIQUE INDEX idx_lookup ON entries(word, lang_code, pos)")
conn.commit()

c.execute("SELECT COUNT(*) FROM entries")
count = c.fetchone()[0]

c.execute("SELECT COUNT(DISTINCT lang_code) FROM entries")
lang_count = c.fetchone()[0]

conn.close()

print("\n✓ Database created successfully!")
print(f"  File: {db_file}")
print(f"  Total entries: {count:,}")
print(f"  Languages: {lang_count}")
print(f"  Processed lines: {i + 1:,}")

# Test the lookup function
print("\n--- Testing lookup function ---")
conn = sqlite3.connect(db_file)
c = conn.cursor()


def get_wiktionary_link(word, lang_code, pos=None):
    if pos:
        c.execute(
            "SELECT url FROM entries WHERE word=? AND lang_code=? AND pos=?",
            (word, lang_code, pos),
        )
    else:
        c.execute(
            "SELECT url FROM entries WHERE word=? AND lang_code=?", (word, lang_code)
        )
    result = c.fetchone()
    return result[0] if result else None


test_cases = [
    ("hello", "en", None),
    ("run", "en", "verb"),
    ("run", "en", "noun"),
    ("run", "en", "adj"),
]

for word, lang_code, pos in test_cases:
    url = get_wiktionary_link(word, lang_code, pos)
    label = f"{word} ({lang_code}, pos={pos})"
    if url:
        print(f"✓ {label}: {url}")
    else:
        print(f"✗ {label}: Not found")

conn.close()
