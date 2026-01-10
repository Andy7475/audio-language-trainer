import gzip
import json
import sqlite3
from tqdm import tqdm
from urllib.parse import quote

# Configuration
input_file = "../outputs/wiktionary_dump/raw-wiktextract-data.jsonl.gz"
db_file = "wiktionary.db"
batch_size = 500_000  # Insert in batches for better performance

print("Creating database...")
# Create database and table
conn = sqlite3.connect(db_file)
c = conn.cursor()

# Drop table if exists (for clean rebuilds)
c.execute('DROP TABLE IF EXISTS entries')

# Create table with composite primary key
c.execute('''CREATE TABLE entries 
             (word TEXT NOT NULL,
              lang_code TEXT NOT NULL,
              url TEXT NOT NULL,
              PRIMARY KEY (word, lang_code))''')

print("Processing Wiktionary dump...")
batch = []
total_entries = 0

with gzip.open(input_file, 'rt', encoding='utf-8') as f:
    for i, line in enumerate(tqdm(f, desc="Processing entries")):
        try:
            item = json.loads(line)
            word = item.get("word")
            lang_code = item.get("lang_code")
            lang = item.get("lang")
            
            # Skip if missing required fields
            if not word or not lang_code or not lang:
                continue
            
            # Build URL
            url = f"https://en.wiktionary.org/wiki/{word}#{lang}"
            safe_url = url.replace(" ", "_")
            safe_url = quote(safe_url, safe=":/#")
            
            # Add to batch
            batch.append((word, lang_code, safe_url))
            total_entries += 1
            
            # Insert batch when it reaches batch_size
            if len(batch) >= batch_size:
                c.executemany('INSERT OR REPLACE INTO entries VALUES (?,?,?)', batch)
                conn.commit()
                batch = []
                
        except json.JSONDecodeError as e:
            print(f"Error parsing line {i}: {e}")
        except Exception as e:
            print(f"Error on line {i}: {e}")

# Insert remaining entries
if batch:
    c.executemany('INSERT OR REPLACE INTO entries VALUES (?,?,?)', batch)
    conn.commit()

print(f"\nCreating index...")
c.execute('CREATE INDEX IF NOT EXISTS idx_lookup ON entries(word, lang_code)')
conn.commit()

# Get statistics
c.execute('SELECT COUNT(*) FROM entries')
count = c.fetchone()[0]

c.execute('SELECT COUNT(DISTINCT lang_code) FROM entries')
lang_count = c.fetchone()[0]

conn.close()

print(f"\n✓ Database created successfully!")
print(f"  File: {db_file}")
print(f"  Total entries: {count:,}")
print(f"  Languages: {lang_count}")
print(f"  Processed lines: {i+1:,}")

# Test the lookup function
print("\n--- Testing lookup function ---")
conn = sqlite3.connect(db_file)
c = conn.cursor()

def get_wiktionary_link(word, lang_code):
    c.execute('SELECT url FROM entries WHERE word=? AND lang_code=?', (word, lang_code))
    result = c.fetchone()
    return result[0] if result else None

# Test with a few examples
test_cases = [
    ("hello", "en"),
    ("g'day", "en"),
    ("you're", "en")
]

for word, lang_code in test_cases:
    url = get_wiktionary_link(word, lang_code)
    if url:
        print(f"✓ {word} ({lang_code}): {url}")
    else:
        print(f"✗ {word} ({lang_code}): Not found")

conn.close()