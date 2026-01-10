import sqlite3
from sqlite3 import Connection
from pathlib import Path
_wiktionary_db = None
DB_FILE = Path(__file__).parents[1] / "wiktionary" / "wiktionary.db"
#print(DB_FILE)
def get_wiktionary_db()->Connection:
    global _wiktionary_db
    if _wiktionary_db is None:
        _wiktionary_db = sqlite3.connect(DB_FILE)
    return _wiktionary_db

