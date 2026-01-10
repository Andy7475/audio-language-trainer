from src.phrases.search import get_phrases_by_collection

COLLECTION = "WarmUp150"

phrases = get_phrases_by_collection(COLLECTION)
for phrase in phrases[:10]:
    print(f"{phrase.english} -> {phrase.translations.get('fr-FR', 'N/A')}")
