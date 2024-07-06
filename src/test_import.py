print("Starting import...")
import spacy

print("spaCy imported successfully")
print("Loading model...")
nlp = spacy.load("en_core_web_md")
print("Model loaded successfully")

print("Processing text...")
doc = nlp("This is a test sentence.")
print("Text processed successfully")

for token in doc:
    print(token.text, token.pos_)
print("Tokens printed successfully")
