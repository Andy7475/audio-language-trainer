
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple


from tqdm import tqdm

from src.connections.gcloud_auth import get_nlp_client
from google.cloud import language_v1
from typing import List, Tuple, Dict, Set



def analyze_text_syntax(
    text: str, 
    language_code: str = "en"
) -> language_v1.AnalyzeSyntaxResponse:
    """
    Analyze text syntax using Google Natural Language API.
    
    Args:
        text: Text to analyze
        language_code: BCP-47 language code (e.g., 'en', 'en-US')
    
    Returns:
        AnalyzeSyntaxResponse containing tokens with lemmas and POS tags
    """
    client = get_nlp_client()
    
    document = language_v1.Document(
        content=text,
        type_=language_v1.Document.Type.PLAIN_TEXT,
        language=language_code,
    )
    
    response = client.analyze_syntax(
        request={
            "document": document,
            "encoding_type": language_v1.EncodingType.UTF8,
        }
    )
    
    return response



def extract_lemmas_and_pos(
    english_phrases: List[str],
    language_code: str = "en"
) -> Set[Tuple[str, str]]:
    """
    Extract unique (lemma, POS) tuples from phrases using Google NLP API.
    
    Args:
        english_phrases: List of phrases to analyze
        language_code: BCP-47 language code (default: 'en')
    
    Returns:
        Set of (lemma, pos) tuples
    """
    vocab_set = set()
    
    for phrase in english_phrases:
        response = analyze_text_syntax(phrase, language_code)
        
        for token in response.tokens:
            # Get POS tag name (e.g., 'VERB', 'NOUN', 'ADJ')
            pos_tag = language_v1.PartOfSpeech.Tag(token.part_of_speech.tag).name
            
            lemma = token.lemma.lower()
            
            vocab_set.add((lemma, pos_tag))
    
    return vocab_set


def get_verbs_from_lemmas_and_pos(
    lemmas_and_pos: Set[Tuple[str, str]])-> list[str]:
    """Extract verbs from a set of (word, pos) tuples."""
    verbs = [word for word, pos in lemmas_and_pos if pos in ["VERB", "AUX"]]
    return verbs

def get_vocab_from_lemmas_and_pos(
    lemmas_and_pos: Set[Tuple[str, str]])-> list[str]:
    """Extract vocab (non-verbs) from a set of (word, pos) tuples."""
    vocab = [word for word, pos in lemmas_and_pos if pos not in ["VERB", "AUX", "PUNCT"]]
    return vocab

def get_tokens_from_lemmas_and_pos(
    lemmas_and_pos: Set[Tuple[str, str]])-> list[str]:
    """Extract tokens from a set of (word, pos) tuples."""
    tokens = [word for word, pos in lemmas_and_pos if pos not in ["PUNCT"]]
    return tokens
