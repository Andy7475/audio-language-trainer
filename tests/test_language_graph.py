import pytest
from src.language_graph import (
    get_sentences_from_dialogue,
    get_directed_graph_from_sentence,
)


@pytest.fixture
def input_dialogue():
    return [
        ("personA", "Hello, can I have a coffee? Can I also have a cake?"),
        ("personB", "Yes, of course. What size would you like?"),
        ("personA", "Medium, please."),
        ("personB", "OK, Goodbye."),
    ]


def test_get_sentences_from_dialogue(input_dialogue):

    result = get_sentences_from_dialogue(input_dialogue)
    first_sentence = result[0]
    comma_first_sentence = first_sentence[1]  # hello (,) <---
    assert len(result) == 6, f"Expected 6 sentences, but got {len(result)}"
    assert len(first_sentence) == 8  # hello , can i have a coffee ? <- 8 tokens
    assert comma_first_sentence["part_of_speech"] == "PUNCT"


@pytest.fixture
def sample_sentence(input_dialogue):
    sentences = get_sentences_from_dialogue(input_dialogue)
    return sentences[0]  # Return the first (and only) sentence


def test_get_directed_graph_from_sentence(sample_sentence):
    G = get_directed_graph_from_sentence(sample_sentence)

    # Check nodes
    expected_nodes = ["hello", "can", "i", "have", "a", "coffee", "?", "<EOS>"]
    assert set(G.nodes()) == set(
        expected_nodes
    ), f"Expected nodes {expected_nodes}, but got {list(G.nodes())}"

    # Check edges
    expected_edges = [
        ("hello", "can"),
        ("can", "i"),
        ("i", "have"),
        ("have", "a"),
        ("a", "coffee"),
        ("coffee", "?"),
        ("?", "<EOS>"),
    ]
    assert set(G.edges()) == set(
        expected_edges
    ), f"Expected edges {expected_edges}, but got {list(G.edges())}"

    # Check POS tags
    assert G.nodes["can"]["pos"] == "AUX", "Expected POS tag 'AUX' for 'can'"
    assert G.nodes["coffee"]["pos"] == "NOUN", "Expected POS tag 'NOUN' for 'coffee'"
    assert G.nodes["?"]["pos"] == "PUNCT", "Expected POS tag 'PUNCT' for '?'"
    assert G.nodes["<EOS>"]["pos"] == "EOS", "Expected POS tag 'EOS' for '<EOS>'"
