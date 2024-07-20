import pytest
from src.language_graph import (
    get_sentences_from_dialogue,
    get_directed_graph_from_sentence,
    WordNode,
)

# TO DO - change dialogue format to JSON
# [{'speaker': 'Alex',
#   'text': 'Hey Sam, are you tired today? We can watch a programme about hiking tonight.'},
#  {'speaker': 'Sam',
#   'text': "Thanks, but I'm so sorry. I have to hurry to the university this evening."},
#  {'speaker': 'Alex',
#   'text': "That's okay. If we had watched it tonight, we would have become better hikers."},
#  {'speaker': 'Sam',
#   'text': "I know. Although I'm busy today, I hope we can watch it next week."},
#  {'speaker': 'Alex',
#   'text': "Sounds good! Why don't we watch it on Friday at 8 o'clock?"},
#  {'speaker': 'Sam',
#   'text': "Yes, that works for me. I'll be here with some snacks."}]


@pytest.fixture
def input_dialogue():
    return [
        {
            "speaker": "personA",
            "text": "Hello, can I have a coffee? Can I also have a cake?",
        },
        {"speaker": "personB", "text": "Yes, of course. What size would you like?"},
        {"speaker": "personA", "text": "Medium, please."},
        {"speaker": "personB", "text": "OK, Goodbye."},
    ]


def test_get_sentences_from_dialogue(input_dialogue):

    result = get_sentences_from_dialogue(input_dialogue)
    first_sentence = result[0]
    comma_first_sentence = first_sentence[1]  # hello (,) <---
    assert len(result) == 6, f"Expected 6 sentences, but got {len(result)}"
    assert len(first_sentence) == 8  # hello , can i have a coffee ? <- 8 tokens
    assert comma_first_sentence.pos == "PUNCT"


@pytest.fixture
def sample_sentence(input_dialogue):
    sentences = get_sentences_from_dialogue(input_dialogue)
    return sentences[0]  # Return the first (and only) sentence


def test_get_directed_graph_from_sentence(sample_sentence):
    G = get_directed_graph_from_sentence(sample_sentence)

    # Check nodes
    expected_nodes = [
        WordNode("hello", "INTJ"),
        WordNode("can", "AUX"),
        WordNode("i", "PRON"),
        WordNode("have", "VERB"),
        WordNode("a", "DET"),
        WordNode("coffee", "NOUN"),
        WordNode("?", "PUNCT"),
        WordNode("<EOS>", "EOS"),
    ]
    assert set(G.nodes()) == set(
        expected_nodes
    ), f"Expected nodes {expected_nodes}, but got {list(G.nodes())}"

    # Check edges
    expected_edges = set(
        {
            (WordNode("hello", "INTJ"), WordNode("can", "AUX")),
            (WordNode("?", "PUNCT"), WordNode("<EOS>", "EOS")),
        }
    )
    assert (
        len(expected_edges.intersection(set(G.edges()))) == 2
    )  # both edges should be in the graph

    # Check POS tags
    for node in G.nodes():
        if node.text == "hello":
            assert node.pos == "INTJ"
        if node.text == "<EOS>":
            assert node.pos == "EOS"
