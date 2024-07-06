from typing import Dict, List, Tuple

import colorcet as cc
import hvplot.networkx as hvnx
import networkx as nx
import pandas as pd
import spacy
import holoviews as hv
from holoviews import opts


def get_sentences_from_dialogue(
    dialogue: List[Tuple[str, str]]
) -> List[List[Dict[str, str]]]:
    """Splits up dialogue into sentences then splits those up into tokens using spacy. So we can
      iterate over sentences and words to build up a NetworkX graph later

      [[{'text': 'Hello', 'text_lower': 'hello', 'part_of_speech': 'INTJ'},
    {'text': ',', 'text_lower': ',', 'part_of_speech': 'PUNCT'},...]]"""

    nlp = spacy.load("en_core_web_md")
    sentences = []

    for _, utterance in dialogue:
        doc = nlp(utterance)
        for sent in doc.sents:
            sentence = []
            for token in sent:
                word_info = {
                    "text": token.text,
                    "text_lower": token.text.lower(),
                    "part_of_speech": token.pos_,
                }
                sentence.append(word_info)
            sentences.append(sentence)

    return sentences


def get_directed_graph_from_sentence(sentence: List[Dict[str, str]]) -> nx.DiGraph:
    G = nx.DiGraph()
    previous_word = None

    for word_info in sentence:
        current_word = word_info["text_lower"]
        pos = word_info["part_of_speech"]

        # Skip punctuation except question mark
        if pos == "PUNCT" and current_word != "?":
            continue

        # Add node if it doesn't exist
        if not G.has_node(current_word):
            G.add_node(current_word, pos=pos)

        # Add edge from previous word to current word
        if previous_word is not None:
            G.add_edge(previous_word, current_word)

        previous_word = current_word

    # Add <EOS> node at the end
    G.add_node("<EOS>", pos="EOS")
    if previous_word is not None:
        G.add_edge(previous_word, "<EOS>")

    return G


def get_merged_graph_from_sentences(
    sentences: List[List[Dict[str, str]]]
) -> nx.DiGraph:
    """
    Create a merged graph from multiple sentences.

    :param sentences: A list of sentences, where each sentence is a list of word dictionaries.
    :return: A single merged NetworkX DiGraph.
    """
    merged_graph = nx.DiGraph()

    for sentence in sentences:
        # Create a graph for the current sentence
        sentence_graph = get_directed_graph_from_sentence(sentence)

        # Merge the sentence graph into the main graph
        merged_graph = nx.compose(merged_graph, sentence_graph)

    return merged_graph


def plot_sentence_graph(G: nx.DiGraph):
    # Define a color map for POS tags
    pos_color_map = {
        "NOUN": "#FFA07A",  # Light Salmon
        "VERB": "#98FB98",  # Pale Green
        "ADJ": "#87CEFA",  # Light Sky Blue
        "ADV": "#DDA0DD",  # Plum
        "AUX": "#F0E68C",  # Khaki
        "PRON": "#FFB6C1",  # Light Pink
        "DET": "#E6E6FA",  # Lavender
        "PUNCT": "#D3D3D3",  # Light Gray
        "EOS": "#FFDAB9",  # Peach Puff
    }
    default_color = "#FFFFFF"  # White for unknown POS

    # Create a list of colors for nodes
    node_colors = [
        pos_color_map.get(G.nodes[node]["pos"], default_color) for node in G.nodes()
    ]

    # Create the layout
    pos = nx.spring_layout(G)

    # Create the graph plot
    plot = hvnx.draw(
        G,
        pos=pos,
        edge_color="black",
        node_color=node_colors,
        with_labels=False,  # We'll add labels separately
        arrows=True,
        node_size=1000,
        width=700,
        height=400,
    )

    # Add labels to the plot
    labels = hv.Labels(
        {(x, y): node for node, (x, y) in pos.items()}, ["x", "y"], "text"
    )
    plot = plot * labels.opts(text_font_size="8pt", text_color="black")

    # Customize the plot
    plot = plot.opts(
        opts.Graph(title="Sentence Graph", bgcolor="white", xaxis=None, yaxis=None),
        opts.Labels(text_font_size="8pt", text_color="black"),
    )

    # Create a color legend
    legend_items = [
        hv.Scatter([], label=pos).opts(color=color, size=10, tools=[])
        for pos, color in pos_color_map.items()
    ]
    legend = hv.Overlay(legend_items).opts(
        opts.Overlay(show_frame=False, legend_position="right", width=150)
    )

    # Combine the plot and legend
    combined_plot = (plot + legend).cols(2)

    return combined_plot


# def create_graph_from_text(self, input_text: list[tuple], custom_stop_words=[",", ";"]):
#     """From a list of tuples like [('personA', 'hello'), ('personB', 'bye')]"""
#     self.lemmas_pos = []
#     self.max_target_phrase_length = 0
#     self.input_phrases = input_text

#     utterances = []
#     for person, utterance in input_text:
#         doc = self.nlp(utterance.lower())
#         utterance = tuple()
#         for sent in doc.sents:
#             sentence_pos = []
#             phrase = tuple()

#             for token in sent:
#                 lemma = token.lemma_
#                 pos = token.pos_
#                 text = token.text
#                 if lemma in custom_stop_words:
#                     continue

#                 if pos == "PUNCT":
#                     if lemma in self.punct_to_EOS:
#                         word = "<EOS>"
#                     else:
#                         word = text
#                 elif pos in ["PRON"]:  # so I and me get written separately
#                     word = text
#                 else:
#                     word = text  # used to be lemmas but removing this now as too hard to correct sentence fragments with lots of lemmas
#                 if word != "<EOS>":  # don't have it as part of target phrases
#                     phrase = phrase + (word,)
#                     utterance = utterance + (word,)
#                 sentence_pos.append({word: {"pos": pos}})

#             self.target_phrases |= {phrase}
#             # phrase_length = len([word for word in phrase if word not in self.EOS_tokens]) #don't include EOS when calculating phrase length
#             self.max_target_phrase_length = max(
#                 len(phrase), self.max_target_phrase_length
#             )

#             self.lemmas_pos.append(sentence_pos)
#         self.target_phrases |= {utterance}
#     for lemmas_pos_sentence in self.lemmas_pos:
#         if len(lemmas_pos_sentence) == 1:  # single word utterance
#             curr_lemma = list(lemmas_pos_sentence[0].keys())[0]
#             curr_pos = lemmas_pos_sentence[0][curr_lemma]["pos"]
#             if curr_pos == "PUNCT":
#                 continue
#             self.add_word(curr_lemma, curr_pos)
#         else:
#             for i in range(len(lemmas_pos_sentence) - 1):
#                 curr_lemma = list(lemmas_pos_sentence[i].keys())[0]
#                 curr_pos = lemmas_pos_sentence[i][curr_lemma]["pos"]
#                 if curr_pos == "PUNCT":
#                     # print("punct lemma ", curr_lemma)
#                     continue

#                 next_lemma = list(lemmas_pos_sentence[i + 1].keys())[0]
#                 next_pos = lemmas_pos_sentence[i + 1][next_lemma]["pos"]
#                 if next_pos == "PUNCT":  # end the sentence
#                     # print("punct lemma ", next_lemma)
#                     if next_lemma in self.punct_to_EOS:
#                         next_lemma = "<EOS>"
#                     elif next_lemma in ["?"]:  # punctuation to keep
#                         next_lemma = next_lemma
#                     else:
#                         continue

#                 self.add_word(curr_lemma, curr_pos)
#                 self.add_word(next_lemma, next_pos)
#                 self.add_word_sequence(curr_lemma, next_lemma)
#     self.target_phrases_not_yet_heard = self.target_phrases.copy()
