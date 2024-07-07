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
        arrowhead_length=0.1,
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


from py2cytoscape.data.cynetwork import CyNetwork
from py2cytoscape.data.cyrest_client import CyRestClient
import networkx as nx


def custom_networkx_to_cytoscape(G):
    cyjs = {"data": {}, "elements": {"nodes": [], "edges": []}}

    for node, data in G.nodes(data=True):
        cyjs["elements"]["nodes"].append(
            {
                "data": {
                    "id": str(node),
                    "name": str(node),
                    **{k: str(v) for k, v in data.items()},
                }
            }
        )

    for source, target, data in G.edges(data=True):
        cyjs["elements"]["edges"].append(
            {
                "data": {
                    "source": str(source),
                    "target": str(target),
                    **{k: str(v) for k, v in data.items()},
                }
            }
        )

    return cyjs


import ipycytoscape
import ipywidgets as widgets
import networkx as nx


class POSNode(ipycytoscape.Node):
    def __init__(self, name, pos):
        super().__init__()
        self.data["id"] = str(name)
        self.data["label"] = str(name) + " (" + str(pos) + ")"
        self.classes = pos


def create_interactive_sentence_graph(G: nx.DiGraph):
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
        "ADP": "#20B2AA",  # Light Sea Green
        "INTJ": "#FF6347",  # Tomato
        "PROPN": "#9370DB",  # Medium Purple
        "UNKNOWN": "#FFFFFF",  # White (for any unrecognized POS)
    }

    cytoscapeobj = ipycytoscape.CytoscapeWidget()

    def update_graph():
        # Create a new NetworkX graph with POSNode objects
        G_pos = nx.DiGraph()
        node_mapping = (
            {}
        )  # To keep track of original nodes and their corresponding POSNode objects
        for node, data in G.nodes(data=True):
            pos = data.get("pos", "UNKNOWN")
            pos_node = POSNode(node, pos)
            G_pos.add_node(pos_node)
            node_mapping[node] = pos_node

        for source, target in G.edges():
            G_pos.add_edge(node_mapping[source], node_mapping[target])

        cytoscapeobj.graph.clear()
        cytoscapeobj.graph.add_graph_from_networkx(G_pos, directed=True)

    update_graph()

    # Set up the visual style
    style = [
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "text-valign": "center",
                "text-halign": "center",
                "font-size": "12px",
                "width": "60px",
                "height": "60px",
            },
        },
        {
            "selector": "edge",
            "style": {
                "width": 3,
                "line-color": "#999",
                "target-arrow-color": "#999",
                "target-arrow-shape": "triangle",
                "curve-style": "bezier",
            },
        },
    ]

    # Add color styles for each POS
    for pos, color in pos_color_map.items():
        style.append({"selector": f"node.{pos}", "style": {"background-color": color}})

    cytoscapeobj.set_style(style)

    # Set the layout
    cytoscapeobj.set_layout(name="cose", nodeSpacing=80)

    # Create a button to regenerate the graph
    button = widgets.Button(description="Regenerate Graph")
    button.on_click(lambda b: update_graph())

    return widgets.VBox([button, cytoscapeobj])


import networkx as nx
import random
from collections import defaultdict


def enrich_graph(G: nx.DiGraph, num_enrichments: int = 5):
    def select_node_to_enrich(pos: str):
        pos_nodes = [n for n in G.nodes() if G.nodes[n]["pos"] == pos]
        if not pos_nodes:
            return None
        weights = [1 / (G.degree(n) + 1) for n in pos_nodes]
        return random.choices(pos_nodes, weights=weights)[0]

    def find_connection_candidate(target_pos: str, exclude_node: str):
        candidates = [
            n
            for n in G.nodes()
            if G.nodes[n]["pos"] == target_pos and n != exclude_node
        ]
        if not candidates:
            return None
        weights = [1 / (G.degree(n) + 1) for n in candidates]
        return random.choices(candidates, weights=weights)[0]

    # Build a dictionary of POS connections and their directions
    pos_connections = defaultdict(lambda: defaultdict(int))
    for source, target in G.edges():
        source_pos = G.nodes[source]["pos"]
        target_pos = G.nodes[target]["pos"]
        pos_connections[source_pos][target_pos] += 1

    for _ in range(num_enrichments):
        # Select a random POS to enrich
        pos_to_enrich = random.choice(
            list(set(nx.get_node_attributes(G, "pos").values()))
        )

        # Select a poorly connected node of that POS
        node_to_enrich = select_node_to_enrich(pos_to_enrich)
        if node_to_enrich is None:
            continue

        # Find a target POS using weighted random sampling
        target_pos_candidates = pos_connections[pos_to_enrich]
        if not target_pos_candidates:
            continue

        target_pos_list = list(target_pos_candidates.keys())
        target_pos_weights = list(target_pos_candidates.values())
        target_pos = random.choices(target_pos_list, weights=target_pos_weights, k=1)[0]

        # Find a poorly connected node of the target POS that isn't already connected
        new_connection = find_connection_candidate(target_pos, node_to_enrich)
        if new_connection is None:
            continue

        # Add the new edge, maintaining the original direction
        if G.has_edge(node_to_enrich, new_connection):
            continue

        G.add_edge(node_to_enrich, new_connection)
        print(
            f"Added edge from {node_to_enrich} ({pos_to_enrich}) to {new_connection} ({target_pos})"
        )

    return G


# Usage example:
# enriched_graph = enrich_graph(original_graph, num_enrichments=10)
