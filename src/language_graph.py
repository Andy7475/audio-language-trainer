import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import ipycytoscape
import ipywidgets as widgets
import networkx as nx
import spacy


class WordNode:
    def __init__(self, text, pos):
        self.text = text
        self.pos = pos
        self.times_used = 0

    def __hash__(self):
        return hash((self.text, self.pos))

    def __eq__(self, other):
        if isinstance(other, WordNode):
            return self.text == other.text and self.pos == other.pos
        return False

    def __repr__(self):
        return f"({self.text}, {self.pos})"


class EdgeUsage:
    def __init__(self):
        self.times_used = 0


def get_sentences_from_dialogue(dialogue: List[Dict[str, str]]) -> List[List[WordNode]]:
    """Splits up dialogue into sentences then splits those up into tokens using spacy.
    Returns a list of sentences, where each sentence is a list of WordNodes."""

    nlp = spacy.load("en_core_web_md")
    sentences = []

    for utterance in dialogue:
        doc = nlp(utterance["text"])
        for sent in doc.sents:
            sentence = []
            for token in sent:
                word_node = WordNode(token.text.lower(), token.pos_)
                sentence.append(word_node)
            sentences.append(sentence)

    return sentences


def get_directed_graph_from_sentence(sentence: List[WordNode]) -> nx.DiGraph:
    G = nx.DiGraph()
    previous_node = None

    for word_node in sentence:
        # Skip punctuation except question mark
        if word_node.pos == "PUNCT" and word_node.text != "?":
            continue

        # Add node if it doesn't exist
        if not G.has_node(word_node):
            G.add_node(word_node)

        # Add edge from previous word to current word
        if previous_node is not None:
            G.add_edge(previous_node, word_node)

        previous_node = word_node

    # Add <EOS> node at the end
    eos_node = WordNode("<EOS>", "EOS")
    G.add_node(eos_node)
    if previous_node is not None:
        G.add_edge(previous_node, eos_node)

    return G


def generate_phrases_strategically(
    graph: nx.DiGraph, max_phrases: int = 1000, max_phrase_length: int = 10
) -> List[Tuple[WordNode, ...]]:
    def bfs_explore(start_node):
        phrases = []
        for path in nx.bfs_edges(graph, start_node, depth_limit=max_phrase_length - 1):
            current_path = nx.shortest_path(graph, start_node, path[1])
            if len(current_path) > 1:
                phrases.append(tuple(current_path))
            if len(phrases) >= max_phrases:
                break
        return phrases

    all_phrases = []
    verb_nodes = [node for node in graph.nodes() if node.pos == "VERB"]

    # Start exploration from each verb
    for verb in verb_nodes:
        phrases = bfs_explore(verb)
        all_phrases.extend(phrases)
        if len(all_phrases) >= max_phrases:
            break

    # If we haven't reached max_phrases, explore from other types of nodes
    if len(all_phrases) < max_phrases:
        other_nodes = [node for node in graph.nodes() if node.pos != "VERB"]
        random.shuffle(other_nodes)
        for node in other_nodes:
            phrases = bfs_explore(node)
            all_phrases.extend(phrases)
            if len(all_phrases) >= max_phrases:
                break

    # Sort phrases by length (shortest first)
    all_phrases.sort(key=len)

    return all_phrases[:max_phrases]


def is_valid_phrase(phrase: Tuple[WordNode, ...]) -> bool:
    # Check if the phrase contains at least one verb
    # has_verb = any(node.pos == "VERB" for node in phrase)
    # Check if the phrase starts and ends with valid nodes
    valid_start = is_valid_start(phrase[0])
    valid_end = is_valid_end(phrase[-1])
    return valid_start and valid_end


def generate_all_phrases(
    graph: nx.DiGraph, max_phrases: int = 1000
) -> Set[Tuple[WordNode, ...]]:
    all_phrases = generate_phrases_strategically(
        graph, max_phrases * 2
    )  # Generate more phrases than needed
    valid_phrases = set(phrase for phrase in all_phrases if is_valid_phrase(phrase))
    return set(list(valid_phrases)[:max_phrases])  # Return at most max_phrases


def generate_phrases(graph: nx.DiGraph) -> List[List[WordNode]]:
    all_paths = generate_all_phrases(graph)
    phrases = []
    edge_usage = {(u, v): EdgeUsage() for u, v in graph.edges()}

    while all_paths:
        path = weighted_random_path(graph, all_paths, edge_usage)
        phrases.append(list(path))
        all_paths.remove(path)
        update_usage(path, edge_usage)

    return phrases


def weighted_random_path(
    graph: nx.DiGraph,
    paths: Set[Tuple[WordNode, ...]],
    edge_usage: Dict[Tuple[WordNode, WordNode], EdgeUsage],
) -> Tuple[WordNode, ...]:
    weights = []
    for path in paths:
        path_weight = calculate_path_weight(graph, path, edge_usage)
        weights.append(path_weight)

    return random.choices(list(paths), weights=weights)[0]


def calculate_path_weight(
    graph: nx.DiGraph,
    path: Tuple[WordNode, ...],
    edge_usage: Dict[Tuple[WordNode, WordNode], EdgeUsage],
) -> float:
    weight = 1.0
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        edge_weight = 1 / (edge_usage[edge].times_used + 1)
        node_weight = 1 / (path[i].times_used + 1)
        weight *= edge_weight * node_weight
    return weight


def update_usage(
    path: Tuple[WordNode, ...], edge_usage: Dict[Tuple[WordNode, WordNode], EdgeUsage]
):
    for node in path:
        node.times_used += 1
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        edge_usage[edge].times_used += 1


def is_valid_start(node: WordNode) -> bool:
    return node.pos in {"PRON", "PROPN", "DET", "ADJ", "NOUN", "INTJ"}


def is_valid_end(node: WordNode) -> bool:
    return node.pos in {"VERB", "NOUN", "EOS", "PUNCT"}


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


class POSNode(ipycytoscape.Node):
    def __init__(self, word_node: WordNode):
        super().__init__()
        self.data["id"] = str(word_node.text)
        self.data["label"] = f"{word_node.text} ({word_node.pos})"
        self.classes = word_node.pos


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
        node_mapping: Dict[WordNode, POSNode] = {}
        for word_node in G.nodes():
            pos_node = POSNode(word_node)
            G_pos.add_node(pos_node)
            node_mapping[word_node] = pos_node

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


def enrich_graph(G: nx.DiGraph, num_enrichments: int = 5):
    def select_node_to_enrich(pos: str):
        pos_nodes = [n for n in G.nodes() if n.pos == pos]
        if not pos_nodes:
            return None
        weights = [1 / (G.degree(n) + 1) for n in pos_nodes]
        return random.choices(pos_nodes, weights=weights)[0]

    def find_connection_candidate(target_pos: str, exclude_node: WordNode):
        candidates = [n for n in G.nodes() if n.pos == target_pos and n != exclude_node]
        if not candidates:
            return None
        weights = [1 / (G.degree(n) + 1) for n in candidates]
        return random.choices(candidates, weights=weights)[0]

    # Define priority connections for language learning
    priority_connections = [
        ("PRON", "VERB"),  # e.g., "I have", "they went"
        ("DET", "NOUN"),  # e.g., "the book", "a car"
        ("ADJ", "NOUN"),  # e.g., "big house", "red apple"
        ("ADV", "VERB"),  # e.g., "quickly ran", "slowly walked"
        ("AUX", "VERB"),  # e.g., "is running", "have eaten"
        ("VERB", "NOUN"),  # e.g., "eat apple", "drive car"
        ("PRON", "AUX"),  # e.g., "I am", "they have"
        ("VERB", "ADP"),  # e.g., "look at", "think about"
        ("ADP", "NOUN"),  # e.g., "in house", "on table"
    ]

    # Build a dictionary of existing POS connections and their frequencies
    pos_connections = defaultdict(lambda: defaultdict(int))
    for source, target in G.edges():
        source_pos = source.pos
        target_pos = target.pos
        pos_connections[source_pos][target_pos] += 1

    for _ in range(num_enrichments):
        # Prioritize creating connections from the priority list
        if random.random() < 0.5:  # 70% chance to use priority connections
            random.shuffle(priority_connections)
            for source_pos, target_pos in priority_connections:
                source_node = select_node_to_enrich(source_pos)
                if source_node is None:
                    continue

                target_node = find_connection_candidate(target_pos, source_node)
                if target_node is None:
                    continue

                if not G.has_edge(source_node, target_node):
                    G.add_edge(source_node, target_node)
                    print(
                        f"Added priority edge from {source_node.text} ({source_pos}) to {target_node.text} ({target_pos})"
                    )
                    break
            else:
                continue  # If no priority connection was added, continue to the next iteration

        else:
            # Fallback to enriching pronouns -> to either verb or aux
            pos_to_enrich = "PRON"
            node_to_enrich = select_node_to_enrich(pos_to_enrich)
            if node_to_enrich is None:
                continue

            target_pos_candidates = pos_connections[pos_to_enrich]
            target_pos_candidates = [
                pos for pos in target_pos_candidates if pos in ["VERB", "AUX"]
            ]
            if not target_pos_candidates:
                continue

            target_pos = random.choices(target_pos_candidates)[0]

            new_connection = find_connection_candidate(target_pos, node_to_enrich)
            if new_connection is None:
                continue

            if not G.has_edge(node_to_enrich, new_connection):
                G.add_edge(node_to_enrich, new_connection)
                print(
                    f"Added edge from {node_to_enrich.text} ({pos_to_enrich}) to {new_connection.text} ({target_pos})"
                )

    return G


def generate_phrase_progression(
    original_sentences: List[List[WordNode]],
    generated_phrases: List[List[WordNode]],
    max_edge_use: int = 6,
    num_short_phrases: int = 10,
    num_medium_phrases: int = 30,
) -> List[str]:
    # Initialize edge usage counter
    edge_usage = defaultdict(int)

    # Function to check if a phrase contains a verb
    def contains_verb(phrase: List[WordNode]) -> bool:
        return any(word.pos == "VERB" for word in phrase)

    # Function to calculate phrase weight based on length
    def phrase_weight(phrase: List[WordNode]) -> float:
        return 1 / (2 * len(phrase))

    # Function to convert a list of WordNodes to a string
    def phrase_to_string(phrase: List[WordNode]) -> str:
        return " ".join(word.text for word in phrase if word.pos != "EOS")

    # Filter and organize phrases
    short_phrases = [p for p in generated_phrases if len(p) == 2 and contains_verb(p)]
    medium_phrases = [
        p
        for p in generated_phrases
        if 2 < len(p) < len(max(original_sentences, key=len))
    ]

    progression = []
    used_words = set()

    # Helper function to add a phrase to the progression
    def add_phrase(phrase: List[WordNode]):
        for i in range(len(phrase) - 1):
            edge = (phrase[i], phrase[i + 1])
            edge_usage[edge] += 1
        progression.append(phrase_to_string(phrase))
        used_words.update(word.text for word in phrase)

    # Start with short phrases
    while short_phrases and len(progression) < num_short_phrases:
        phrase = random.choices(
            short_phrases, weights=[phrase_weight(p) for p in short_phrases]
        )[0]
        add_phrase(phrase)
        short_phrases.remove(phrase)

    # Add medium phrases
    while medium_phrases and len(progression) < num_medium_phrases:
        valid_phrases = [
            p
            for p in medium_phrases
            if all(
                edge_usage[(p[i], p[i + 1])] < max_edge_use for i in range(len(p) - 1)
            )
            and any(word.text not in used_words for word in p)
        ]
        if not valid_phrases:
            break
        phrase = random.choices(
            valid_phrases, weights=[phrase_weight(p) for p in valid_phrases]
        )[0]
        add_phrase(phrase)
        medium_phrases.remove(phrase)

    progression.sort(key=len)
    # Add original sentences as strings at the end
    progression.extend(phrase_to_string(sentence) for sentence in original_sentences)

    return progression
