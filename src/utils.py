import json
import os
import re
from collections import defaultdict

from anthropic import AnthropicVertex
from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.config_loader import config

load_dotenv()  # so we can use environment variables for various global settings

PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")


def convert_defaultdict(d):
    if isinstance(d, defaultdict):
        d = {k: convert_defaultdict(v) for k, v in d.items()}
    return d


def save_defaultdict(d, filepath):
    normal_dict = convert_defaultdict(d)
    save_json(normal_dict, filepath)


def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def save_json(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=2)
    print(f"Data saved to {file_path}")


def anthropic_generate(prompt: str, max_tokens: int = 1024, model: str = None) -> str:
    """given a prompt generates an LLM response. The default model is specified in the config file.
    Most likely the largest Anthropic model. The region paramater in the config will have to match where that model
    is available"""
    client = AnthropicVertex(region=config.ANTHROPIC_REGION, project_id=PROJECT_ID)

    if model is None:
        model = config.ANTHROPIC_MODEL_NAME
    message = client.messages.create(
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
    )
    response_json = message.model_dump_json(indent=2)

    response = json.loads(response_json)
    return response["content"][0]["text"]


def extract_json_from_llm_response(response):
    """
    Extract JSON from an LLM response.

    :param response: String containing the LLM's response
    :return: Extracted JSON as a Python object, or None if no valid JSON is found
    """
    # Try to find JSON-like structure in the response
    json_pattern = (
        r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}))*\}"
    )
    json_match = re.search(json_pattern, response)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("Found JSON-like structure, but it's not valid JSON")
            return None
    else:
        print("No JSON-like structure found in the response")
        return None


from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def create_pdf_booklet(story_data_dict, output_filename="pdf_booklet.pdf"):
    # Register a Unicode font that supports Serbian characters

    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSans.ttf")
    pdfmetrics.registerFont(TTFont("DejaVuSans", font_path))

    doc = SimpleDocTemplate(output_filename, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    title_style.alignment = 1  # Center alignment
    title_style.fontName = "DejaVuSans"
    subtitle_style = styles["Heading2"]
    subtitle_style.alignment = 1  # Center alignment
    subtitle_style.fontName = "DejaVuSans"

    # Create a custom style for table cells
    cell_style = ParagraphStyle(
        "CellStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=12,
        alignment=TA_LEFT,
        fontName="DejaVuSans",
    )

    elements.append(Paragraph("Comprehensive Story Translation", title_style))
    elements.append(Spacer(1, 12))

    for story_part, data in story_data_dict.items():
        # Add story part title
        elements.append(Paragraph(story_part.capitalize(), subtitle_style))
        elements.append(Spacer(1, 12))

        # Process translated phrases
        if "translated_phrase_list" in data:
            elements.append(Paragraph("Translated Phrases", styles["Heading3"]))
            phrases_data = [["English", "Target Language"]]  # Table header
            for eng, swe in data["translated_phrase_list"]:
                phrases_data.append(
                    [Paragraph(eng, cell_style), Paragraph(swe, cell_style)]
                )

            phrases_table = Table(phrases_data, colWidths=[250, 250])
            phrases_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans"),
                        ("FONTSIZE", (0, 0), (-1, 0), 14),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                        ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
                        ("FONTSIZE", (0, 1), (-1, -1), 10),
                        ("TOPPADDING", (0, 1), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            elements.append(phrases_table)
            elements.append(Spacer(1, 12))

        # Process dialogue
        if "dialogue" in data and "translated_dialogue" in data:
            elements.append(Paragraph("Dialogue", styles["Heading3"]))
            dialogue_data = [["English", "Target Language"]]  # Table header
            for eng, swe in zip(data["dialogue"], data["translated_dialogue"]):
                dialogue_data.append(
                    [
                        Paragraph(f"{eng['text']}", cell_style),
                        Paragraph(f"{swe['text']}", cell_style),
                    ]
                )

            dialogue_table = Table(dialogue_data, colWidths=[250, 250])
            dialogue_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans"),
                        ("FONTSIZE", (0, 0), (-1, 0), 14),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                        ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
                        ("FONTSIZE", (0, 1), (-1, -1), 10),
                        ("TOPPADDING", (0, 1), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            elements.append(dialogue_table)
            elements.append(Spacer(1, 24))  # Add extra space between story parts

    doc.build(elements)
