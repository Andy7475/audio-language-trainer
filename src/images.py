import copy
import io
import os
import time
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
import requests
import vertexai
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
from vertexai.preview.vision_models import ImageGenerationModel

from src.config_loader import config
from src.utils import anthropic_generate, clean_filename, ok_to_query_api

load_dotenv()  # so we can use environment variables for various global settings
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")


def create_image_generation_prompt_for_story_part(
    story_part: Union[Dict, List[Dict]], anthropic_model: str = None
) -> str:
    """
    Create an image generation prompt from a story part or list of story parts.

    Args:
        story_part: Either a dictionary containing a 'dialogue' key, or a list of such dictionaries.
            Each dialogue entry should be a list of speaker/text pairs.
        anthropic_model: Optional model name for Claude

    Returns:
        str: A prompt suitable for image generation
    """
    # Convert single part to list for consistent processing
    story_parts = story_part if isinstance(story_part, list) else [story_part]

    # Extract all dialogue text, removing speaker information
    all_dialogue = []
    for part in story_parts:
        if "dialogue" in part:
            all_dialogue.extend([utterance["text"] for utterance in part["dialogue"]])

    dialogue_text = " ".join(all_dialogue)

    llm_prompt = f"""
    Analyze this dialogue and create a first-person perspective image prompt to visualize the scene:

    {dialogue_text}
    
    Create a detailed prompt for generating an image that captures the location and atmosphere of this scene.
    
    Requirements:
    1. Use first-person perspective as if the viewer is participating in the scene
    2. Focus on the environment, setting, and background elements
    3. Do not include the main characters (Alex and Sam) in the description
    4. You may include background people for atmosphere, but keep them generic
    5. Include time of day, weather, and atmospheric details if mentioned or implied
    6. Capture the emotional tone of the conversation in the environment
    7. Limit output to 1-2 sentences focused on the visual elements only
    
    Example output format:
    "View of a bustling city square from a cafe terrace, morning light streaming through trees, people walking past market stalls"
    """

    # Watercolor style - light, airy, organic
    # base_style = "watercolor illustration style, fluid brush strokes, soft color transitions, vibrant pigments, organic textures, white paper showing through, loose expressive style, dynamic washes"

    # Oil painting style - rich, textured, detailed
    # base_style = "oil painting style, rich impasto textures, visible brushstrokes, saturated colors, warm undertones, glazed layers, painterly details"

    # Children's book illustration - whimsical, friendly
    # base_style = "children's book illustration style, clean linework, bright cheerful colors, decorative details, playful shapes, gentle shading, hand-drawn feel"

    # Studio Ghibli inspired - atmospheric, detailed
    base_style = "Studio Ghibli art style, soft atmospheric colors, detailed backgrounds, gentle gradients, natural elements, dreamy lighting, painted textures"

    # Modern animated style - clean, bold
    # base_style = "modern animation art style, clean vector-like shapes, bold color palette, subtle textures, smooth gradients, graphic design elements, minimalist details"

    # Traditional gouache style - flat, bold
    # base_style = "gouache painting style, matte finish, bold flat colors, painterly textures, crisp edges, vintage poster feel, decorative elements"

    # Use anthropic_generate to get the response
    image_prompt = anthropic_generate(llm_prompt, model=anthropic_model)
    image_prompt = image_prompt.strip('".')

    return image_prompt + f" in the style of {base_style}"


def create_image_generation_prompt(phrase, anthropic_model: str = None):
    """
    Create a specific image generation prompt based on a language learning phrase.

    :param phrase: The language learning phrase to visualize
    :return: A specific prompt for image generation
    """

    llm_prompt = f"""
    Given the following phrase for language learners: "{phrase}"
    
    Create a specific, detailed prompt for generating an image that will help learners remember this phrase.
    Focus on key nouns, verbs, or concepts that can be visually represented.
    The image should be memorable and directly related to the meaning of the phrase.
    
    Your prompt should:
    1. Ensure you consider every element from the phrase to visualize.
    2. Suggest a simple scene or composition that incorporates these elements. You can use your imagination to make it more memorable
    3. Include any relevant emotions, or atmosphere that would enhance memory retention.
    4. Limit your output to 1 - 2 sentences. Do not add details of the image style, this will be added later.
    
    Provide only the image generation prompt, without any explanations or additional text.

    Example phrase: "The bride watched the sunset from the balcony"
    Example Output: "A bride on a balcony, looking at sunset over the horizon, tropical island, villa"
    """

    base_style = "a children's book illustration, Axel Scheffler style, thick brushstrokes, colored pencil texture, expressive characters, bold outlines, textured shading, pastel color palette"

    # Use the anthropic_generate function to get the LLM's response
    image_prompt = anthropic_generate(llm_prompt, model=anthropic_model)
    image_prompt.strip('".')

    return image_prompt + f" in the style of {base_style}"


def generate_image_deepai(
    prompt: str,
    width: Union[str, int] = "512",
    height: Union[str, int] = "512",
    model: Literal["standard", "hd"] = "hd",
    negative_prompt: Optional[str] = None,
) -> Image.Image:
    """
    Generate an image using DeepAI's text2img API and return it as a PIL Image object.

    Args:
        prompt (str): The text prompt to generate the image from
        width (Union[str, int]): Image width (128-1536, default 512)
        height (Union[str, int]): Image height (128-1536, default 512)
        model (str): Model version ("standard" or "hd")
        negative_prompt (Optional[str]): Text describing what to remove from the image

    Returns:
        PIL.Image.Image: The generated image as a PIL Image object

    Raises:
        Exception: If there's an error in image generation or processing
        EnvironmentError: If DEEPAI_API_KEY environment variable is not set
    """
    # Get API key from environment variable
    api_key = os.getenv("DEEPAI_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPAI_API_KEY environment variable not set")

    try:
        # Convert width and height to strings if they're integers
        width = str(width)
        height = str(height)

        # Prepare the API request data
        data = {
            "text": prompt,
            "width": width,
            "height": height,
            "image_generator_version": model,
        }

        # Add negative prompt if provided
        if negative_prompt:
            data["negative_prompt"] = negative_prompt

        # Make the API request
        response = requests.post(
            "https://api.deepai.org/api/text2img",
            data=data,
            headers={"api-key": api_key},
        )

        # Check if the request was successful
        response.raise_for_status()

        # Get the URL of the generated image from the response
        result = response.json()
        if "output_url" not in result:
            raise Exception(f"Unexpected API response: {result}")

        # Download the image from the URL
        image_response = requests.get(result["output_url"])
        image_response.raise_for_status()

        # Convert to PIL Image
        image = Image.open(BytesIO(image_response.content))

        return image

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error making request to DeepAI API: {str(e)}")
    except Exception as e:
        raise Exception(f"Error generating image with DeepAI: {str(e)}")


def generate_image_stability(
    prompt: str,
    negative_prompt: str = "",
    style_preset: Optional[
        Literal[
            "3d-model",
            "analog-film",
            "anime",
            "cinematic",
            "comic-book",
            "digital-art",
            "enhance",
            "fantasy-art",
            "isometric",
            "line-art",
            "low-poly",
            "modeling-compound",
            "neon-punk",
            "origami",
            "photographic",
            "pixel-art",
            "tile-texture",
        ]
    ] = None,
    endpoint=config.STABILITY_ENDPOINT,
) -> Optional[Image.Image]:
    """
    Generate an image using Stability AI's core model API.

    Args:
        prompt (str): Text description of the desired image
        negative_prompt (str): Text description of what to avoid in the image
        style_preset (str, optional): Style preset to use for image generation

    Returns:
        bytes: Generated image data, or None if generation fails

    Raises:
        ValueError: If API key is missing
        Warning: If content is filtered (NSFW)
        requests.RequestException: If the API request fails
    """
    if not STABILITY_API_KEY:
        raise ValueError("STABILITY_API_KEY environment variable not set")

    # Prepare headers
    headers = {"Accept": "image/*", "Authorization": f"Bearer {STABILITY_API_KEY}"}

    # Prepare form data
    files = {
        "prompt": (None, prompt),
    }

    # Add optional parameters if provided
    if negative_prompt:
        files["negative_prompt"] = (None, negative_prompt)

    if style_preset:
        files["style_preset"] = (None, style_preset)

    try:
        # Make the API request
        response = requests.post(
            endpoint,
            headers=headers,
            files=files,
        )

        # Check if request was successful
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.content}")
            return None

        # Check for content filtering
        finish_reason = response.headers.get("finish-reason")
        if finish_reason == "CONTENT_FILTERED":
            raise Warning("Generation failed NSFW classifier")

        # Return the raw image content
        return Image.open(io.BytesIO(response.content))

    except requests.RequestException as e:
        print(f"Request failed: {str(e)}")
        return None
    except Warning as w:
        print(f"Content filtered: {str(w)}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None


def generate_image_imagen(
    prompt: str,
    model: Literal[
        "imagen-3.0-fast-generate-001", "imagen-3.0-generate-001"
    ] = "imagen-3.0-generate-001",
) -> Optional[Image.Image]:
    """
    Generate an image using the Vertex AI Imagen model with retry logic.

    Args:
        prompt: The text prompt to generate the image from
        model: The Imagen model to use
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds between retries
        delay_multiplier: Factor to multiply delay by after each attempt

    Returns:
        Generated image bytes from the model

    Raises:
        Exception: If image generation fails after all retry attempts
    """
    vertexai.init(project=config.PROJECT_ID, location=config.VERTEX_REGION)
    generation_model = ImageGenerationModel.from_pretrained(model)

    try:
        # Generate the image
        images = generation_model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="1:1",
            # person_generation="allow_adult",
            # safety_filter_level="block_fewest",
        )

        if len(images.images) > 0:

            return Image.open(io.BytesIO(images.images[0]._image_bytes))
        else:
            print(f"No image generated using {model} with prompt: {prompt}")
            return None

    except Exception as e:
        print(f"Imagen generation failed with error {e}")
        return None


def resize_image(generated_image, height=500, width=500):

    # Get the image bytes directly
    image_data = generated_image._image_bytes

    # Convert the image to PIL Image for potential resizing
    image = Image.open(io.BytesIO(image_data))

    # Resize the image if it's not 500x500
    if image.size != (height, width):
        image = image.resize((height, width))

        # If we resized, convert the resized image back to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        image_data = img_byte_arr.getvalue()

    return image_data


def add_images_to_phrases(
    phrases: List[str],
    output_dir: str,
    image_format: str = "png",
    imagen_model: Literal[
        "imagen-3.0-fast-generate-001", "imagen-3.0-generate-001"
    ] = "imagen-3.0-generate-001",
    anthropic_model=config.ANTHROPIC_MODEL_NAME,
) -> Dict:
    """
    Process a list of phrases to create a dictionary with prompts and image paths.

    Args:
        phrases: List of English phrases
        output_dir: Directory where images will be saved
        generate_image_prompt: Function that takes a phrase and returns a prompt
        generate_image: Function that takes a prompt and returns image data
        image_format: Image file format (default: 'png')

    Returns:
        Dictionary containing phrases, prompts, and image paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize results dictionary
    results = {}

    for phrase in tqdm(phrases):
        # Create a clean filename from the phrase
        clean_name = clean_filename(phrase)

        # Generate image filename
        image_filename = f"{clean_name}.{image_format}"
        image_path = os.path.join(output_dir, image_filename)

        if os.path.exists(image_path):
            print(f"Warning: Image already exists for '{phrase}', skipping generation")
            results[clean_name] = {
                "phrase": phrase,
                "prompt": None,
                "image_path": image_path,
            }
            continue
        # Generate prompt for the phrase
        ok_to_query_api()
        prompt = create_image_generation_prompt(phrase, anthropic_model)

        # Generate and save the image
        try:
            ok_to_query_api()
            image = generate_image_imagen(prompt, model=imagen_model)
            if image is None:
                image = generate_image_deepai(prompt)

            if image is None:
                print("Both image generation attempts failed, skipping")
                continue
            # Save image to file
            image.save(image_path)

            # Store results in dictionary
            results[clean_name] = {
                "phrase": phrase,
                "prompt": prompt,
                "image_path": image_path,
            }

        except Exception as e:
            print(f"Error processing phrase '{phrase}': {str(e)}")
            continue

    return results


def add_image_paths(story_dict: Dict[str, Any], image_dir: str) -> Dict[str, Any]:
    """
    Add image paths to the story dictionary based on the English phrases.

    Args:
        story_dict: Dictionary containing story data with translated_phrase_list
        image_dir: Directory containing the images

    Returns:
        Updated dictionary with image_path added for each story part

    Note:
        For each story part, expects translated_phrase_list to be a list of tuples
        where each tuple is (english_text, target_text)
    """
    # Create a deep copy of the dictionary to avoid modifying nested structures
    updated_dict = copy.deepcopy(story_dict)

    for story_part, data in tqdm(updated_dict.items(), desc="Processing story parts"):
        # Initialize image_path list for this story part
        data["image_path"] = []

        # Get the phrases from translated_phrase_list
        phrase_list = data.get("translated_phrase_list", [])

        for eng_phrase, _ in tqdm(
            phrase_list, desc=f"Adding image paths for {story_part}", leave=False
        ):
            # Generate the expected image filename from English phrase
            clean_name = clean_filename(eng_phrase)
            image_filename = f"{clean_name}.png"
            full_path = os.path.join(image_dir, image_filename)

            # Check if the image exists and is readable
            if os.path.isfile(full_path) and os.access(full_path, os.R_OK):
                data["image_path"].append(full_path)
            else:
                print(f"Warning: Image not found or not readable: {full_path}")
                data["image_path"].append(None)

        # Verify lengths match
        if len(data["image_path"]) != len(phrase_list):
            raise ValueError(
                f"Mismatch in {story_part}: {len(data['image_path'])} images "
                f"vs {len(phrase_list)} phrases"
            )

    return updated_dict


def generate_story_image(story_dialogue: str):
    """
    Generate an image for a story using Google Cloud Vertex AI's Image Generation API.

    :param story_plan: A string containing the story plan
    :param project_id: Your Google Cloud project ID
    :param location: The location of your Vertex AI endpoint
    :return: Image data as bytes
    """
    # Initialize Vertex AI
    vertexai.init(project=config.PROJECT_ID, location=config.VERTEX_REGION)

    # Initialize the Image Generation model
    generation_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

    # Craft the prompt
    prompt = f"""
    Create a colorful, engaging image for a language learning story. 
    The image should be suitable as album art for an educational audio file.
    The story dialogue is: {story_dialogue:str}
    The image should be family-friendly and appropriate for all ages.
    The style should be hand-painted (not a photo).
    """

    ok_to_query_api()
    # Generate the image
    images = generation_model.generate_images(
        prompt=prompt,
        negative_prompt="people",
        number_of_images=1,
        aspect_ratio="1:1",
        # safety_filter_level="block_some",
        person_generation="don't allow",
    )

    # Get the first (and only) generated image
    generated_image = images[0]

    image_data = resize_image(generated_image, 500, 500)
    # Get the image bytes directly

    return image_data
